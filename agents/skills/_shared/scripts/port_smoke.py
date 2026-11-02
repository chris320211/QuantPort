#!/usr/bin/env python3
"""Behavioral smoke test for a port overlay on a 2-layer slice of the real model.

``slice`` cuts the first N decoder layers (real weights, so activation
outliers survive) out of the gather snapshot once per slug. ``run`` applies an
overlay to a temp worktree, runs its wrapper against that slice through the
smoke hooks, reloads the saved artifact the way verify does, and compares
logits with the fp16 slice. The gate: wrapper exits 0, the artifact reloads,
logits are finite, every target linear is quantized, and relative logit error
is under the spec's limit. Runs on the job GPU; a lock keeps smokes serial.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import re
import shutil
import struct
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))

import env
import overlay as overlay_mod
import paths
from arch_probe import log_tail, run_in_venv, wants_remote_code
from io_utils import atomic_write_text
from paths import venv_python
from request import load_request
from validate_script import validate_script
from verify import LOADER_SOURCE

DEFAULT_LAYERS = 2
LAYER_COUNT_KEYS = ("num_hidden_layers", "n_layer", "num_layers", "n_layers")
# Decoder-layer containers in HF checkpoints: model.layers.N, transformer.h.N, ...
LAYER_KEY_PATTERN = r"(?:^|\.)(?:layers|h|blocks|layer|decoder_layers)\.(\d+)\."
_LAYER_KEY_RE = re.compile(LAYER_KEY_PATTERN)
_WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".gguf", ".msgpack", ".h5"}
_WEIGHT_INDEXES = {"model.safetensors.index.json", "pytorch_model.bin.index.json"}
# MoE routers stay in high precision in every method we have seen.
DEFAULT_EXCLUDE_REGEX = r"(\.router|\.gate|shared_expert_gate)$"
COVERED_CLASSES = ("packed", "fake_quant", "wrapped")


class SmokeSettings(BaseModel):
    """Thresholds every candidate is judged by. Lives in the port spec's ``smoke`` block."""

    weight_bits: int = Field(8, ge=1, le=16)
    group_size: int | None = Field(None, ge=1)
    target_regex: str | None = None
    exclude_regex: str | None = DEFAULT_EXCLUDE_REGEX
    max_rel_error: float = Field(0.5, gt=0)
    min_coverage: float = Field(1.0, ge=0, le=1)
    calib_samples: int = Field(8, ge=1)
    seq_len: int = Field(256, ge=16)


SLICE_SCRIPT = r'''#!/usr/bin/env python
"""Copy the first N decoder layers' tensors (plus embeddings/norm/head) to one file."""
from __future__ import annotations

import json
import os
import re
import sys

import torch


def main():
    plan = json.loads(os.environ["SMOKE_SLICE_PLAN"])
    layer_re = re.compile(plan["layer_regex"])
    keep_layers = int(plan["num_layers"])

    def keep(key):
        match = layer_re.search(key)
        return match is None or int(match.group(1)) < keep_layers

    tensors = {}
    seen = set()
    for path in plan["files"]:
        if path.endswith(".safetensors"):
            from safetensors import safe_open
            with safe_open(path, framework="pt") as handle:
                items = [(key, handle.get_tensor(key)) for key in handle.keys() if keep(key)]
        else:
            state = torch.load(path, map_location="cpu", weights_only=True)
            items = [(key, value) for key, value in state.items() if keep(key)]
        for key, tensor in items:
            tensor = tensor.contiguous()
            if tensor.data_ptr() in seen:
                tensor = tensor.clone()
            seen.add(tensor.data_ptr())
            tensors[key] = tensor
    from safetensors.torch import save_file
    save_file(tensors, os.path.join(plan["dest"], "model.safetensors"), metadata={"format": "pt"})
    print(json.dumps({"tensors": len(tensors)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

CHECK_SCRIPT = r'''#!/usr/bin/env python
"""Reload a smoke artifact, compare logits with the fp16 slice, inspect target linears."""
from __future__ import annotations

import gc
import importlib.util
import json
import math
import os
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TEXT = (
    "The tower is 324 metres tall, about the same height as an 81-storey building, "
    "and the tallest structure in Paris. Its base is square, measuring 125 metres on "
    "each side. During its construction, the Eiffel Tower surpassed the Washington "
    "Monument to become the tallest man-made structure in the world, a title it held "
    "for 41 years until the Chrysler Building in New York City was finished in 1930. "
    "In 1957 an aerial was added to the top, and the tower now stands taller than the "
    "Chrysler Building by 5.2 metres. Excluding transmitters, it is the second tallest "
    "free-standing structure in France after the Millau Viaduct. Robert Boyle wrote that "
    "the spring of the air could be measured; Ada Lovelace described an engine that "
    "might compose elaborate pieces of music; def fib(n): return n if n < 2 else "
    "fib(n - 1) + fib(n - 2). Prices rose 3.4% in March, while 17 of 20 regions reported "
    "growth above 2 percent, and the committee met on 2024-06-11 at 09:30 UTC."
)
''' + LOADER_SOURCE + r'''

def _layer_container(model):
    best, best_len = None, 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and not re.search(r"\.\d+(\.|$)", name):
            if len(module) > best_len:
                best, best_len = name, len(module)
    return best


def _targets(model):
    target_re = os.environ.get("SMOKE_TARGET_REGEX") or None
    exclude_re = os.environ.get("SMOKE_EXCLUDE_REGEX") or None
    container = _layer_container(model)
    names = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if target_re:
            if not re.search(target_re, name):
                continue
        elif not container or not name.startswith(container + "."):
            continue
        if exclude_re and re.search(exclude_re, name):
            continue
        names.append(name)
    return names


def _logits(model, input_ids):
    device = model.get_input_embeddings().weight.device
    with torch.inference_mode():
        out = model(input_ids=input_ids.to(device))
    logits = out.logits if hasattr(out, "logits") else out[0]
    return logits.float().cpu()


def _module_stats(module):
    tensors = list(module.parameters()) + list(module.buffers())
    int_storage = any(
        not t.is_floating_point() and not t.is_complex() and t.dtype != torch.bool and t.numel() > 1
        for t in tensors
    )
    weight = None
    for pname, param in list(module.named_parameters()) + list(module.named_buffers()):
        if pname.split(".")[-1] == "weight" and param.dim() == 2 and param.is_floating_point():
            weight = param
            break
    unique_max = None
    in_features = None
    if weight is not None:
        rows = weight.detach()[: min(64, weight.shape[0])].float().cpu()
        unique_max = max(int(torch.unique(row).numel()) for row in rows)
        in_features = int(weight.shape[1])
    return {
        "cls": type(module).__name__,
        "is_plain_linear": type(module) is torch.nn.Linear,
        "int_storage": bool(int_storage),
        "unique_per_row_max": unique_max,
        "in_features": in_features,
    }


def main():
    ref_dir = os.environ["SMOKE_REF_DIR"]
    artifact = os.environ["MEASURE_MODEL_PATH"]
    model_id = os.environ.get("MEASURE_MODEL_ID", artifact)
    adapter_path = os.environ.get("MEASURE_ADAPTER_PATH") or None
    output_json = os.environ["MEASURE_OUTPUT_JSON"]
    trust = os.environ.get("MEASURE_TRUST_REMOTE_CODE") == "1"
    seq_len = int(os.environ.get("QUANT_AGENT_SMOKE_SEQ_LEN") or 256)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(ref_dir, trust_remote_code=trust, local_files_only=True)
    input_ids = tokenizer(TEXT, return_tensors="pt").input_ids[:, :seq_len]

    ref = AutoModelForCausalLM.from_pretrained(
        ref_dir,
        trust_remote_code=trust,
        local_files_only=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()
    targets = _targets(ref)
    ref_logits = _logits(ref, input_ids)
    del ref
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model, _ = _load_model_and_tokenizer(artifact, model_id, adapter_path, "float16", trust)
    q_logits = _logits(model, input_ids)
    modules = dict(model.named_modules())
    stats = {name: (_module_stats(modules[name]) if name in modules else None) for name in targets}

    finite = bool(torch.isfinite(q_logits).all())
    payload = {
        "seq_len": int(input_ids.shape[-1]),
        "finite": finite,
        "shape_match": list(q_logits.shape) == list(ref_logits.shape),
        "targets": stats,
        "rel_logit_error": None,
        "top1_agreement": None,
        "kl": None,
        "peak_vram_gb": (torch.cuda.max_memory_allocated() / (1024 ** 3)) if device == "cuda" else None,
        "cuda": device == "cuda",
        "artifact_files": sorted(os.listdir(artifact)),
    }
    if finite and payload["shape_match"]:
        diff = (q_logits - ref_logits).norm().item()
        payload["rel_logit_error"] = diff / max(ref_logits.norm().item(), 1e-12)
        payload["top1_agreement"] = (q_logits.argmax(-1) == ref_logits.argmax(-1)).float().mean().item()
        log_p = torch.log_softmax(ref_logits, dim=-1)
        log_q = torch.log_softmax(q_logits, dim=-1)
        kl = (log_p.exp() * (log_p - log_q)).sum(-1).mean().item()
        payload["kl"] = kl if math.isfinite(kl) else None
    tmp = output_json + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp, output_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


# --- slice ------------------------------------------------------------------


def truncate_config(config: dict, num_layers: int) -> dict:
    """Same config with ``num_layers`` decoder layers; per-layer lists are cut to match."""
    out = json.loads(json.dumps(config))
    found = False
    for scope in (out, out.get("text_config")):
        if not isinstance(scope, dict):
            continue
        for key in LAYER_COUNT_KEYS:
            original = scope.get(key)
            if isinstance(original, int) and original > 0:
                found = True
                keep = min(num_layers, original)
                scope[key] = keep
                for name, value in list(scope.items()):
                    if isinstance(value, list) and len(value) == original:
                        scope[name] = value[:keep]
                break
    if not found:
        raise ValueError("config has no decoder layer count (num_hidden_layers / n_layer / num_layers)")
    return out


def layer_of(key: str) -> int | None:
    match = _LAYER_KEY_RE.search(key)
    return int(match.group(1)) if match else None


def keep_key(key: str, num_layers: int) -> bool:
    layer = layer_of(key)
    return layer is None or layer < num_layers


def safetensors_keys(path: Path) -> list[str]:
    """Tensor names from a safetensors header, without torch."""
    with path.open("rb") as handle:
        (size,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(size))
    return [key for key in header if key != "__metadata__"]


def weight_files(snapshot: Path) -> list[Path]:
    files = sorted(snapshot.glob("*.safetensors"))
    if files:
        return files
    return sorted(p for p in snapshot.glob("*.bin") if p.name.startswith("pytorch_model"))


def slice_dir_for(slug: str, num_layers: int) -> Path:
    return paths.SMOKE_SLICES / f"{paths.require_slug(slug)}-L{num_layers}"


def build_slice(request: dict, *, num_layers: int = DEFAULT_LAYERS, force: bool = False, timeout_s: int = 1200) -> dict:
    slug = paths.require_slug(str(request["slug"]))
    snapshot = Path(request["hf_snapshot_path"]).expanduser().resolve()
    dest = slice_dir_for(slug, num_layers)
    marker = dest / "slice.json"
    if marker.is_file() and not force:
        record = json.loads(marker.read_text())
        if record.get("source_snapshot") == str(snapshot) and record.get("num_layers") == num_layers:
            return {"status": "ok", "reused": True, **record}
    config = json.loads((snapshot / "config.json").read_text())
    truncated = truncate_config(config, num_layers)
    files = weight_files(snapshot)
    if not files:
        raise RuntimeError(f"snapshot has no safetensors or pytorch_model*.bin weights: {snapshot}")
    kept = None
    if all(p.suffix == ".safetensors" for p in files):
        # Only read the shards that hold a kept tensor.
        needed = []
        kept = 0
        for path in files:
            keys = [key for key in safetensors_keys(path) if keep_key(key, num_layers)]
            if keys:
                needed.append(path)
                kept += len(keys)
        files = needed
    py = venv_python(slug)
    if not py.exists():
        raise RuntimeError(f"Venv python not found at {py}. Run install_venv for this slug first.")
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    for item in snapshot.iterdir():
        if item.is_file() and item.suffix not in _WEIGHT_SUFFIXES and item.name not in _WEIGHT_INDEXES:
            shutil.copy2(item, dest / item.name)
    atomic_write_text(dest / "config.json", json.dumps(truncated, indent=2) + "\n", mode=0o644)
    plan = {
        "files": [str(p.resolve()) for p in files],
        "dest": str(dest),
        "layer_regex": LAYER_KEY_PATTERN,
        "num_layers": num_layers,
    }
    code, log = run_in_venv(
        venv_py=py,
        source=SLICE_SCRIPT,
        workdir=dest / ".slice",
        name="slice",
        extra={"SMOKE_SLICE_PLAN": json.dumps(plan)},
        timeout_s=timeout_s,
    )
    if code != 0 or not (dest / "model.safetensors").is_file():
        raise RuntimeError(f"slice failed (exit={code}). Last log lines:\n{log_tail(log)}")
    shutil.rmtree(dest / ".slice", ignore_errors=True)
    original = next((config[k] for k in LAYER_COUNT_KEYS if isinstance(config.get(k), int)), None)
    record = {
        "slice_dir": str(dest),
        "source_snapshot": str(snapshot),
        "num_layers": num_layers,
        "original_layers": original,
        "kept_tensors": kept,
    }
    atomic_write_text(marker, json.dumps(record, indent=2) + "\n", mode=0o644)
    return {"status": "ok", "reused": False, **record}


# --- gate (pure) --------------------------------------------------------------


def classify(stat: dict | None, *, weight_bits: int, group_size: int | None) -> str:
    """packed | fake_quant | wrapped | float | missing for one target linear."""
    if stat is None:
        return "missing"
    if stat.get("int_storage"):
        return "packed"
    unique = stat.get("unique_per_row_max")
    in_features = stat.get("in_features")
    if isinstance(unique, int) and isinstance(in_features, int) and in_features > 0:
        groups = math.ceil(in_features / group_size) if group_size else 1
        if unique <= (2 ** weight_bits) * groups and unique < in_features:
            return "fake_quant"
    if not stat.get("is_plain_linear", True):
        return "wrapped"
    return "float"


def evaluate(check: dict, settings: SmokeSettings) -> dict:
    """Pass/fail plus the metrics ranking uses. Pure: no torch, no files."""
    targets = check.get("targets") or {}
    classes = {
        name: classify(stat, weight_bits=settings.weight_bits, group_size=settings.group_size)
        for name, stat in targets.items()
    }
    counts = {label: 0 for label in (*COVERED_CLASSES, "float", "missing")}
    for label in classes.values():
        counts[label] += 1
    covered = sum(counts[label] for label in COVERED_CLASSES)
    coverage = covered / len(classes) if classes else 0.0
    reasons: list[str] = []
    if not classes:
        reasons.append("no_target_linears")
    if not check.get("finite"):
        reasons.append("non_finite_logits")
    if not check.get("shape_match", True):
        reasons.append("logit_shape_mismatch")
    rel = check.get("rel_logit_error")
    if rel is not None and rel > settings.max_rel_error:
        reasons.append(f"rel_logit_error {rel:.3f} > {settings.max_rel_error}")
    if classes and coverage < settings.min_coverage:
        reasons.append(f"coverage {coverage:.2f} < {settings.min_coverage}")
    uncovered = [name for name, label in classes.items() if label not in COVERED_CLASSES]
    return {
        "status": "passed" if not reasons else "failed",
        "stage": "check",
        "reasons": reasons,
        "metrics": {
            "coverage": coverage,
            "rel_logit_error": rel,
            "top1_agreement": check.get("top1_agreement"),
            "kl": check.get("kl"),
            "targets": len(classes),
            "classes": counts,
        },
        "uncovered": uncovered[:20],
    }


# --- run ----------------------------------------------------------------------


def smoke_dir(slug: str, overlay_dir: Path) -> Path:
    """out/smoke/<slug>/<plan>/<hash>/ for out/overlays/<slug>/<plan>/<hash>/."""
    overlay_dir = overlay_dir.resolve()
    return paths.SMOKE_ROOT / paths.require_slug(slug) / overlay_dir.parent.name / overlay_dir.name


def read_result(slug: str, overlay_dir: Path) -> dict | None:
    path = smoke_dir(slug, overlay_dir) / "smoke.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def load_settings(spec_path: Path | None, overrides: dict) -> SmokeSettings:
    base: dict = {}
    if spec_path is not None and spec_path.is_file():
        spec = json.loads(spec_path.read_text())
        if isinstance(spec, dict) and isinstance(spec.get("smoke"), dict):
            base = dict(spec["smoke"])
    base.update({k: v for k, v in overrides.items() if v is not None})
    return SmokeSettings.model_validate(base)


@contextmanager
def _gpu_lock(slug: str):
    lock_path = paths.SMOKE_ROOT / slug / ".gpu.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run_smoke(
    request: dict,
    *,
    overlay_dir: Path,
    script: Path,
    settings: SmokeSettings,
    num_layers: int = DEFAULT_LAYERS,
    timeout_s: int = 1800,
    keep_artifact: bool = False,
) -> dict:
    env.require_host_execution("port smoke")
    slug = paths.require_slug(str(request["slug"]))
    model_id = str(request["model_id"])
    overlay_dir = overlay_dir.expanduser().resolve()
    manifest = overlay_mod.validate_overlay_bundle(overlay_dir)
    if manifest.method_id != slug or manifest.model_id != model_id:
        raise RuntimeError(f"overlay identity does not match request: {manifest.method_id}/{manifest.model_id}")
    code = script.read_text()
    checked = validate_script(
        code,
        model_id=model_id,
        output_dir=f"./quantized/{slug}",
        overlay_dir=overlay_dir,
        require_smoke_hooks=True,
    )
    if not checked["ok"]:
        raise RuntimeError(f"wrapper failed validation ({checked['stage']}): {checked['message']}")
    slice_path = slice_dir_for(slug, num_layers)
    if not (slice_path / "slice.json").is_file():
        raise RuntimeError(f"no smoke slice at {slice_path}; run port_smoke.py slice first")
    py = venv_python(slug)
    if not py.exists():
        raise RuntimeError(f"Venv python not found at {py}. Run install_venv for this slug first.")
    repo = Path(request["repo_path"]).expanduser().resolve()
    trust = "1" if wants_remote_code(Path(request["hf_snapshot_path"]).expanduser()) else "0"

    dest = smoke_dir(slug, overlay_dir)
    artifact = dest / "artifact"
    result: dict = {
        "schema_version": 1,
        "overlay_dir": str(overlay_dir),
        "script": str(script.resolve()),
        "plan": overlay_dir.parent.name,
        "settings": settings.model_dump(),
        "slice_layers": num_layers,
    }
    with _gpu_lock(slug):
        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True)
        worktree = overlay_mod.prepare_overlay_worktree(
            repo=repo,
            overlay_manifest=manifest,
            overlay_snapshot=overlay_dir,
            worktree=dest / "method-repo",
        )
        try:
            started = time.monotonic()
            exit_code, log = run_in_venv(
                venv_py=py,
                source=code,
                workdir=dest,
                name="quantize",
                extra={
                    "QUANT_AGENT_OVERLAY_DIR": str(overlay_dir),
                    "QUANT_AGENT_METHOD_REPO": str(worktree),
                    "PYTHONPATH": str(worktree),
                    "QUANT_AGENT_SMOKE": "1",
                    "QUANT_AGENT_SMOKE_MODEL_DIR": str(slice_path),
                    "QUANT_AGENT_SMOKE_OUTPUT_DIR": str(artifact),
                    "QUANT_AGENT_SMOKE_CALIB_SAMPLES": str(settings.calib_samples),
                    "QUANT_AGENT_SMOKE_SEQ_LEN": str(settings.seq_len),
                    "PYTHONUNBUFFERED": "1",
                },
                timeout_s=timeout_s,
                include_hf=True,
            )
            result["quantize_seconds"] = round(time.monotonic() - started, 1)
            if exit_code != 0:
                result.update(status="failed", stage="quantize", reasons=[f"wrapper exit {exit_code}"], log_tail=log_tail(log))
            elif not artifact.is_dir() or not any(artifact.iterdir()):
                result.update(status="failed", stage="save", reasons=["wrapper wrote nothing to QUANT_AGENT_SMOKE_OUTPUT_DIR"], log_tail=log_tail(log))
            else:
                check_json = dest / "check.json"
                extra = {
                    "SMOKE_REF_DIR": str(slice_path),
                    "MEASURE_MODEL_PATH": str(artifact),
                    "MEASURE_MODEL_ID": model_id,
                    "MEASURE_OUTPUT_JSON": str(check_json),
                    "MEASURE_DTYPE": "float16",
                    "MEASURE_TRUST_REMOTE_CODE": trust,
                    "QUANT_AGENT_METHOD_REPO": str(worktree),
                    "PYTHONPATH": str(worktree),
                    "QUANT_AGENT_SMOKE_SEQ_LEN": str(settings.seq_len),
                    "SMOKE_EXCLUDE_REGEX": settings.exclude_regex or "",
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                }
                if settings.target_regex:
                    extra["SMOKE_TARGET_REGEX"] = settings.target_regex
                if manifest.inference_adapter_path:
                    extra["MEASURE_ADAPTER_PATH"] = str(worktree / manifest.inference_adapter_path)
                exit_code, check_log = run_in_venv(
                    venv_py=py,
                    source=CHECK_SCRIPT,
                    workdir=dest,
                    name="check",
                    extra=extra,
                    timeout_s=600,
                    cwd=worktree,
                )
                if exit_code != 0 or not check_json.is_file():
                    result.update(status="failed", stage="reload", reasons=[f"reload/compare exit {exit_code}"], log_tail=log_tail(check_log))
                else:
                    check = json.loads(check_json.read_text())
                    result.update(evaluate(check, settings))
                    result["peak_vram_gb"] = check.get("peak_vram_gb")
                    result["artifact_files"] = check.get("artifact_files")
        finally:
            overlay_mod.cleanup_overlay_worktree(worktree, repo)
            if not keep_artifact:
                shutil.rmtree(artifact, ignore_errors=True)
    atomic_write_text(dest / "smoke.json", json.dumps(result, indent=2) + "\n", mode=0o644)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="2-layer behavioral smoke test for port overlays")
    sub = parser.add_subparsers(dest="command", required=True)

    slice_p = sub.add_parser("slice", help="Cut the first N layers of the snapshot (once per slug)")
    slice_p.add_argument("--request", type=Path, required=True)
    slice_p.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    slice_p.add_argument("--force", action="store_true")

    run_p = sub.add_parser("run", help="Quantize the slice with an overlay, reload, compare")
    run_p.add_argument("--request", type=Path, required=True)
    run_p.add_argument("--overlay-dir", type=Path, required=True)
    run_p.add_argument("--script", type=Path, help="default <overlay-dir>/../quantize.py")
    run_p.add_argument("--spec", type=Path, help="default out/analysis/<slug>/differences.json")
    run_p.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    run_p.add_argument("--weight-bits", type=int)
    run_p.add_argument("--group-size", type=int)
    run_p.add_argument("--target-regex")
    run_p.add_argument("--max-rel-error", type=float)
    run_p.add_argument("--min-coverage", type=float)
    run_p.add_argument("--timeout", type=int, default=1800)
    run_p.add_argument("--keep-artifact", action="store_true")
    run_p.add_argument("--allow-unsafe-host-execution", action="store_true")

    args = parser.parse_args()
    try:
        request = load_request(args.request)
        if args.command == "slice":
            payload = build_slice(request, num_layers=args.layers, force=args.force)
        else:
            slug = str(request["slug"])
            spec = args.spec or paths.ANALYSIS_ROOT / slug / "differences.json"
            settings = load_settings(spec, {
                "weight_bits": args.weight_bits,
                "group_size": args.group_size,
                "target_regex": args.target_regex,
                "max_rel_error": args.max_rel_error,
                "min_coverage": args.min_coverage,
            })
            script = args.script or args.overlay_dir.expanduser().resolve().parent / "quantize.py"
            with env.host_execution_policy(args.allow_unsafe_host_execution):
                payload = run_smoke(
                    request,
                    overlay_dir=args.overlay_dir,
                    script=script,
                    settings=settings,
                    num_layers=args.layers,
                    timeout_s=args.timeout,
                    keep_artifact=args.keep_artifact,
                )
    except (ValueError, RuntimeError, OSError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, indent=2))
        return 1
    print(json.dumps(payload, indent=2))
    return 0 if payload.get("status") in ("ok", "passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())

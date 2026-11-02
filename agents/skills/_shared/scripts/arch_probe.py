#!/usr/bin/env python3
"""Inventory a model's structure on the meta device (no weights) for porting.

The child runs in the method venv, so it sees the transformers version the
method will run with, builds the model from its config without allocating
weights, and dumps every module. The host turns that dump into the summary a
port planner compares against what the method repo handles: layer layout,
linears and their biases, norms, tied embeddings, attention variant, MoE.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import env
import paths
from io_utils import atomic_write_text
from paths import venv_python
from request import load_request

PROBE_SCRIPT = r'''#!/usr/bin/env python
"""Build the model on the meta device and dump its modules. No weights load."""
from __future__ import annotations

import json
import os
import sys

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM


def main():
    model_path = os.environ["MEASURE_MODEL_PATH"]
    output_json = os.environ["MEASURE_OUTPUT_JSON"]
    trust = os.environ.get("MEASURE_TRUST_REMOTE_CODE") == "1"
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust, local_files_only=True)
    try:
        from accelerate import init_empty_weights
        context = init_empty_weights()
    except ImportError:
        context = torch.device("meta")
    with context:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust)
    try:
        model.tie_weights()
    except Exception:
        pass
    modules = []
    for name, module in model.named_modules():
        entry = {"name": name, "cls": type(module).__name__}
        params = {
            pname: list(param.shape)
            for pname, param in module.named_parameters(recurse=False)
        }
        if params:
            entry["params"] = params
        for attr in ("in_features", "out_features", "num_embeddings", "embedding_dim"):
            value = getattr(module, attr, None)
            if isinstance(value, int):
                entry[attr] = value
        modules.append(entry)
    inputs = model.get_input_embeddings()
    outputs = model.get_output_embeddings()
    tied = bool(
        inputs is not None and outputs is not None
        and getattr(outputs, "weight", None) is getattr(inputs, "weight", None)
    )
    payload = {
        "model_class": type(model).__name__,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "attn_implementation": getattr(config, "_attn_implementation", None),
        "tied_embeddings": tied,
        "config": json.loads(json.dumps(config.to_dict(), default=str)),
        "modules": modules,
    }
    tmp = output_json + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(payload, handle)
    os.replace(tmp, output_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

# Config keys a port planner needs; the full config stays in the snapshot.
CONFIG_KEYS = (
    "model_type", "architectures", "hidden_size", "intermediate_size",
    "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
    "vocab_size", "max_position_embeddings", "tie_word_embeddings", "hidden_act",
    "hidden_activation", "rms_norm_eps", "layer_norm_eps", "rope_theta", "rope_scaling",
    "partial_rotary_factor", "sliding_window", "use_sliding_window", "layer_types",
    "attn_logit_softcapping", "final_logit_softcapping", "query_pre_attn_scalar",
    "attention_bias", "mlp_bias", "num_experts", "num_local_experts", "n_routed_experts",
    "num_experts_per_tok", "moe_intermediate_size", "shared_expert_intermediate_size",
    "torch_dtype", "quantization_config", "auto_map",
)

_INDEXED_RE = re.compile(r"^(?P<prefix>.+)\.(?P<idx>\d+)$")
_NUMERIC_SEGMENT_RE = re.compile(r"\.\d+(\.|$)")
_EXPERT_RE = re.compile(r"(^|\.)experts\.\d+(\.|$)")


def run_in_venv(
    *,
    venv_py: Path,
    source: str,
    workdir: Path,
    name: str,
    extra: dict[str, str],
    timeout_s: int,
    include_hf: bool = False,
    cwd: Path | None = None,
) -> tuple[int, Path]:
    """Write ``source`` to ``workdir/<name>.py`` and run it; return (exit, log path)."""
    workdir.mkdir(parents=True, exist_ok=True)
    script = workdir / f"{name}.py"
    script.write_text(source)
    log_path = workdir / f"{name}.log"
    child = env.child_env(
        extra,
        include_hf=include_hf,
        scratch_home=None if include_hf else workdir / "hf-scratch",
    )
    with log_path.open("wb") as logf:
        try:
            proc = subprocess.run(
                [str(venv_py), str(script)],
                stdout=logf,
                stderr=subprocess.STDOUT,
                env=child,
                timeout=timeout_s,
                check=False,
                cwd=str(cwd or paths.REPO_ROOT),
            )
            code = proc.returncode
        except subprocess.TimeoutExpired:
            logf.write(f"\n[timeout after {timeout_s}s]\n".encode())
            code = 124
    return code, log_path


def log_tail(path: Path, n: int = 30) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def wants_remote_code(snapshot: Path) -> bool:
    if any(snapshot.glob("modeling_*.py")):
        return True
    try:
        config = json.loads((snapshot / "config.json").read_text())
    except (OSError, ValueError):
        return False
    return isinstance(config, dict) and bool(config.get("auto_map"))


def find_layer_container(modules: list[dict]) -> str | None:
    """The outermost indexed module list with the most entries (decoder layers)."""
    counts: Counter[str] = Counter()
    for entry in modules:
        match = _INDEXED_RE.match(entry["name"])
        if match and not _NUMERIC_SEGMENT_RE.search(match.group("prefix")):
            counts[match.group("prefix")] += 1
    if not counts:
        return None
    return max(counts, key=lambda prefix: (counts[prefix], -len(prefix)))


def _layer_entries(modules: list[dict], container: str, idx: int) -> list[dict]:
    base = f"{container}.{idx}"
    out = []
    for entry in modules:
        name = entry["name"]
        if name == base or name.startswith(base + "."):
            rel = name[len(base) + 1:] if name != base else ""
            out.append({**entry, "rel": rel})
    return out


def _signature(entries: list[dict]) -> tuple:
    collapsed = set()
    for entry in entries:
        rel = _EXPERT_RE.sub(r"\1experts.*\2", entry["rel"])
        shapes = tuple(sorted((k, tuple(v)) for k, v in (entry.get("params") or {}).items()))
        collapsed.add((rel, entry["cls"], shapes))
    return tuple(sorted(collapsed))


def _is_linear(entry: dict) -> bool:
    return isinstance(entry.get("in_features"), int) and isinstance(entry.get("out_features"), int)


def _is_norm(entry: dict) -> bool:
    return "norm" in entry["cls"].lower() and bool(entry.get("params"))


def summarize(raw: dict, *, model_id: str = "", snapshot: str = "") -> dict:
    """Structure summary plus ``flags``: where this model departs from a plain Llama decoder."""
    modules = raw.get("modules") or []
    config_full = raw.get("config") or {}
    config = {key: config_full[key] for key in CONFIG_KEYS if config_full.get(key) is not None}
    container = find_layer_container(modules)
    num_layers = 0
    groups: dict[tuple, list[int]] = {}
    layer0: list[dict] = []
    if container:
        indices = sorted({
            int(m.group("idx"))
            for entry in modules
            if (m := _INDEXED_RE.match(entry["name"])) and m.group("prefix") == container
        })
        num_layers = len(indices)
        for idx in indices:
            entries = _layer_entries(modules, container, idx)
            groups.setdefault(_signature(entries), []).append(idx)
        layer0 = _layer_entries(modules, container, indices[0]) if indices else []

    by_rel = {entry["rel"]: entry for entry in layer0}
    attention = next(
        (e for e in sorted(layer0, key=lambda e: len(e["rel"])) if e["rel"] and "attention" in e["cls"].lower()),
        None,
    )
    attn_rel = attention["rel"] if attention else None

    def in_attention(rel: str) -> bool:
        return bool(attn_rel) and (rel == attn_rel or rel.startswith(attn_rel + "."))

    expert_ids = {
        m.group(0) for e in layer0 if (m := re.search(r"experts\.\d+", e["rel"]))
    }
    linears = []
    seen_linear = set()
    for entry in layer0:
        if not _is_linear(entry):
            continue
        rel = _EXPERT_RE.sub(r"\1experts.*\2", entry["rel"])
        if rel in seen_linear:
            continue
        seen_linear.add(rel)
        linears.append({
            "name": rel,
            "cls": entry["cls"],
            "in_features": entry["in_features"],
            "out_features": entry["out_features"],
            "bias": "bias" in (entry.get("params") or {}),
            "in_attention": in_attention(entry["rel"]),
        })
    norms = [
        {
            "name": entry["rel"],
            "cls": entry["cls"],
            "bias": "bias" in (entry.get("params") or {}),
            "in_attention": in_attention(entry["rel"]),
        }
        for entry in layer0
        if _is_norm(entry)
    ]
    top_level = [
        {k: v for k, v in entry.items() if k != "rel"}
        for entry in modules
        if entry["name"] and (not container or not entry["name"].startswith(container + "."))
        and entry["name"] != container
        and (entry.get("params") or _is_linear(entry))
    ]

    flags: list[str] = []
    attn_linears = [lin for lin in linears if lin["in_attention"]]
    mlp_linears = [lin for lin in linears if not lin["in_attention"]]
    if any(lin["bias"] for lin in attn_linears):
        flags.append("attention_bias")
    if any(lin["bias"] for lin in mlp_linears):
        flags.append("mlp_bias")
    names = " ".join(lin["name"].lower() for lin in linears)
    if any(key in names for key in ("qkv", "query_key_value", "c_attn", "wqkv")):
        flags.append("fused_qkv")
    if "gate_up" in names or "w13" in names:
        flags.append("fused_gate_up")
    heads = config.get("num_attention_heads")
    kv_heads = config.get("num_key_value_heads")
    if isinstance(heads, int) and isinstance(kv_heads, int) and kv_heads < heads:
        flags.append("mqa" if kv_heads == 1 else "gqa")
    head_dim = config.get("head_dim")
    hidden = config.get("hidden_size")
    if isinstance(head_dim, int) and isinstance(heads, int) and isinstance(hidden, int) and head_dim * heads != hidden:
        flags.append("head_dim_not_hidden_over_heads")
    if raw.get("tied_embeddings"):
        flags.append("tied_embeddings")
    if any(norm["bias"] for norm in norms):
        flags.append("layernorm_with_bias")
    if any(norm["in_attention"] for norm in norms):
        flags.append("qk_norm")
    if len([norm for norm in norms if not norm["in_attention"]]) > 2:
        flags.append("extra_layer_norms")
    if config.get("attn_logit_softcapping"):
        flags.append("attn_softcapping")
    if config.get("final_logit_softcapping"):
        flags.append("final_logit_softcapping")
    if config.get("sliding_window") and config.get("use_sliding_window") is not False:
        flags.append("sliding_window")
    layer_types = config.get("layer_types")
    if isinstance(layer_types, list) and len(set(map(str, layer_types))) > 1:
        flags.append("mixed_layer_types")
    if expert_ids or any(config.get(k) for k in ("num_experts", "num_local_experts", "n_routed_experts")):
        flags.append("moe")
    partial = config.get("partial_rotary_factor")
    if isinstance(partial, (int, float)) and partial < 1:
        flags.append("partial_rotary")
    if config.get("rope_scaling"):
        flags.append("rope_scaling")
    if len(groups) > 1:
        flags.append("heterogeneous_layers")
    act = str(config.get("hidden_act") or config.get("hidden_activation") or "").lower()
    if act and act not in {"silu", "swish"}:
        flags.append("non_silu_activation")
    if config.get("quantization_config"):
        flags.append("prequantized")
    if config.get("auto_map"):
        flags.append("remote_code")

    return {
        "schema_version": 1,
        "model_id": model_id,
        "snapshot": snapshot,
        "model_class": raw.get("model_class"),
        "transformers_version": raw.get("transformers_version"),
        "torch_version": raw.get("torch_version"),
        "attn_implementation": raw.get("attn_implementation"),
        "config": config,
        "layer_container": container,
        "num_layers": num_layers,
        "layer_groups": [layers for layers in sorted(groups.values())],
        "layer0": {
            "attention": attention["cls"] if attention else None,
            "attention_path": attn_rel,
            "mlp": next(
                (by_rel[rel]["cls"] for rel in ("mlp", "feed_forward", "block_sparse_moe") if rel in by_rel),
                None,
            ),
            "num_experts": len(expert_ids),
            "linears": linears,
            "norms": norms,
        },
        "top_level": top_level,
        "tied_embeddings": bool(raw.get("tied_embeddings")),
        "flags": flags,
    }


def probe(request: dict, *, out: Path, timeout_s: int = 600) -> dict:
    env.require_host_execution("arch probe")
    slug = paths.require_slug(str(request["slug"]))
    snapshot = Path(request["hf_snapshot_path"]).expanduser().resolve()
    if not (snapshot / "config.json").is_file():
        raise RuntimeError(f"snapshot has no config.json: {snapshot}")
    py = venv_python(slug)
    if not py.exists():
        raise RuntimeError(f"Venv python not found at {py}. Run install_venv for this slug first.")
    workdir = out.parent / "probe"
    raw_json = workdir / "raw.json"
    raw_json.unlink(missing_ok=True)
    code, log = run_in_venv(
        venv_py=py,
        source=PROBE_SCRIPT,
        workdir=workdir,
        name="probe",
        extra={
            "MEASURE_MODEL_PATH": str(snapshot),
            "MEASURE_OUTPUT_JSON": str(raw_json),
            "MEASURE_TRUST_REMOTE_CODE": "1" if wants_remote_code(snapshot) else "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        timeout_s=timeout_s,
    )
    if code != 0 or not raw_json.is_file():
        raise RuntimeError(f"arch probe failed (exit={code}). Last log lines:\n{log_tail(log)}")
    summary = summarize(
        json.loads(raw_json.read_text()),
        model_id=str(request["model_id"]),
        snapshot=str(snapshot),
    )
    atomic_write_text(out, json.dumps(summary, indent=2) + "\n", mode=0o644)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Meta-device structure inventory for porting")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="default out/analysis/<slug>/arch.json")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--allow-unsafe-host-execution", action="store_true")
    args = parser.parse_args()
    try:
        request = load_request(args.request)
        out = args.out or paths.ANALYSIS_ROOT / str(request["slug"]) / "arch.json"
        with env.host_execution_policy(args.allow_unsafe_host_execution):
            summary = probe(request, out=out, timeout_s=args.timeout)
    except (ValueError, RuntimeError, OSError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, indent=2))
        return 1
    print(json.dumps({
        "status": "ok",
        "path": str(out),
        "model_type": summary["config"].get("model_type"),
        "num_layers": summary["num_layers"],
        "flags": summary["flags"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

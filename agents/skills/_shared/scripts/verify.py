#!/usr/bin/env python3
"""Reload a completed job's saved artifact and require prompt generation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import env
import jobs as jobs_mod
import overlay as overlay_mod
import paths
from adapter import adapter_loads_hub_id, validate_adapter_file, validate_adapter_source
from paths import venv_python
from request import load_request

# Shared with port_smoke.py: load a saved artifact through its adapter, if any.
LOADER_SOURCE = r'''
def _load_model_and_tokenizer(model_path, model_id, adapter_path, dtype_name, trust_remote_code):
    dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16, "auto": "auto"}
    if dtype_name not in dtypes:
        raise ValueError(f"unsupported MEASURE_DTYPE={dtype_name!r}")
    method_repo = os.environ.get("QUANT_AGENT_METHOD_REPO")
    if method_repo:
        sys.path.insert(0, method_repo)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if adapter_path:
        spec = importlib.util.spec_from_file_location(
            "quant_agent_generated_inference_adapter", adapter_path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not import inference adapter: {adapter_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loader = getattr(module, "load_model_and_tokenizer", None)
        if not callable(loader):
            raise RuntimeError("inference adapter lacks load_model_and_tokenizer")
        loaded = loader(
            model_path=model_path,
            model_id=model_id,
            dtype=dtype_name,
            device=device,
            trust_remote_code=trust_remote_code,
        )
        if not isinstance(loaded, tuple) or len(loaded) != 2:
            raise RuntimeError("inference adapter must return (model, tokenizer)")
        model, tokenizer = loaded
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code, local_files_only=True
        )
        kwargs = {
            "trust_remote_code": trust_remote_code,
            "torch_dtype": dtypes[dtype_name] if device == "cuda" else torch.float32,
            "local_files_only": True,
        }
        if device == "cuda":
            kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        if device != "cuda":
            model.to(device)
    if not hasattr(model, "generate") or not hasattr(model, "get_input_embeddings"):
        raise RuntimeError("loaded adapter model does not implement the causal-LM interface")
    model.eval()
    return model, tokenizer
'''

SMOKE_SCRIPT = r'''#!/usr/bin/env python
"""Smoke-load a persisted quantized artifact and require alphanumeric generation."""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

''' + LOADER_SOURCE + r'''

def _input_device(model):
    return model.get_input_embeddings().weight.device


def _smoke_generation(model, tokenizer):
    prompt = "Reply with exactly one short word confirming readiness."
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            input_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            inputs = {"input_ids": input_ids.to(_input_device(model))}
        except (TypeError, ValueError, AttributeError):
            inputs = tokenizer(prompt, return_tensors="pt")
    else:
        inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(_input_device(model)) for key, value in inputs.items()}
    input_length = int(inputs["input_ids"].shape[-1])
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=pad_token_id,
        )
    new_tokens = output[0, input_length:]
    if int(new_tokens.numel()) < 1:
        raise RuntimeError("inference smoke generated zero new tokens")
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    if not output_text.strip() or not any(char.isalnum() for char in output_text):
        raise RuntimeError(
            f"inference smoke decoded only empty/non-alphanumeric text: {output_text!r}"
        )
    return {
        "prompt": prompt,
        "output_text": output_text,
        "generated_token_ids": [int(token) for token in new_tokens.detach().cpu().tolist()],
        "new_token_count": int(new_tokens.numel()),
    }


def main():
    model_path = os.environ["MEASURE_MODEL_PATH"]
    model_id = os.environ.get("MEASURE_MODEL_ID", model_path)
    adapter_path = os.environ.get("MEASURE_ADAPTER_PATH") or None
    output_json = os.environ["MEASURE_OUTPUT_JSON"]
    dtype_name = os.environ.get("MEASURE_DTYPE", "float16")
    trust = os.environ.get("MEASURE_TRUST_REMOTE_CODE") == "1"

    peak_vram_gb = None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model, tokenizer = _load_model_and_tokenizer(
        model_path, model_id, adapter_path, dtype_name, trust,
    )
    inference_smoke = _smoke_generation(model, tokenizer)
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    payload = {
        **inference_smoke,
        "model_path": model_path,
        "model_id": model_id,
        "adapter_path": adapter_path,
        "dtype": dtype_name,
        "peak_vram_gb": peak_vram_gb,
        "cuda": bool(torch.cuda.is_available()),
    }
    tmp = output_json + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, output_json)
    print("INFERENCE_RESULT=" + json.dumps(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


_WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".gguf", ".npz"}


def resolve_workspace_path(raw: str | Path) -> Path:
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = paths.REPO_ROOT / candidate
    return candidate.resolve()


def _output_missing_or_empty(output_dir: Path) -> bool:
    if not output_dir.exists():
        return True
    if output_dir.is_file():
        return output_dir.stat().st_size == 0
    return not any(path.is_file() for path in output_dir.rglob("*"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _weight_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    files = []
    for path in directory.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        if path.suffix.lower() in _WEIGHT_SUFFIXES:
            files.append(path)
    return files


def adapter_source_for_overlay(overlay_dir: Path, adapter_file: Path | None = None) -> str | None:
    manifest = overlay_mod.validate_overlay_bundle(overlay_dir)
    if manifest.inference_adapter_path is None:
        return None
    if adapter_file is not None and adapter_file.is_file():
        return adapter_file.read_text()
    patch = (overlay_dir / "overlay.patch").read_text()
    return overlay_mod.extract_new_file(patch, manifest.inference_adapter_path)


def check_authenticity(
    *,
    output_dir: str | Path,
    hf_snapshot_path: str | Path,
    adapter_source: str | None = None,
) -> None:
    """P0: the verified artifact must be the saved quantized weights, not the Hub snapshot."""
    resolved_output = resolve_workspace_path(output_dir)
    resolved_snapshot = resolve_workspace_path(hf_snapshot_path)
    if resolved_output == resolved_snapshot:
        raise RuntimeError(
            "output_dir resolves to the original Hugging Face snapshot; "
            "verify must reload the saved quantized artifact"
        )
    if _output_missing_or_empty(resolved_output):
        raise RuntimeError(f"output_dir is missing or empty: {resolved_output}")
    weights = _weight_files(resolved_output)
    if not weights:
        raise RuntimeError(
            f"output_dir has no weight files (.safetensors/.bin/.pt/.gguf/.npz): {resolved_output}"
        )
    matching = 0
    for weight in weights:
        rel = weight.relative_to(resolved_output)
        other = resolved_snapshot / rel
        if not other.is_file():
            continue
        wstat, ostat = weight.stat(), other.stat()
        if (wstat.st_dev, wstat.st_ino) == (ostat.st_dev, ostat.st_ino):
            raise RuntimeError(
                "output_dir weight file is hardlinked to the Hugging Face snapshot"
            )
        if _file_sha256(weight) == _file_sha256(other):
            matching += 1
    if matching == len(weights):
        raise RuntimeError("output_dir weights match the Hugging Face snapshot")
    if adapter_source:
        validate_adapter_source(adapter_source)
        if adapter_loads_hub_id(adapter_source):
            raise RuntimeError(
                "inference adapter loads Hub model_id instead of the saved artifact"
            )


@contextmanager
def materialize_inference_runtime(meta: jobs_mod.JobMeta, request: dict):
    """Yield ``(adapter_path, method_repo)`` for smoke inference."""
    repo = Path(request["repo_path"]).expanduser().resolve()
    if not meta.overlay_path:
        yield None, repo
        return
    overlay_snapshot = Path(meta.overlay_path).resolve()
    manifest = overlay_mod.validate_overlay_bundle(overlay_snapshot)
    if manifest.method_id != meta.method_id or manifest.model_id != meta.model_id:
        raise RuntimeError("job overlay identity does not match inference request")
    if manifest.inference_adapter_path is None:
        yield None, repo
        return

    job_directory = jobs_mod.job_dir(meta.job_id)
    worktree_name = f"inference-method-repo-{secrets.token_hex(3)}"
    worktree = job_directory / worktree_name
    overlay_mod._assert_not_venv_write(worktree)
    worktree = overlay_mod.prepare_overlay_worktree(
        repo=repo,
        overlay_manifest=manifest,
        overlay_snapshot=overlay_snapshot,
        worktree=worktree,
    )
    try:
        adapter = validate_adapter_file(worktree / manifest.inference_adapter_path)
        yield adapter, worktree
    finally:
        overlay_mod.cleanup_overlay_worktree(worktree, repo)


def _run_smoke(
    *,
    job_directory: Path,
    model_path: Path,
    model_id: str,
    venv_py: Path,
    adapter_path: Path | None,
    method_repo: Path | None,
    timeout_s: int = 600,
    result_name: str = "inference",
) -> dict:
    script_path = job_directory / "inference.py"
    script_path.write_text(SMOKE_SCRIPT)
    script_path.chmod(0o755)
    output_json = job_directory / f"{result_name}.json"
    log_path = job_directory / f"{result_name}.log"
    output_json.unlink(missing_ok=True)
    extra = {
        "MEASURE_MODEL_PATH": str(model_path),
        "MEASURE_MODEL_ID": model_id,
        "MEASURE_OUTPUT_JSON": str(output_json),
        "MEASURE_DTYPE": "float16",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    }
    if adapter_path is not None:
        extra["MEASURE_ADAPTER_PATH"] = str(adapter_path)
    if method_repo is not None:
        extra["QUANT_AGENT_METHOD_REPO"] = str(method_repo.resolve())
        extra["PYTHONPATH"] = str(method_repo.resolve())
    scratch = job_directory / "hf-scratch"
    child = env.child_env(extra, include_hf=False, scratch_home=scratch)
    with log_path.open("wb") as logf:
        kwargs = {
            "stdout": logf,
            "stderr": subprocess.STDOUT,
            "env": child,
            "timeout": timeout_s,
            "check": False,
        }
        if method_repo is not None:
            kwargs["cwd"] = str(method_repo.resolve())
        proc = subprocess.run([str(venv_py), str(script_path)], **kwargs)
    if proc.returncode != 0 or not output_json.exists():
        tail = log_path.read_text(errors="replace").splitlines()[-30:]
        raise RuntimeError(
            f"Inference verification failed (exit={proc.returncode}). Last log lines:\n"
            + "\n".join(tail)
        )
    payload = json.loads(output_json.read_text())
    if int(payload.get("new_token_count", 0)) < 1:
        raise RuntimeError("Inference verification produced no new tokens")
    output_text = str(payload.get("output_text") or "")
    if not output_text.strip() or not any(char.isalnum() for char in output_text):
        raise RuntimeError(
            f"inference smoke decoded only empty/non-alphanumeric text: {output_text!r}"
        )
    return payload


def verify_job(
    *,
    job_id: str,
    request: dict,
    baseline: bool = False,
) -> dict:
    env.require_host_execution("quantized-model inference verification")
    meta = jobs_mod.refresh_status(job_id)
    if meta.status != "completed":
        raise RuntimeError(
            f"inference verification requires a completed job, got {meta.status}"
        )
    slug = str(request["slug"])
    if meta.model_id != request["model_id"] or meta.method_id != slug:
        raise RuntimeError("job identity does not match request JSON")
    venv_py = venv_python(slug)
    if not venv_py.exists():
        raise RuntimeError(f"method venv is missing: {venv_py}")

    adapter_source = None
    if meta.overlay_path:
        overlay_dir = Path(meta.overlay_path)
        adapter_source = adapter_source_for_overlay(overlay_dir)
    check_authenticity(
        output_dir=meta.output_dir,
        hf_snapshot_path=request["hf_snapshot_path"],
        adapter_source=adapter_source,
    )

    job_directory = jobs_mod.job_dir(meta.job_id)
    model_path = resolve_workspace_path(meta.output_dir)
    with materialize_inference_runtime(meta, request) as (adapter, method_repo):
        if adapter is not None:
            check_authenticity(
                output_dir=meta.output_dir,
                hf_snapshot_path=request["hf_snapshot_path"],
                adapter_source=adapter.read_text(),
            )
        result = _run_smoke(
            job_directory=job_directory,
            model_path=model_path,
            model_id=meta.model_id,
            venv_py=venv_py,
            adapter_path=adapter,
            method_repo=method_repo,
        )

    if baseline:
        snapshot = resolve_workspace_path(request["hf_snapshot_path"])
        fp16 = _run_smoke(
            job_directory=job_directory,
            model_path=snapshot,
            model_id=str(request["model_id"]),
            venv_py=venv_py,
            adapter_path=None,
            method_repo=None,
            result_name="fp16_baseline",
        )
        q_vram = result.get("peak_vram_gb")
        b_vram = fp16.get("peak_vram_gb")
        improved = None
        if isinstance(q_vram, (int, float)) and isinstance(b_vram, (int, float)):
            improved = q_vram < b_vram
        result["fp16_baseline"] = {
            "peak_vram_gb": b_vram,
            "new_token_count": fp16.get("new_token_count"),
            "cuda": fp16.get("cuda"),
        }
        result["improved_vs_fp16"] = improved

    meta.inference = result
    meta.verification_status = "passed"
    meta.verification_error = None
    jobs_mod.write_meta(meta)
    return {
        "status": "passed",
        "job_id": meta.job_id,
        "inference": result,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify a completed quantization job by reloading saved weights"
    )
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--allow-unsafe-host-execution", action="store_true")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Also smoke the original HF snapshot as an fp16 VRAM baseline",
    )
    args = parser.parse_args()
    payload = load_request(args.request)
    try:
        with env.host_execution_policy(args.allow_unsafe_host_execution):
            result = verify_job(
                job_id=args.job_id,
                request=payload,
                baseline=args.baseline,
            )
    except Exception as exc:  # noqa: BLE001 — persist the exact failure as job evidence
        try:
            meta = jobs_mod.read_meta(args.job_id)
            meta.inference = None
            meta.verification_status = "failed"
            meta.verification_error = str(exc)
            jobs_mod.write_meta(meta)
        except FileNotFoundError:
            pass
        print(json.dumps({"status": "failed", "message": str(exc)}, indent=2))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

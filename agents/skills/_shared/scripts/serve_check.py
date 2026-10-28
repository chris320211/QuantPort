#!/usr/bin/env python3
"""Load a vLLM export with vLLM itself and measure it against vLLM fp16.

GPU only. Runs in its own venv (`.venvs/vllm-serve`) because vLLM pins its own
torch; the method venv is never touched. Both sides use the WikiText-2 corpus
and 2048-token windows from quant-benchmark, so the vLLM PPL ratio is directly
comparable to `benchmark.json`.

Gates:
- runtime: the serve venv imports vLLM and sees a CUDA device (`vllm_runtime_unavailable`)
- smoke: the export loads and generates alphanumeric text (`vllm_load_failed`)
- baseline: vLLM fp16 loads and gives a finite PPL (`vllm_baseline_failed`)

Both sides run in float16. When vLLM itself refuses float16 for the
architecture (e.g. Gemma 2, for numerical stability), both rerun in bfloat16
and the report records `dtype`.
- quality: vLLM PPL ratio vs vLLM fp16 <= 1.5 (`vllm_quality_failed`)
- consistency: vLLM PPL ratio / HF PPL ratio <= 1.02 (`vllm_ppl_mismatch`, an
  export bug such as a wrong scale convention, not a quantization problem)
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import env
import jobs as jobs_mod
import paths
from benchmark import (
    DEFAULT_MAX_SEQ_LEN,
    DEFAULT_MAX_TOKENS,
    DEFAULT_STRIDE,
    QUALITY_PPL_RATIO_MAX,
    ensure_wikitext2_corpus,
)
from export_vllm import EXPORT_RECORD
from io_utils import atomic_write_text
from paths import venv_python
from runtime_pins import SERVE_PACKAGES
from verify import resolve_workspace_path

SERVE_VENV_SLUG = "vllm-serve"
SERVE_MIN_PYTHON = (3, 11)
EXPORT_PPL_DRIFT_MAX = 1.02
DECODE_BATCHES = (1, 16)
DECODE_PROMPT_TOKENS = 128
DECODE_NEW_TOKENS = 256
GPU_MEMORY_UTILIZATION = 0.85
_RUN_TIMEOUT_S = 1800
_INSTALL_TIMEOUT_S = 3600  # vLLM + CUDA wheels are ~5 GB; slow mirrors need the room
_PROBE_TIMEOUT_S = 300
DEFAULT_DTYPE = "float16"
FALLBACK_DTYPE = "bfloat16"
# vLLM's refusal when an architecture is not allowed to run in float16.
_FP16_REFUSED = "does not support float16"
_DTYPE_LABELS = {"float16": "fp16", "bfloat16": "bf16"}
_VLLM_RELEASE_WHEEL = (
    "https://github.com/vllm-project/vllm/releases/download/v{version}/"
    "vllm-{version}+{cuda}-cp38-abi3-manylinux_2_28_x86_64.whl"
)
# Offline: no Hub calls and no vLLM usage-stats upload.
SERVE_CHILD_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "PYTHONUNBUFFERED": "1",
    "VLLM_NO_USAGE_STATS": "1",
    "DO_NOT_TRACK": "1",
}
# Import check run after install and before a check, so a broken venv or a
# CUDA runtime mismatch is not reported as a bad export.
RUNTIME_PROBE = (
    "import torch, vllm\n"
    "assert torch.cuda.is_available(), 'torch sees no CUDA device'\n"
    "print(vllm.__version__)\n"
)

SERVE_SCRIPT = r'''#!/usr/bin/env python
"""vLLM smoke + WikiText-2 perplexity + prefill/decode throughput for one checkpoint."""
from __future__ import annotations

import json
import math
import os
import sys
import time


def _load_eval_text(path):
    parts = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("text") if isinstance(row, dict) else None
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    if not parts:
        raise RuntimeError(f"eval corpus is empty: {path}")
    return "\n\n".join(parts)


def _windows(ids, max_seq_len, stride):
    """Same windows as quant-benchmark's _perplexity loop."""
    out = []
    n_ids = len(ids)
    start = 0
    while start < n_ids - 1:
        end = min(start + max_seq_len, n_ids)
        chunk = ids[start:end]
        if len(chunk) - 1 < 1:
            break
        out.append(chunk)
        if end >= n_ids:
            break
        start += stride
    return out


def _logprob(entry, token_id):
    value = entry[token_id]
    return float(getattr(value, "logprob", value))


def _smoke(llm, tokenizer, SamplingParams):
    prompt = "Reply with exactly one short word confirming readiness."
    # Render to text first: with tokenize=True, transformers 5 returns a
    # BatchEncoding, and list() of that is its keys, not token ids.
    try:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    except (TypeError, ValueError, AttributeError):
        ids = tokenizer(prompt)["input_ids"]
    out = llm.generate(
        [{"prompt_token_ids": [int(t) for t in ids]}],
        SamplingParams(max_tokens=8, temperature=0.0),
        use_tqdm=False,
    )[0].outputs[0]
    text = out.text
    if not text.strip() or not any(ch.isalnum() for ch in text):
        raise RuntimeError(f"vLLM smoke decoded only empty/non-alphanumeric text: {text!r}")
    return {"prompt": prompt, "output_text": text, "new_token_count": len(out.token_ids)}


def _perplexity(llm, windows, SamplingParams):
    params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    outs = llm.generate([{"prompt_token_ids": w} for w in windows], params, use_tqdm=False)
    total_nll = 0.0
    total_tokens = 0
    for window, out in zip(windows, outs):
        plp = out.prompt_logprobs
        for i in range(1, len(window)):
            total_nll -= _logprob(plp[i], window[i])
            total_tokens += 1
    if total_tokens < 1:
        raise RuntimeError("vLLM perplexity counted zero tokens")
    mean_nll = total_nll / total_tokens
    if not math.isfinite(mean_nll):
        raise RuntimeError("vLLM perplexity is not finite")
    return {
        "perplexity": math.exp(mean_nll),
        "loss": mean_nll,
        "tokens_evaluated": total_tokens,
        "windows": len(windows),
    }


def _prefill(llm, windows, SamplingParams):
    params = SamplingParams(max_tokens=1, temperature=0.0)
    llm.generate([{"prompt_token_ids": windows[0]}], params, use_tqdm=False)  # warm up
    begin = time.perf_counter()
    llm.generate([{"prompt_token_ids": w} for w in windows], params, use_tqdm=False)
    elapsed = time.perf_counter() - begin
    tokens = sum(len(w) for w in windows)
    return {"tokens_per_s": tokens / elapsed, "tokens": tokens, "batch": len(windows), "seconds": elapsed}


def _decode(llm, ids, batch, prompt_tokens, new_tokens, SamplingParams):
    prompts = [ids[i * prompt_tokens:(i + 1) * prompt_tokens] for i in range(batch)]
    params = SamplingParams(max_tokens=new_tokens, temperature=0.0, ignore_eos=True)
    llm.generate([{"prompt_token_ids": prompts[0]}], SamplingParams(max_tokens=8, temperature=0.0), use_tqdm=False)
    begin = time.perf_counter()
    outs = llm.generate([{"prompt_token_ids": p} for p in prompts], params, use_tqdm=False)
    elapsed = time.perf_counter() - begin
    generated = sum(len(o.outputs[0].token_ids) for o in outs)
    return {
        "batch": batch,
        "tokens_per_s": generated / elapsed,
        "tokens_per_s_per_sequence": generated / elapsed / batch,
        "generated_tokens": generated,
        "seconds": elapsed,
    }


def main():
    with open(sys.argv[1], encoding="utf-8") as handle:
        cfg = json.load(handle)
    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model_path = cfg["model_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    ids = tokenizer(_load_eval_text(cfg["eval_path"]), add_special_tokens=False)["input_ids"]
    if cfg["max_tokens"] > 0:
        ids = ids[: cfg["max_tokens"]]
    windows = _windows(ids, cfg["max_seq_len"], cfg["stride"])
    if not windows:
        raise RuntimeError("eval corpus produced no windows")
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        dtype=cfg["dtype"],
        max_model_len=cfg["max_seq_len"] + 1,
        gpu_memory_utilization=cfg["gpu_memory_utilization"],
        enable_prefix_caching=False,
        seed=0,
    )
    payload = {
        "model_path": model_path,
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(0),
        "compute_capability": ".".join(str(v) for v in torch.cuda.get_device_capability(0)),
        "smoke": _smoke(llm, tokenizer, SamplingParams),
        **_perplexity(llm, windows, SamplingParams),
        "prefill": _prefill(llm, windows, SamplingParams),
        "decode": [
            _decode(llm, ids, batch, cfg["decode_prompt_tokens"], cfg["decode_new_tokens"], SamplingParams)
            for batch in cfg["decode_batches"]
        ],
    }
    tmp = cfg["output_json"] + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, cfg["output_json"])
    print("SERVE_RESULT=" + json.dumps({"perplexity": payload["perplexity"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def serve_venv_python() -> Path:
    return venv_python(SERVE_VENV_SLUG)


def install_serve_venv(python: str, vllm_spec: str | None = None, vllm_cuda: str | None = None) -> dict:
    """Create `.venvs/vllm-serve` with the pinned vLLM. Separate from every method venv.

    `vllm_spec` (e.g. `vllm==0.19.0`) overrides the pin when the host driver is
    older than the pinned wheel's CUDA runtime. `vllm_cuda` (e.g. `cu129`)
    keeps the version but installs vLLM's release wheel built for that CUDA,
    with torch from the matching PyTorch index (PyPI's wheel targets CUDA 13).
    """
    from install_venv import _run_argv

    env.require_host_execution("vLLM serve venv install")
    probe = subprocess.run(
        [python, "-c", "import sys; print(sys.version_info[0], sys.version_info[1])"],
        capture_output=True, text=True, check=False,
    )
    if probe.returncode != 0:
        raise RuntimeError(f"cannot run interpreter {python!r}")
    version = tuple(int(v) for v in probe.stdout.split())
    if version < SERVE_MIN_PYTHON:
        raise RuntimeError(
            f"vLLM needs Python >= {'.'.join(map(str, SERVE_MIN_PYTHON))}; {python} is "
            f"{'.'.join(map(str, version))}. Pass --python with a newer interpreter."
        )
    packages = list(SERVE_PACKAGES)
    if vllm_spec:
        if not vllm_spec.startswith("vllm==") or any(ch.isspace() for ch in vllm_spec):
            raise ValueError(f"--vllm-spec must look like vllm==X.Y.Z, got {vllm_spec!r}")
        packages = [vllm_spec if pkg.startswith("vllm==") else pkg for pkg in packages]
    index_args: list[str] = []
    if vllm_cuda:
        if not re.fullmatch(r"cu1\d{2}", vllm_cuda):
            raise ValueError(f"--vllm-cuda must look like cu129, got {vllm_cuda!r}")
        version = next(pkg for pkg in packages if pkg.startswith("vllm==")).split("==", 1)[1]
        wheel = _VLLM_RELEASE_WHEEL.format(version=version, cuda=vllm_cuda)
        packages = [f"vllm @ {wheel}" if pkg.startswith("vllm==") else pkg for pkg in packages]
        index_args = ["--extra-index-url", f"https://download.pytorch.org/whl/{vllm_cuda}"]
    venv_dir = paths.VENV_ROOT / SERVE_VENV_SLUG
    scratch = paths.VENV_ROOT / ".scratch" / SERVE_VENV_SLUG
    steps = [
        [python, "-m", "venv", str(venv_dir)],
        [str(venv_dir / "bin" / "python"), "-m", "pip", "install", "--upgrade", "pip", "wheel"],
        [str(venv_dir / "bin" / "python"), "-m", "pip", "install", *index_args, *packages],
    ]
    results = []
    for argv in steps:
        result = _run_argv(argv, None, _INSTALL_TIMEOUT_S, scratch_home=scratch)
        results.append({"argv": argv[1:], **result})
        if not result["ok"]:
            return {"status": "failed", "venv": str(venv_dir), "steps": results}
    error = probe_runtime(venv_dir / "bin" / "python", scratch)
    if error:
        return {
            "status": "failed",
            "venv": str(venv_dir),
            "packages": packages,
            "error": error,
            "hint": (
                "if the error names libcudart or a CUDA version, reinstall with --vllm-cuda cu129 "
                "(CUDA 12.x driver) or --vllm-spec vllm==X.Y.Z"
            ),
        }
    return {"status": "installed", "venv": str(venv_dir), "packages": packages}


def probe_runtime(venv_py: Path, scratch: Path) -> str | None:
    """Import vLLM and check CUDA in the serve venv. Returns the error tail, or None."""
    try:
        proc = subprocess.run(
            [str(venv_py), "-c", RUNTIME_PROBE],
            capture_output=True, text=True, check=False, timeout=_PROBE_TIMEOUT_S,
            env=env.child_env(SERVE_CHILD_ENV, include_hf=False, scratch_home=scratch),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"{type(exc).__name__}: {exc}"
    if proc.returncode == 0:
        return None
    tail = (proc.stderr or proc.stdout).strip().splitlines()[-5:]
    return "\n".join(tail) or f"exit {proc.returncode}"


def dtype_label(report: dict | None) -> str:
    """Short name of the dtype both sides ran in (`fp16` for reports that predate `dtype`)."""
    dtype = (report or {}).get("dtype") or DEFAULT_DTYPE
    return _DTYPE_LABELS.get(dtype, dtype)


def _run_side(
    *, model_path: Path, work_dir: Path, venv_py: Path, corpus: Path, result_name: str, dtype: str = DEFAULT_DTYPE
) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)
    script_path = work_dir / "serve_infer.py"
    script_path.write_text(SERVE_SCRIPT)
    script_path.chmod(0o755)
    output_json = work_dir / f"{result_name}.json"
    output_json.unlink(missing_ok=True)
    cfg_path = work_dir / f"{result_name}.config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "model_path": str(model_path),
                "output_json": str(output_json),
                "eval_path": str(corpus),
                "dtype": dtype,
                "max_seq_len": DEFAULT_MAX_SEQ_LEN,
                "stride": DEFAULT_STRIDE,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
                "decode_batches": list(DECODE_BATCHES),
                "decode_prompt_tokens": DECODE_PROMPT_TOKENS,
                "decode_new_tokens": DECODE_NEW_TOKENS,
            },
            indent=2,
        )
    )
    log_path = work_dir / f"{result_name}.log"
    child = env.child_env(
        SERVE_CHILD_ENV,
        include_hf=False,
        scratch_home=work_dir / f"{result_name}-hf-scratch",
    )
    with log_path.open("wb") as logf:
        try:
            proc = subprocess.run(
                [str(venv_py), str(script_path), str(cfg_path)],
                stdout=logf, stderr=subprocess.STDOUT, env=child,
                timeout=_RUN_TIMEOUT_S, check=False,
            )
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            returncode = -1
    if returncode != 0 or not output_json.exists():
        tail = log_path.read_text(errors="replace").splitlines()[-40:]
        raise RuntimeError(
            f"vLLM serve check failed for {result_name} (exit={returncode}). Last log lines:\n"
            + "\n".join(tail)
        )
    return json.loads(output_json.read_text())


def compare_serve(quantized: dict, baseline: dict, hf_benchmark: dict | None = None) -> dict:
    """vLLM quantized vs vLLM fp16, plus drift against the HF benchmark ratio. Pure."""
    q_ppl = float(quantized["perplexity"])
    b_ppl = float(baseline["perplexity"])
    if not math.isfinite(q_ppl) or not math.isfinite(b_ppl) or b_ppl <= 0:
        raise ValueError("perplexity must be finite and baseline ppl must be > 0")
    ratio = q_ppl / b_ppl
    hf_ratio = None
    if hf_benchmark:
        value = (hf_benchmark.get("comparison") or {}).get("ppl_ratio")
        if isinstance(value, (int, float)) and value > 0:
            hf_ratio = float(value)
    drift = ratio / hf_ratio if hf_ratio else None
    consistent = None if drift is None else drift <= EXPORT_PPL_DRIFT_MAX
    prefill_speedup = quantized["prefill"]["tokens_per_s"] / baseline["prefill"]["tokens_per_s"]
    base_decode = {row["batch"]: row["tokens_per_s"] for row in baseline["decode"]}
    decode_speedup = {
        str(row["batch"]): row["tokens_per_s"] / base_decode[row["batch"]]
        for row in quantized["decode"]
        if base_decode.get(row["batch"])
    }
    quality_ok = ratio <= QUALITY_PPL_RATIO_MAX
    issues = []
    if not quality_ok:
        issues.append("vllm_quality_failed")
    if consistent is False:
        issues.append("vllm_ppl_mismatch")
    return {
        "ppl_ratio": ratio,
        "hf_ppl_ratio": hf_ratio,
        "ppl_ratio_drift": drift,
        "export_consistent": consistent,
        "quality_ok": quality_ok,
        "prefill_speedup": prefill_speedup,
        "decode_speedup": decode_speedup,
        "improved_throughput": prefill_speedup > 1 or any(v > 1 for v in decode_speedup.values()),
        "issue_codes": issues,
    }


def serve_check(
    *,
    model_dir: Path,
    baseline_dir: Path,
    work_dir: Path,
    hf_benchmark: dict | None = None,
) -> dict:
    env.require_host_execution("vLLM serve check")
    model_dir = Path(model_dir).resolve()
    record_path = model_dir / EXPORT_RECORD
    if record_path.is_file():
        export_record = json.loads(record_path.read_text())
        if export_record.get("status") != "exported":
            raise RuntimeError(f"{model_dir} is not a vLLM export (status={export_record.get('status')})")
    venv_py = serve_venv_python()
    if not venv_py.exists():
        raise RuntimeError(f"serve venv is missing: {venv_py}. Run serve_check.py --install first.")
    report: dict = {
        "model_dir": str(model_dir),
        "baseline_dir": str(Path(baseline_dir).resolve()),
        "dataset": "wikitext-2-raw-v1",
        "split": "test",
        "max_seq_len": DEFAULT_MAX_SEQ_LEN,
        "stride": DEFAULT_STRIDE,
        "max_tokens": DEFAULT_MAX_TOKENS,
    }
    runtime_error = probe_runtime(venv_py, Path(work_dir) / "probe-hf-scratch")
    if runtime_error:
        return {**report, "status": "failed", "issue_codes": ["vllm_runtime_unavailable"], "error": runtime_error}
    corpus = ensure_wikitext2_corpus(venv_py)
    dtype = DEFAULT_DTYPE
    try:
        try:
            quantized = _run_side(
                model_path=model_dir, work_dir=work_dir, venv_py=venv_py, corpus=corpus,
                result_name="vllm_quantized", dtype=dtype,
            )
        except RuntimeError as exc:
            if _FP16_REFUSED not in str(exc):
                raise
            dtype = FALLBACK_DTYPE
            quantized = _run_side(
                model_path=model_dir, work_dir=work_dir, venv_py=venv_py, corpus=corpus,
                result_name="vllm_quantized", dtype=dtype,
            )
    except RuntimeError as exc:
        return {**report, "dtype": dtype, "status": "failed", "issue_codes": ["vllm_load_failed"], "error": str(exc)}
    report["dtype"] = dtype
    # From here on keep the quantized measurements, whatever the baseline does.
    # The baseline runs in the same dtype, so the ratio compares like with like.
    try:
        baseline = _run_side(
            model_path=Path(baseline_dir).resolve(), work_dir=work_dir, venv_py=venv_py, corpus=corpus,
            result_name="vllm_fp16", dtype=dtype,
        )
    except RuntimeError as exc:
        return {**report, "status": "failed", "issue_codes": ["vllm_baseline_failed"], "quantized": quantized, "error": str(exc)}
    try:
        comparison = compare_serve(quantized, baseline, hf_benchmark)
    except (ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
        baseline_ppl = baseline.get("perplexity")
        bad_baseline = not isinstance(baseline_ppl, (int, float)) or not math.isfinite(baseline_ppl) or baseline_ppl <= 0
        return {
            **report,
            "status": "failed",
            "issue_codes": ["vllm_baseline_failed" if bad_baseline else "vllm_quality_failed"],
            "quantized": quantized,
            "fp16_baseline": baseline,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        **report,
        "status": "passed" if not comparison["issue_codes"] else "failed",
        "issue_codes": comparison["issue_codes"],
        "quantized": quantized,
        "fp16_baseline": baseline,
        "comparison": comparison,
    }


def _mark_export(model_dir: Path, report: dict) -> None:
    """Stamp the export record so publish only advertises vLLM after a passed check."""
    record_path = Path(model_dir) / EXPORT_RECORD
    if not record_path.is_file():
        return
    record = json.loads(record_path.read_text())
    record["serve_check"] = report["status"]
    record["serve_issue_codes"] = report.get("issue_codes", [])
    comparison = report.get("comparison") or {}
    for key in ("ppl_ratio", "prefill_speedup", "decode_speedup"):
        if key in comparison:
            record[f"vllm_{key}"] = comparison[key]
    if report.get("quantized"):
        record["vllm_version"] = report["quantized"].get("vllm_version")
        record["vllm_gpu"] = report["quantized"].get("gpu_name")
    atomic_write_text(record_path, json.dumps(record, indent=2) + "\n", mode=0o644)


def _job_inputs(job_id: str, request_path: Path) -> tuple[Path, Path, Path, dict | None, Path]:
    from request import load_request

    request = load_request(request_path)
    job_directory = jobs_mod.job_dir(job_id)
    export_path = job_directory / EXPORT_RECORD
    if not export_path.is_file():
        raise RuntimeError(f"run export_vllm.py --job-id {job_id} first ({export_path} missing)")
    export_record = json.loads(export_path.read_text())
    if export_record.get("status") != "exported":
        raise RuntimeError(f"export status is {export_record.get('status')}: {export_record.get('reason_code')}")
    benchmark_path = job_directory / "benchmark.json"
    hf_benchmark = json.loads(benchmark_path.read_text()) if benchmark_path.is_file() else None
    return (
        Path(export_record["output_dir"]),
        resolve_workspace_path(request["hf_snapshot_path"]),
        job_directory / "vllm",
        hf_benchmark,
        job_directory / "vllm.json",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve-check a vLLM export against vLLM fp16")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--install", action="store_true", help="Create .venvs/vllm-serve")
    mode.add_argument("--job-id", help="Pipeline mode: the export recorded on jobs/<id>")
    mode.add_argument("--model-dir", type=Path, help="Backfill mode: an export directory")
    parser.add_argument("--request", type=Path, help="Request JSON (with --job-id)")
    parser.add_argument("--baseline-dir", type=Path, help="fp16 snapshot (with --model-dir)")
    parser.add_argument("--hf-benchmark", type=Path, help="benchmark.json for the drift gate (with --model-dir)")
    parser.add_argument("--out", type=Path, help="Report path (with --model-dir)")
    parser.add_argument("--python", default=sys.executable, help="Interpreter for --install (>= 3.11)")
    parser.add_argument("--vllm-spec", help="Override the vLLM pin for --install (vllm==X.Y.Z)")
    parser.add_argument("--vllm-cuda", help="--install vLLM's release wheel for this CUDA (cu129 on a 12.x driver)")
    parser.add_argument("--allow-unsafe-host-execution", action="store_true")
    args = parser.parse_args()
    try:
        with env.host_execution_policy(args.allow_unsafe_host_execution):
            if args.install:
                result = install_serve_venv(args.python, args.vllm_spec, args.vllm_cuda)
                print(json.dumps(result, indent=2))
                return 0 if result["status"] == "installed" else 1
            if args.job_id:
                if not args.request:
                    raise ValueError("--request is required with --job-id")
                model_dir, baseline_dir, work_dir, hf_benchmark, out = _job_inputs(args.job_id, args.request)
            else:
                if not args.baseline_dir or not args.out:
                    raise ValueError("--baseline-dir and --out are required with --model-dir")
                model_dir, baseline_dir = args.model_dir, args.baseline_dir
                out = args.out
                work_dir = out.with_suffix("")
                hf_benchmark = json.loads(args.hf_benchmark.read_text()) if args.hf_benchmark else None
            report = serve_check(
                model_dir=model_dir, baseline_dir=baseline_dir, work_dir=work_dir, hf_benchmark=hf_benchmark
            )
    except Exception as exc:  # noqa: BLE001 — report the exact failure as JSON
        print(json.dumps({"status": "failed", "message": str(exc)}, indent=2))
        return 1
    atomic_write_text(out, json.dumps(report, indent=2) + "\n", mode=0o644)
    _mark_export(model_dir, report)
    summary = {k: report.get(k) for k in ("status", "issue_codes", "comparison")}
    print(json.dumps(summary, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

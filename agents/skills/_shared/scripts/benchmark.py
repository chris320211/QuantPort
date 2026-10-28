#!/usr/bin/env python3
"""Compare quantized LLM metrics against the fp16 snapshot baseline.

Always runs both sides: WikiText-2 perplexity, throughput, and peak VRAM.
Does not treat generation smoke or VRAM-only verify --baseline as this report.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import library as library_mod
import env
import jobs as jobs_mod
import paths
from paths import venv_python
from request import load_request
from verify import (
    check_authenticity,
    materialize_inference_runtime,
    resolve_workspace_path,
)

QUALITY_PPL_RATIO_MAX = 1.5
DEFAULT_MAX_SEQ_LEN = 2048
DEFAULT_STRIDE = 2048
DEFAULT_MAX_TOKENS = 65536
_FETCH_TIMEOUT_S = 300
_RUN_TIMEOUT_S = 1800

FETCH_SCRIPT = r'''#!/usr/bin/env python
"""Write WikiText-2 test documents to JSONL (public dataset, no tokens)."""
from __future__ import annotations

import json
import os
import sys

from datasets import load_dataset


def main() -> int:
    dest, cache = sys.argv[1], sys.argv[2]
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    # The canonical repo id; newer huggingface_hub rejects the bare "wikitext".
    ds = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="test",
        cache_dir=cache,
    )
    rows = []
    for text in ds["text"]:
        if isinstance(text, str) and text.strip():
            rows.append({"text": text})
    if len(rows) < 8:
        raise RuntimeError(f"wikitext-2 test too small: {len(rows)} documents")
    tmp = dest + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, dest)
    print(dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

BENCHMARK_SCRIPT = r'''#!/usr/bin/env python
"""WikiText-2 perplexity / throughput / VRAM for one local checkpoint."""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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
        raise RuntimeError("loaded model does not implement the causal-LM interface")
    model.eval()
    return model, tokenizer


def _input_device(model):
    return model.get_input_embeddings().weight.device


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


def _perplexity(model, tokenizer, text, max_seq_len, stride, max_tokens):
    device = _input_device(model)
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"][0]
    if int(input_ids.numel()) < 2:
        raise RuntimeError("eval corpus tokenized to fewer than 2 tokens")
    if max_tokens > 0:
        input_ids = input_ids[:max_tokens]
    n_ids = int(input_ids.numel())
    total_nll = 0.0
    total_tokens = 0
    windows = 0
    start = 0
    while start < n_ids - 1:
        end = min(start + max_seq_len, n_ids)
        chunk = input_ids[start:end].unsqueeze(0).to(device)
        n_tok = int(chunk.shape[-1]) - 1
        if n_tok < 1:
            break
        with torch.inference_mode():
            loss = model(input_ids=chunk, labels=chunk).loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at window {windows}")
        total_nll += float(loss.item()) * n_tok
        total_tokens += n_tok
        windows += 1
        if end >= n_ids:
            break
        start += stride
    if total_tokens < 1:
        raise RuntimeError("benchmark counted zero eval tokens")
    mean_nll = total_nll / total_tokens
    return {
        "perplexity": math.exp(mean_nll),
        "loss": mean_nll,
        "tokens_evaluated": total_tokens,
        "windows": windows,
        "seq_len": max_seq_len,
        "stride": stride,
    }


def main():
    model_path = os.environ["MEASURE_MODEL_PATH"]
    model_id = os.environ.get("MEASURE_MODEL_ID", model_path)
    adapter_path = os.environ.get("MEASURE_ADAPTER_PATH") or None
    output_json = os.environ["MEASURE_OUTPUT_JSON"]
    eval_path = os.environ["MEASURE_EVAL_PATH"]
    dtype_name = os.environ.get("MEASURE_DTYPE", "float16")
    trust = os.environ.get("MEASURE_TRUST_REMOTE_CODE") == "1"
    max_seq_len = int(os.environ.get("MEASURE_MAX_SEQ_LEN", "2048"))
    stride = int(os.environ.get("MEASURE_STRIDE", "2048"))
    max_tokens = int(os.environ.get("MEASURE_MAX_TOKENS", "65536"))

    text = _load_eval_text(eval_path)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    model, tokenizer = _load_model_and_tokenizer(
        model_path, model_id, adapter_path, dtype_name, trust,
    )
    # Packed CUDA / Triton kernels compile on the first forward. Time steady-state
    # 2048-token prefill, not compile. PPL still uses the full corpus below.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        warm_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        n_warm = min(max_seq_len, max(2, int(warm_ids.numel())))
        dummy = warm_ids[:n_warm].unsqueeze(0).to(_input_device(model))
        with torch.inference_mode():
            model(input_ids=dummy, labels=dummy)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    metrics = _perplexity(model, tokenizer, text, max_seq_len, stride, max_tokens)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_vram_gb = None
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    tokens = int(metrics["tokens_evaluated"])
    payload = {
        **metrics,
        "elapsed_s": elapsed,
        "tokens_per_s": (tokens / elapsed) if elapsed > 0 else None,
        "peak_vram_gb": peak_vram_gb,
        "model_path": model_path,
        "model_id": model_id,
        "adapter_path": adapter_path,
        "dtype": dtype_name,
        "cuda": bool(torch.cuda.is_available()),
        "eval_corpus": eval_path,
    }
    tmp = output_json + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, output_json)
    print("BENCHMARK_RESULT=" + json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def compare_metrics(quantized: dict, baseline: dict) -> dict:
    """Derive quality vs efficiency vs the fp16 snapshot. Does not load a model."""
    q_ppl = float(quantized["perplexity"])
    b_ppl = float(baseline["perplexity"])
    if not math.isfinite(q_ppl) or not math.isfinite(b_ppl) or b_ppl <= 0:
        raise ValueError("perplexity must be finite and baseline ppl must be > 0")
    ratio = q_ppl / b_ppl
    q_vram = quantized.get("peak_vram_gb")
    b_vram = baseline.get("peak_vram_gb")
    q_tps = quantized.get("tokens_per_s")
    b_tps = baseline.get("tokens_per_s")
    improved_vram = None
    if isinstance(q_vram, (int, float)) and isinstance(b_vram, (int, float)):
        improved_vram = q_vram < b_vram
    improved_throughput = None
    if isinstance(q_tps, (int, float)) and isinstance(b_tps, (int, float)):
        improved_throughput = q_tps > b_tps
    vram_delta = None
    if isinstance(q_vram, (int, float)) and isinstance(b_vram, (int, float)):
        vram_delta = q_vram - b_vram
    tps_delta = None
    if isinstance(q_tps, (int, float)) and isinstance(b_tps, (int, float)):
        tps_delta = q_tps - b_tps
    return {
        "ppl_delta": q_ppl - b_ppl,
        "ppl_ratio": ratio,
        "peak_vram_delta_gb": vram_delta,
        "tokens_per_s_delta": tps_delta,
        "quality_ok": ratio <= QUALITY_PPL_RATIO_MAX,
        "improved_vram": improved_vram,
        "improved_throughput": improved_throughput,
        "efficiency_improved": bool(improved_vram or improved_throughput),
    }


def eval_corpus_path() -> Path:
    return paths.EVAL_CACHE / "wikitext-2-raw-v1-test.jsonl"


def ensure_wikitext2_corpus(venv_py: Path) -> Path:
    dest = eval_corpus_path()
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    cache = paths.EVAL_CACHE / "hf"
    cache.mkdir(parents=True, exist_ok=True)
    fetch_path = dest.parent / "fetch_wikitext2.py"
    fetch_path.write_text(FETCH_SCRIPT)
    fetch_path.chmod(0o755)
    log_path = dest.parent / "fetch_wikitext2.log"
    extra = {
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "PYTHONUNBUFFERED": "1",
    }
    child = env.child_env(extra, include_hf=False, scratch_home=cache / "scratch")
    # Public WikiText download needs a writable HF cache, not user tokens.
    child["HF_HOME"] = str(cache)
    child["HF_DATASETS_CACHE"] = str(cache / "datasets")
    child["HUGGINGFACE_HUB_CACHE"] = str(cache)
    with log_path.open("wb") as logf:
        proc = subprocess.run(
            [str(venv_py), str(fetch_path), str(dest), str(cache)],
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=child,
            timeout=_FETCH_TIMEOUT_S,
            check=False,
        )
    if proc.returncode != 0 or not dest.is_file():
        tail = log_path.read_text(errors="replace").splitlines()[-40:]
        raise RuntimeError(
            f"WikiText-2 corpus fetch failed (exit={proc.returncode}). Last log lines:\n"
            + "\n".join(tail)
        )
    return dest


def _run_side(
    *,
    job_directory: Path,
    model_path: Path,
    model_id: str,
    venv_py: Path,
    adapter_path: Path | None,
    method_repo: Path | None,
    eval_path: Path,
    result_name: str,
    max_seq_len: int,
    stride: int,
    max_tokens: int,
) -> dict:
    script_path = job_directory / "benchmark_infer.py"
    script_path.write_text(BENCHMARK_SCRIPT)
    script_path.chmod(0o755)
    output_json = job_directory / f"{result_name}.json"
    log_path = job_directory / f"{result_name}.log"
    output_json.unlink(missing_ok=True)
    extra = {
        "MEASURE_MODEL_PATH": str(model_path),
        "MEASURE_MODEL_ID": model_id,
        "MEASURE_OUTPUT_JSON": str(output_json),
        "MEASURE_EVAL_PATH": str(eval_path),
        "MEASURE_DTYPE": "float16",
        "MEASURE_MAX_SEQ_LEN": str(max_seq_len),
        "MEASURE_STRIDE": str(stride),
        "MEASURE_MAX_TOKENS": str(max_tokens),
        "MEASURE_TRUST_REMOTE_CODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "PYTHONUNBUFFERED": "1",
    }
    if adapter_path is not None:
        extra["MEASURE_ADAPTER_PATH"] = str(adapter_path)
    if method_repo is not None:
        extra["QUANT_AGENT_METHOD_REPO"] = str(method_repo.resolve())
        extra["PYTHONPATH"] = str(method_repo.resolve())
    scratch = job_directory / f"{result_name}-hf-scratch"
    child = env.child_env(extra, include_hf=False, scratch_home=scratch)
    with log_path.open("wb") as logf:
        kwargs = {
            "stdout": logf,
            "stderr": subprocess.STDOUT,
            "env": child,
            "timeout": _RUN_TIMEOUT_S,
            "check": False,
        }
        if method_repo is not None:
            kwargs["cwd"] = str(method_repo.resolve())
        proc = subprocess.run([str(venv_py), str(script_path)], **kwargs)
    if proc.returncode != 0 or not output_json.exists():
        tail = log_path.read_text(errors="replace").splitlines()[-40:]
        raise RuntimeError(
            f"LLM benchmark failed for {result_name} (exit={proc.returncode}). Last log lines:\n"
            + "\n".join(tail)
        )
    payload = json.loads(output_json.read_text())
    if not math.isfinite(float(payload.get("perplexity", float("nan")))):
        raise RuntimeError(f"{result_name} perplexity is not finite")
    if int(payload.get("tokens_evaluated", 0)) < 1:
        raise RuntimeError(f"{result_name} evaluated zero tokens")
    return payload


def _trust_remote(model_path: Path) -> bool:
    return (model_path / "config.json").is_file() or any(model_path.glob("modeling_*.py"))


def benchmark_job(
    *,
    job_id: str,
    request: dict,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    stride: int = DEFAULT_STRIDE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict:
    env.require_host_execution("quantized vs fp16 LLM metric benchmark")
    meta = jobs_mod.refresh_status(job_id)
    if meta.status != "completed":
        raise RuntimeError(f"benchmark requires a completed job, got {meta.status}")
    if meta.verification_status != "passed":
        raise RuntimeError(
            "benchmark requires quant-verify passed; "
            f"got verification_status={meta.verification_status!r}"
        )
    slug = str(request["slug"])
    if meta.model_id != request["model_id"] or meta.method_id != slug:
        raise RuntimeError("job identity does not match request JSON")
    venv_py = venv_python(slug)
    if not venv_py.exists():
        raise RuntimeError(f"method venv is missing: {venv_py}")

    adapter_source = None
    if meta.overlay_path:
        from verify import adapter_source_for_overlay

        adapter_source = adapter_source_for_overlay(Path(meta.overlay_path))
    check_authenticity(
        output_dir=meta.output_dir,
        hf_snapshot_path=request["hf_snapshot_path"],
        adapter_source=adapter_source,
    )

    job_directory = jobs_mod.job_dir(meta.job_id)
    model_path = resolve_workspace_path(meta.output_dir)
    snapshot = resolve_workspace_path(request["hf_snapshot_path"])
    corpus = ensure_wikitext2_corpus(venv_py)

    with materialize_inference_runtime(meta, request) as (adapter, method_repo):
        if adapter is not None:
            check_authenticity(
                output_dir=meta.output_dir,
                hf_snapshot_path=request["hf_snapshot_path"],
                adapter_source=adapter.read_text(),
            )
        quantized = _run_side(
            job_directory=job_directory,
            model_path=model_path,
            model_id=meta.model_id,
            venv_py=venv_py,
            adapter_path=adapter,
            method_repo=method_repo,
            eval_path=corpus,
            result_name="benchmark_quantized",
            max_seq_len=max_seq_len,
            stride=stride,
            max_tokens=max_tokens,
        )

    baseline = _run_side(
        job_directory=job_directory,
        model_path=snapshot,
        model_id=str(request["model_id"]),
        venv_py=venv_py,
        adapter_path=None,
        method_repo=None,
        eval_path=corpus,
        result_name="benchmark_fp16",
        max_seq_len=max_seq_len,
        stride=stride,
        max_tokens=max_tokens,
    )
    comparison = compare_metrics(quantized, baseline)
    report = {
        "dataset": "wikitext-2-raw-v1",
        "split": "test",
        "max_seq_len": max_seq_len,
        "stride": stride,
        "max_tokens": max_tokens,
        "eval_corpus": str(corpus),
        "quantized": quantized,
        "fp16_baseline": baseline,
        "comparison": comparison,
        "trust_remote_code": _trust_remote(model_path),
    }
    report_path = job_directory / "benchmark.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    meta.metrics = report
    meta.benchmark_status = "passed"
    meta.benchmark_error = None
    jobs_mod.write_meta(meta)
    compare_payload: dict | None = None
    compare_error: str | None = None
    try:
        rows = library_mod.query(
            model_id=str(request.get("model_id") or ""),
            gpu_instance=str(request.get("gpu_instance") or ""),
            best_only=True,
        )
        compare_payload = {
            "recorded": False,
            "best": rows[0] if rows else None,
        }
    except Exception as exc:  # noqa: BLE001 — ranking must not fail a passed WikiText-2 run
        compare_error = str(exc)
    return {
        "status": "passed",
        "job_id": meta.job_id,
        "benchmark": report,
        "compare": compare_payload,
        "compare_error": compare_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark a verified quantized artifact against the fp16 snapshot"
    )
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--allow-unsafe-host-execution", action="store_true")
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    args = parser.parse_args()
    payload = load_request(args.request)
    try:
        with env.host_execution_policy(args.allow_unsafe_host_execution):
            result = benchmark_job(
                job_id=args.job_id,
                request=payload,
                max_seq_len=args.max_seq_len,
                stride=args.stride,
                max_tokens=args.max_tokens,
            )
    except Exception as exc:  # noqa: BLE001 — persist the exact failure as job evidence
        try:
            meta = jobs_mod.read_meta(args.job_id)
            meta.benchmark_status = "failed"
            meta.benchmark_error = str(exc)
            jobs_mod.write_meta(meta)
        except FileNotFoundError:
            pass
        print(json.dumps({"status": "failed", "message": str(exc)}, indent=2))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

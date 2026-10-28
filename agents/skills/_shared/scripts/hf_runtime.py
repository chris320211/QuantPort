#!/usr/bin/env python3
"""Measure the native artifact in plain Transformers next to the vLLM serve check.

`measure` loads one checkpoint in Transformers (the native quantized artifact
through its inference adapter, or the original model) and records WikiText-2
PPL, prefill tok/s, decode tok/s and peak VRAM on the same windows and decode
test as serve_check. It imports torch, so run it with the method venv's python
(the venv that benchmarked the artifact), never the vLLM serve venv.

`attach` folds the quantized and baseline measurements into a passed serve
report as `hf_runtime`, so the library line and the Hub cards can say what the
vLLM build buys over plain Transformers on the same GPU. Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from io_utils import atomic_write_text

MAX_SEQ_LEN = 2048
MAX_TOKENS = 65536
PREFILL_MICRO_BATCH = 8
DECODE_BATCHES = (1, 16)
DECODE_PROMPT_TOKENS = 128
DECODE_NEW_TOKENS = 256
_SIDE_KEYS = ("dtype", "perplexity", "tokens_evaluated", "windows", "prefill", "decode", "peak_vram_gb")


def _load(model_path: str, dtype: str, adapter: str | None):
    import importlib.util

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if adapter:
        spec = importlib.util.spec_from_file_location("quant_agent_inference_adapter", adapter)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not import inference adapter: {adapter}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.load_model_and_tokenizer(
            model_path=model_path, model_id=model_path, dtype=dtype, device="cuda", trust_remote_code=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=getattr(torch, dtype), device_map="cuda", local_files_only=True,
    )
    return model.eval(), tokenizer


def _eval_text(path: str) -> str:
    parts = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                text = json.loads(line).get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
    if not parts:
        raise RuntimeError(f"eval corpus is empty: {path}")
    return "\n\n".join(parts)


def measure(model_path: str, eval_path: str, dtype: str, adapter: str | None = None) -> dict:
    import inspect
    import time

    import torch
    import transformers

    if not torch.cuda.is_available():
        raise RuntimeError("torch sees no CUDA device")
    torch.manual_seed(0)
    model, tokenizer = _load(model_path, dtype, adapter)
    device = model.get_input_embeddings().weight.device
    ids = tokenizer(_eval_text(eval_path), return_tensors="pt", add_special_tokens=False)["input_ids"][0][:MAX_TOKENS]
    windows = [ids[s:s + MAX_SEQ_LEN] for s in range(0, ids.numel() - 1, MAX_SEQ_LEN)]
    windows = [w for w in windows if w.numel() == MAX_SEQ_LEN]
    params = inspect.signature(model.forward).parameters
    # Prefill keeps only the last token's logits, as vLLM does; the name changed across releases.
    keep = next(({name: 1} for name in ("logits_to_keep", "num_logits_to_keep") if name in params), {})

    with torch.inference_mode():
        first = windows[0][None].to(device)
        model(input_ids=first, labels=first)  # compile Triton kernels before timing
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        nll, counted = 0.0, 0
        for window in windows:
            x = window[None].to(device)
            nll += float(model(input_ids=x, labels=x).loss) * (x.shape[1] - 1)
            counted += x.shape[1] - 1

        stacked = torch.stack(windows).to(device)
        micro = PREFILL_MICRO_BATCH
        try:
            model(input_ids=stacked[:micro], use_cache=False, **keep)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            micro = 1
            model(input_ids=stacked[:micro], use_cache=False, **keep)
        torch.cuda.synchronize()
        begin = time.perf_counter()
        for i in range(0, stacked.shape[0], micro):
            model(input_ids=stacked[i:i + micro], use_cache=False, **keep)
        torch.cuda.synchronize()
        prefill_s = time.perf_counter() - begin
        del stacked
        torch.cuda.empty_cache()

        decode = []
        gen = {"do_sample": False, "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id}
        for batch in DECODE_BATCHES:
            prompts = torch.stack(
                [ids[i * DECODE_PROMPT_TOKENS:(i + 1) * DECODE_PROMPT_TOKENS] for i in range(batch)]
            ).to(device)
            mask = torch.ones_like(prompts)
            model.generate(input_ids=prompts[:1], attention_mask=mask[:1], max_new_tokens=8, **gen)
            torch.cuda.synchronize()
            begin = time.perf_counter()
            out = model.generate(
                input_ids=prompts, attention_mask=mask,
                max_new_tokens=DECODE_NEW_TOKENS, min_new_tokens=DECODE_NEW_TOKENS, **gen,
            )
            torch.cuda.synchronize()
            seconds = time.perf_counter() - begin
            generated = (out.shape[1] - DECODE_PROMPT_TOKENS) * batch
            decode.append({"batch": batch, "tokens_per_s": generated / seconds, "generated_tokens": generated})
            torch.cuda.empty_cache()

    return {
        "runtime": "transformers",
        "quantized": bool(adapter),
        "dtype": dtype,
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "perplexity": math.exp(nll / counted),
        "tokens_evaluated": counted,
        "windows": len(windows),
        "prefill": {"tokens_per_s": len(windows) * MAX_SEQ_LEN / prefill_s, "micro_batch": micro},
        "decode": decode,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
    }


def _decode_map(side: dict) -> dict[int, float]:
    return {int(row["batch"]): float(row["tokens_per_s"]) for row in side.get("decode") or []}


def _speedups(fast: dict, slow: dict) -> tuple[float, dict[str, float]]:
    prefill = float(fast["prefill"]["tokens_per_s"]) / float(slow["prefill"]["tokens_per_s"])
    slow_decode = _decode_map(slow)
    decode = {str(b): v / slow_decode[b] for b, v in sorted(_decode_map(fast).items()) if slow_decode.get(b)}
    return prefill, decode


def compare_hf_runtime(report: dict, quantized: dict, baseline: dict) -> dict:
    """The `hf_runtime` block for a passed serve report. Pure."""
    if report.get("status") != "passed":
        raise ValueError("attach plain-Transformers numbers to a passed serve check only")
    if not quantized.get("quantized") or baseline.get("quantized"):
        raise ValueError("pass the native artifact's measurement as quantized and the original model's as baseline")
    vllm_quant = report.get("quantized") or {}
    gpus = {vllm_quant.get("gpu_name"), quantized.get("gpu_name"), baseline.get("gpu_name")}
    if len(gpus) != 1 or None in gpus:
        raise ValueError(f"all sides must be measured on one GPU, got {sorted(map(str, gpus))}")
    serve_dtype = report.get("dtype") or "float16"
    if baseline.get("dtype") != serve_dtype:
        raise ValueError(
            f"baseline dtype {baseline.get('dtype')!r} differs from the serve check's {serve_dtype!r}"
        )
    for side in (quantized, baseline):
        if not math.isfinite(float(side["perplexity"])) or float(side["perplexity"]) <= 0:
            raise ValueError("perplexity must be finite and > 0")
    int8_prefill, int8_decode = _speedups(quantized, baseline)
    vllm_prefill, vllm_decode = _speedups(vllm_quant, baseline)
    return {
        "gpu_name": baseline["gpu_name"],
        "dtype": serve_dtype,
        "torch_version": baseline.get("torch_version"),
        "transformers_version": baseline.get("transformers_version"),
        "quantized": {k: quantized[k] for k in _SIDE_KEYS if k in quantized},
        "baseline": {k: baseline[k] for k in _SIDE_KEYS if k in baseline},
        "comparison": {
            "ppl_ratio": float(quantized["perplexity"]) / float(baseline["perplexity"]),
            "int8_prefill_speedup": int8_prefill,
            "int8_decode_speedup": int8_decode,
            "vllm_prefill_speedup": vllm_prefill,
            "vllm_decode_speedup": vllm_decode,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Plain-Transformers numbers next to the vLLM serve check")
    sub = parser.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure", help="Measure one checkpoint (method venv python, GPU)")
    m.add_argument("--model-dir", required=True)
    m.add_argument("--eval-path", required=True, help=".cache/eval/wikitext-2-raw-v1-test.jsonl")
    m.add_argument("--dtype", default="float16", choices=("float16", "bfloat16"))
    m.add_argument("--adapter", help="quant_agent_inference_adapter.py (native quantized artifact)")
    m.add_argument("--out", type=Path, required=True)
    a = sub.add_parser("attach", help="Add hf_runtime to a passed serve report")
    a.add_argument("--report", type=Path, required=True, help="serve_check report, updated in place")
    a.add_argument("--quantized", type=Path, required=True, help="measure output for the native artifact")
    a.add_argument("--baseline", type=Path, required=True, help="measure output for the original model")
    args = parser.parse_args()
    try:
        if args.cmd == "measure":
            result = measure(args.model_dir, args.eval_path, args.dtype, args.adapter)
            atomic_write_text(args.out, json.dumps(result, indent=2) + "\n", mode=0o644)
            print(json.dumps({k: result[k] for k in ("perplexity", "prefill", "decode", "peak_vram_gb")}, indent=2))
            return 0
        report = json.loads(args.report.read_text())
        report["hf_runtime"] = compare_hf_runtime(
            report, json.loads(args.quantized.read_text()), json.loads(args.baseline.read_text())
        )
        atomic_write_text(args.report, json.dumps(report, indent=2) + "\n", mode=0o644)
        print(json.dumps(report["hf_runtime"]["comparison"], indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001 — report the exact failure as JSON
        print(json.dumps({"status": "failed", "message": str(exc)}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

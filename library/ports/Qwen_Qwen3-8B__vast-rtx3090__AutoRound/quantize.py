#!/usr/bin/env python
# QUANT_AGENT_OVERLAY_DIR=<overlay>
"""AutoRound (SignRound, arXiv 2309.05516) W4A16 for Qwen/Qwen3-8B (plan repo_api_qwen3).

Calls auto-round's public API unchanged (the overlay patch adds only a README):
  AutoRound(model, scheme="W4A16" -> int4, group 128, full-range sym (D7),
            iters 200, nsamples 128, seqlen 2048, batch 8, lr 1/iters, seed 42,
            device_map 0, low_gpu_mem_usage False (D9))
  quantize_and_save(format="auto_round")  -> GPTQ-packed QuantLinear (D1)
The repo nests the export in <model>-w4g128/; its files are moved up so OUTPUT_DIR
holds config.json + quantization_config.json + safetensors (D2). The model is loaded
the way the repo loads it (dtype "auto" = bf16 for Qwen3; AMP bf16 during tuning).
Calibration (D8): NeelNanda/pile-10k (repo default, shuffle seed 42) when a timed Hub
probe succeeds, else the first texts with >= seqlen tokens read line by line from the
local Pile validation val.jsonl. After saving, the artifact is reloaded in fp16 with
the tritonv2_zp kernel pinned (D3) for a finite-loss probe and a short generate.
"""
import gc
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

MODEL_ID = "Qwen/Qwen3-8B"
OUTPUT_DIR = "./quantized/autoround-qwen3-8b-vastrtx3090"

# Repo defaults for the "auto-round" recipe / paper section 4.1.
SCHEME = "W4A16"
BITS = 4
GROUP_SIZE = 128
SYM = True
ITERS = 200
NSAMPLES = 128
SEQLEN = 2048
BATCH_SIZE = 8
SEED = 42
DEVICE_MAP = 0
LOW_GPU_MEM_USAGE = False
EXPORT_FORMAT = "auto_round"
SMOKE_ITERS = 5  # QUANT_AGENT_SMOKE=1: a few SignSGD steps per block (lr auto = 1/iters)

CALIB_DATASET = "NeelNanda/pile-10k"
HUB_PROBE_TIMEOUT_S = 900
PILE_VAL_REL = (".cache/datasets/pile-val-backup/val.jsonl", ".cache/calib/pile-val-backup/val.jsonl")
PILE_VAL_MAX_LINES = 200000

RELOAD_BACKEND = "auto_round:tritonv2_zp"
RELOAD_DTYPE = torch.float16  # pipeline verify/benchmark dtype
PROBE_TEXT = (
    "The tower is 324 metres tall, about the same height as an 81-storey building, and the "
    "tallest structure in Paris. Its base is square, measuring 125 metres on each side."
)
GEN_PROMPT = "The capital of France is"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [quantize] %(message)s")
log = logging.getLogger("quantport.autoround")


def _env_int(name):
    raw = os.environ.get(name)
    return int(raw) if raw else None


def _use_method_repo():
    repo = os.environ.get("QUANT_AGENT_METHOD_REPO")
    if repo and repo not in sys.path:
        sys.path.insert(0, repo)
    return repo


def _workspace_roots():
    roots = [Path.cwd()]
    overlay = os.environ.get("QUANT_AGENT_OVERLAY_DIR")
    if overlay:
        parents = Path(overlay).resolve().parents  # out/overlays/<slug>/<plan>/<hash>
        if len(parents) > 4:
            roots.append(parents[4])
    return roots


def _pile_val_jsonl():
    for root in _workspace_roots():
        for rel in PILE_VAL_REL:
            path = root / rel
            if path.is_file():
                return path.resolve()
    return None


def _hub_dataset_ok():
    """Load NeelNanda/pile-10k once in a child process (fills the datasets cache)."""
    code = (
        "from datasets import load_dataset\n"
        f"ds = load_dataset({CALIB_DATASET!r}, split='train')\n"
        "print('rows', len(ds))\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=HUB_PROBE_TIMEOUT_S
        )
    except subprocess.TimeoutExpired:
        log.warning("calibration: %s Hub probe timed out after %ss", CALIB_DATASET, HUB_PROBE_TIMEOUT_S)
        return False
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        log.warning("calibration: %s Hub probe failed: %s", CALIB_DATASET, " | ".join(tail))
        return False
    log.info("calibration: %s Hub probe ok (%s)", CALIB_DATASET, proc.stdout.strip())
    return True


def _pile_val_texts(model_src, nsamples, seqlen):
    """First `nsamples` Pile-validation texts (file order) that tokenize to >= seqlen tokens."""
    from transformers import AutoTokenizer

    path = _pile_val_jsonl()
    if path is None:
        raise RuntimeError(
            f"{CALIB_DATASET} is unreachable and no local Pile validation file found under {PILE_VAL_REL}"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_src)
    texts, scanned = [], 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            scanned += 1
            if scanned > PILE_VAL_MAX_LINES or len(texts) >= nsamples:
                break
            try:
                text = json.loads(line).get("text") or ""
            except ValueError:
                continue
            if len(text) < seqlen:  # cheap prefilter; >= 1 char per token
                continue
            if len(tokenizer(text, truncation=True, max_length=seqlen).input_ids) >= seqlen:
                texts.append(text)
    if not texts:
        raise RuntimeError(f"no Pile validation text in {path} reaches {seqlen} tokens")
    if len(texts) < nsamples:
        log.warning("calibration: only %d/%d Pile-val texts reach %d tokens", len(texts), nsamples, seqlen)
    log.info("calibration: local Pile validation %s, first %d texts >= %d tokens (%d lines scanned)",
             path, len(texts), seqlen, scanned)
    return texts


def _calibration(model_src, nsamples, seqlen):
    if _hub_dataset_ok():
        log.info("calibration: using %s (repo loader, train split, shuffle seed %d)", CALIB_DATASET, SEED)
        return CALIB_DATASET, CALIB_DATASET
    log.warning("calibration: falling back to local Pile validation texts")
    return _pile_val_texts(model_src, nsamples, seqlen), "pile-val-backup(local val.jsonl)"


def _time_blocks():
    """Log wall time per SignRound block (instrumentation only)."""
    try:
        from auto_round.algorithms.quantization.sign_round.quantizer import SignRoundQuantizer
    except ImportError:
        return []
    times = []
    original = SignRoundQuantizer.quantize_block

    def timed(self, *args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.monotonic()
        try:
            return original(self, *args, **kwargs)
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.monotonic() - start)
            log.info("block %d tuned in %.1fs (peak alloc %.2f GB)", len(times), times[-1],
                     torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0)

    SignRoundQuantizer.quantize_block = timed
    return times


def _flatten_export(folder, out_dir):
    """D2: move the files of <out_dir>/_autoround_export/<model>-w4g128 into out_dir."""
    folder = Path(folder).resolve()
    out_dir = Path(out_dir).resolve()
    if folder == out_dir:
        return
    for entry in sorted(folder.iterdir()):
        dest = out_dir / entry.name
        if dest.exists():
            shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
        shutil.move(str(entry), str(dest))


def _check_saved_config(out_dir):
    config = json.loads((Path(out_dir) / "config.json").read_text())
    qcfg = config.get("quantization_config") or {}
    expected = {
        "quant_method": "auto-round",
        "packing_format": "auto_round:auto_gptq",
        "bits": BITS,
        "group_size": GROUP_SIZE,
        "sym": SYM,
    }
    bad = {k: qcfg.get(k) for k, v in expected.items() if qcfg.get(k) != v}
    if bad:
        raise RuntimeError(f"config.json quantization_config mismatch: {bad} (expected {expected})")
    act_bits = qcfg.get("act_bits")
    if isinstance(act_bits, int) and act_bits < 16:
        raise RuntimeError(f"weight-only run saved act_bits={act_bits}")
    side = Path(out_dir) / "quantization_config.json"
    if side.is_file():
        side_cfg = json.loads(side.read_text())
        clash = {k: side_cfg[k] for k in expected if k in side_cfg and side_cfg[k] != qcfg.get(k)}
        if clash:
            raise RuntimeError(f"quantization_config.json contradicts config.json: {clash}")
    log.info("config.json quantization_config: %s", json.dumps(qcfg, sort_keys=True))
    log.info("artifact files: %s", sorted(p.name for p in Path(out_dir).iterdir()))


def _reload_check(out_dir, dtype):
    from transformers import AutoModelForCausalLM, AutoRoundConfig, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(out_dir)
    model = AutoModelForCausalLM.from_pretrained(
        out_dir,
        quantization_config=AutoRoundConfig(backend=RELOAD_BACKEND),
        torch_dtype=dtype,
        device_map="cuda:0" if torch.cuda.is_available() else None,
    ).eval()
    kernels = {}
    for name, module in model.named_modules():
        if hasattr(module, "qweight"):
            key = f"{type(module).__module__}.{type(module).__name__}"
            kernels[key] = kernels.get(key, 0) + 1
    if not kernels:
        raise RuntimeError("reload produced no packed QuantLinear modules")
    log.info("reload (%s) backend %s -> kernels %s", dtype, RELOAD_BACKEND, kernels)
    device = model.get_input_embeddings().weight.device
    with torch.inference_mode():
        ids = tokenizer(PROBE_TEXT, return_tensors="pt").input_ids.to(device)
        loss = model(input_ids=ids, labels=ids).loss.float().item()
        if not math.isfinite(loss):
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return loss
        log.info("reload probe: loss %.4f, ppl %.2f on %d tokens", loss, math.exp(min(loss, 50)), ids.shape[-1])
        prompt = tokenizer(GEN_PROMPT, return_tensors="pt").to(device)
        out = model.generate(**prompt, max_new_tokens=16, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    new = out[0, prompt["input_ids"].shape[-1]:]
    if new.numel() < 1:
        raise RuntimeError("reloaded model generated no tokens")
    log.info("reload generate: %r", tokenizer.decode(new, skip_special_tokens=True))
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return loss


def main():
    method_repo = _use_method_repo()
    smoke = os.environ.get("QUANT_AGENT_SMOKE") == "1"
    model_src = os.environ.get("QUANT_AGENT_SMOKE_MODEL_DIR") or MODEL_ID
    out_dir = os.environ.get("QUANT_AGENT_SMOKE_OUTPUT_DIR") or OUTPUT_DIR
    nsamples = _env_int("QUANT_AGENT_SMOKE_CALIB_SAMPLES") or NSAMPLES
    seqlen = _env_int("QUANT_AGENT_SMOKE_SEQ_LEN") or SEQLEN
    iters = SMOKE_ITERS if smoke else ITERS
    batch_size = min(BATCH_SIZE, nsamples)

    import auto_round
    from auto_round import AutoRound

    log.info("auto_round %s from %s (QUANT_AGENT_METHOD_REPO=%s)", auto_round.__version__,
             os.path.dirname(auto_round.__file__), method_repo)
    log.info("model %s -> %s | scheme %s iters %d nsamples %d seqlen %d batch %d seed %d "
             "low_gpu_mem_usage %s smoke %s", model_src, out_dir, SCHEME, iters, nsamples, seqlen,
             batch_size, SEED, LOW_GPU_MEM_USAGE, smoke)

    dataset, dataset_label = _calibration(model_src, nsamples, seqlen)

    out_path = Path(out_dir)
    if out_path.exists():
        log.info("removing previous contents of %s", out_path)
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True)
    stage = out_path / "_autoround_export"

    block_times = _time_blocks()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    ar = AutoRound(
        model_src,
        scheme=SCHEME,
        dataset=dataset,
        iters=iters,
        nsamples=nsamples,
        seqlen=seqlen,
        batch_size=batch_size,
        seed=SEED,
        device_map=DEVICE_MAP,
        low_gpu_mem_usage=LOW_GPU_MEM_USAGE,
    )
    model = getattr(ar, "model", None)
    log.info("loaded model dtype %s (calibration: %s)", getattr(model, "dtype", "?"), dataset_label)
    _, folders = ar.quantize_and_save(output_dir=str(stage), format=EXPORT_FORMAT)
    elapsed = time.monotonic() - started
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    folder = folders if isinstance(folders, str) else folders[0]
    log.info("quantize_and_save done in %.1fs, peak GPU alloc %.2f GB, export folder %s", elapsed, peak, folder)
    if block_times:
        log.info("blocks tuned: %d, mean %.1fs, total %.1fs", len(block_times),
                 sum(block_times) / len(block_times), sum(block_times))
    del ar, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _flatten_export(folder, out_path)
    shutil.rmtree(stage, ignore_errors=True)
    _check_saved_config(out_path)
    loss = _reload_check(str(out_path), RELOAD_DTYPE)
    if not math.isfinite(loss):
        alt = _reload_check(str(out_path), torch.bfloat16)  # diagnose fp16 overflow vs broken weights
        raise RuntimeError(f"reloaded artifact gives non-finite loss in {RELOAD_DTYPE} ({loss}); "
                           f"bfloat16 loss {alt}")
    log.info("done: %s", out_path.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())

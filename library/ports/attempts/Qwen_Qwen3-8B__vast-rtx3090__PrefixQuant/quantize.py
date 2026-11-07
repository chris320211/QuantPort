#!/usr/bin/env python
# QUANT_AGENT_OVERLAY_DIR=<overlay>
"""PrefixQuant-O1 W4A4KV4 (dynamic) for Qwen/Qwen3-8B (plan diagnose_fix, from repo_pipeline_qwen3).

Why O1, not O2: on Qwen3-8B layer 16 the down_proj input has token-wise outliers triggered by
the content token "2" (id 17) at arbitrary positions (top-1/median up to ~2000). A prefix cannot
absorb a content-triggered outlier, so the O2 per-tensor static A4 scale of that quantizer
collapses (block 16 MSE-init loss 9.5 vs 0.0015 at block 15, block loss ~30). O1 (paper Table 1:
per-channel W, per-token dynamic A, group-wise dynamic KV) computes the range per token, plus
the paper's learnable per-layer activation clipping (--activation_clipping), D16.

The repo's own pipeline (main.quantize_model, patched for Qwen3 on transformers 4.51):
fuse residual norms by name (q_norm/k_norm untouched, D2) and rotate (R1/R2, R4 weight half)
-> QuantLinear/QuantRMSNorm wrap -> R4 online Hadamard on down_proj -> R3 wrapper after
q_norm/k_norm + RoPE (D1) -> prefixed-token detection on Pile (generic start token, zero-outlier
fallback, D5) -> fp prefill of the prefix KV (plain tensors, D4) -> activation stats with the
prefix -> block_ap with mse_init (weight scales; clipping factors of the dynamic quantizers) and
20 epochs (position_embeddings and a fresh prefix DynamicCache in every block call, D3/D4/D9)
-> quant_inplace -> save.

Flags (README "For dynamic quantization, you should add --activation_clipping"; paper Table 1 O1,
group size 128 = Qwen3 head_dim): --wbits 4 --w_group_size -1 --input_bits 4 --input_mode dynamic
--k_bits 4 --v_bits 4 --kv_group_size 128 --kv_mode dynamic --activation_clipping --mse_init
--pre_rotate --down_online_had --qk_online_had --set_prefixed_tokens --epochs 20
(quant_lr 5e-5, weight_lr 5e-6, 512 train + 64 val Pile x 1024, batch 4). The repo gives the
dynamic K quantizer no clipping factor (init_k_quantizer); V, norm outputs, o_proj/down_proj
inputs get one each. PREFIXQUANT_EPOCHS=<int> overrides the epoch count (0 = mse_init only).
The artifact (fake-quant fp16 weights + weight scales + clipping factors + prefix KV; no int4
kernel exists, D14) reloads through quant_agent_inference_adapter.py (D6).
"""
import gc
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

MODEL_ID = "Qwen/Qwen3-8B"
OUTPUT_DIR = "./quantized/prefixquant-qwen3-8b-vastrtx3090-o1-e0"
REPO_URL = "https://github.com/ChenMnZ/PrefixQuant"
REPO_COMMIT = "60b74237cf2cf20e3354569d998ab708ff598ce3"

# Full-run calibration (paper/README O2 recipe)
TRAIN_SIZE = 512
VAL_SIZE = 64
TRAINING_SEQLEN = 1024
BATCH_SIZE = 4
EPOCHS = 0  # no-fine-tuning sanity pass
MSE_INIT_SIZE = 8
STAT_SIZE = 64        # prefixed-token detection + activation statistics
STAT_SEQLEN = 512
# Smoke (QUANT_AGENT_SMOKE=1): same flags, a few optimizer steps per block
SMOKE_EPOCHS = 2
# Explicit epoch override (int, 0 allowed: MSE/clipping init only); wins over EPOCHS and SMOKE_EPOCHS
EPOCHS_ENV = "PREFIXQUANT_EPOCHS"

PILE_REL_DIR = ".cache/datasets/pile-val-backup"
PILE_REPO = "mit-han-lab/pile-val-backup"


def _env_int(name):
    raw = os.environ.get(name)
    return int(raw) if raw else None


def _workspace_roots():
    roots = [Path.cwd()]
    overlay = os.environ.get("QUANT_AGENT_OVERLAY_DIR")
    if overlay:
        parents = Path(overlay).resolve().parents  # out/overlays/<slug>/<plan>/<hash>
        if len(parents) > 4:
            roots.append(parents[4])
    return roots


def _pile_jsonl():
    """Decompressed val.jsonl of mit-han-lab/pile-val-backup under .cache (downloaded once)."""
    for root in _workspace_roots():
        path = root / PILE_REL_DIR / "val.jsonl"
        if path.is_file():
            return path.resolve()
    dest = _workspace_roots()[-1] / PILE_REL_DIR
    dest.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download

    zst = hf_hub_download(PILE_REPO, "val.jsonl.zst", repo_type="dataset", local_dir=str(dest))
    tmp = dest / f"val.jsonl.{os.getpid()}.tmp"
    subprocess.run(["zstd", "-d", "-f", "-q", "-o", str(tmp), zst], check=True)
    os.replace(tmp, dest / "val.jsonl")
    return (dest / "val.jsonl").resolve()


def _argv(model_src, model_name, output_dir, cache_dir, log_dir, sizes):
    return [
        "--model_path", model_src,
        "--model_name", model_name,
        "--cache_dir", cache_dir,
        "--output_dir", log_dir,
        "--save_quant_dir", output_dir,
        "--wbits", "4", "--w_group_size", "-1",
        "--input_bits", "4", "--input_mode", "dynamic",
        "--k_bits", "4", "--v_bits", "4", "--kv_group_size", "128", "--kv_mode", "dynamic",
        "--activation_clipping",
        "--mse_init", "--mse_init_size", str(sizes["mse_init_size"]),
        "--pre_rotate", "--down_online_had", "--qk_online_had", "--set_prefixed_tokens",
        "--epochs", str(sizes["epochs"]),
        "--quant_lr", "5e-5", "--weight_lr", "5e-6",
        "--train_size", str(sizes["train_size"]), "--val_size", str(sizes["val_size"]),
        "--training_seqlen", str(sizes["training_seqlen"]), "--batch_size", str(BATCH_SIZE),
        "--calib_dataset", "pile",
        "--stat_size", str(sizes["stat_size"]), "--stat_seqlen", str(sizes["stat_seqlen"]),
        "--max_memory", "22GiB",
        "--use_fast_tokenizer",
    ]


def _reload_check(output_dir, probe_ids):
    """Reopen the saved artifact through the inference adapter: finite loss + greedy generate."""
    import importlib.util

    adapter_path = Path(os.environ["QUANT_AGENT_METHOD_REPO"]) / "quant_agent_inference_adapter.py"
    spec = importlib.util.spec_from_file_location("quant_agent_inference_adapter", adapter_path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    model, tokenizer = adapter.load_model_and_tokenizer(
        model_path=output_dir, model_id=MODEL_ID, dtype="float16", device="cuda", trust_remote_code=False
    )
    ids = probe_ids.to("cuda")
    with torch.inference_mode():
        loss = model(input_ids=ids, labels=ids).loss.float().item()
        prompt = ids[:, :32]
        out = model.generate(input_ids=prompt, max_new_tokens=8, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    if not torch.equal(out[:, : prompt.shape[1]], prompt):
        raise RuntimeError("generate output does not start with the caller's input_ids")
    check = {
        "probe_tokens": int(ids.shape[-1]),
        "probe_loss": loss,
        "finite": bool(torch.isfinite(torch.tensor(loss))),
        "generated": tokenizer.decode(out[0, prompt.shape[1]:], skip_special_tokens=True),
        "runtime": getattr(model, "quant_agent_runtime", None),
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return check


def main():
    method_repo = os.environ.get("QUANT_AGENT_METHOD_REPO")
    if not method_repo:
        raise SystemExit("QUANT_AGENT_METHOD_REPO is not set; run through the launcher")
    print(f"QUANT_AGENT_OVERLAY_DIR={os.environ.get('QUANT_AGENT_OVERLAY_DIR')}", flush=True)
    sys.path.insert(0, method_repo)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required (fast_hadamard_transform, block-wise training)")

    model_src = os.environ.get("QUANT_AGENT_SMOKE_MODEL_DIR") or MODEL_ID
    output_dir = os.environ.get("QUANT_AGENT_SMOKE_OUTPUT_DIR") or OUTPUT_DIR
    smoke = os.environ.get("QUANT_AGENT_SMOKE") == "1"
    calib = _env_int("QUANT_AGENT_SMOKE_CALIB_SAMPLES")
    seqlen = _env_int("QUANT_AGENT_SMOKE_SEQ_LEN")

    sizes = {
        "train_size": TRAIN_SIZE, "val_size": VAL_SIZE, "training_seqlen": TRAINING_SEQLEN,
        "epochs": EPOCHS, "mse_init_size": MSE_INIT_SIZE,
        "stat_size": STAT_SIZE, "stat_seqlen": STAT_SEQLEN,
    }
    if calib:
        sizes["train_size"] = max(BATCH_SIZE, calib // BATCH_SIZE * BATCH_SIZE)
        sizes["val_size"] = BATCH_SIZE
        sizes["stat_size"] = min(STAT_SIZE, calib)
        sizes["mse_init_size"] = min(MSE_INIT_SIZE, sizes["train_size"])
    if seqlen:
        sizes["training_seqlen"] = seqlen
        sizes["stat_seqlen"] = min(STAT_SEQLEN, seqlen)
    if smoke:
        sizes["epochs"] = SMOKE_EPOCHS
    epochs_override = _env_int(EPOCHS_ENV)
    if epochs_override is not None:
        if epochs_override < 0:
            raise SystemExit(f"{EPOCHS_ENV} must be >= 0, got {epochs_override}")
        sizes["epochs"] = epochs_override
    # mse_init batches and the epochs=0 path (train_size := mse_init_size) need whole batches
    sizes["mse_init_size"] = max(BATCH_SIZE, sizes["mse_init_size"] // BATCH_SIZE * BATCH_SIZE)
    steps_per_block = sizes["epochs"] * (sizes["train_size"] // BATCH_SIZE)
    print(f"epochs={sizes['epochs']} (source: "
          f"{EPOCHS_ENV if epochs_override is not None else ('smoke' if smoke else 'default')})", flush=True)

    os.environ["PREFIXQUANT_PILE_JSONL"] = str(_pile_jsonl())
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.makedirs(output_dir, exist_ok=True)
    work = tempfile.mkdtemp(prefix="prefixquant-")
    model_name = MODEL_ID.split("/")[-1]
    print(f"model={model_src} output={output_dir} smoke={smoke} sizes={json.dumps(sizes)} "
          f"optimizer_steps_per_block={steps_per_block} pile={os.environ['PREFIXQUANT_PILE_JSONL']}", flush=True)

    import main as pq_main
    from utils import train_utils
    from utils.data_utils import get_loaders

    args = pq_main.build_parser().parse_args(
        _argv(model_src, model_name, output_dir, os.path.join(work, "cache"), os.path.join(work, "log"), sizes)
    )
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    pq_main.seed_everything(args.seed)
    logger = train_utils.create_logger(Path(args.output_dir))
    logger.info(args)

    started = time.time()
    model, tokenizer, prefixed_key_values, prefixed_tokens = pq_main.quantize_model(args, logger)
    model.half()
    pq_main.model_utils.restore_rope_fp32(model)
    torch.cuda.empty_cache()
    quant_seconds = time.time() - started
    extra = {
        "method": "PrefixQuant",
        "setting": "PrefixQuant-O1 W4A4KV4: per-channel W4 sym, per-token dynamic A4 sym with learnable "
                   "per-layer clipping, group-wise (128 = head_dim) dynamic KV4 sym (V clipped, K unclipped)",
        "base_model": MODEL_ID,
        "repo_url": REPO_URL,
        "repo_commit": REPO_COMMIT,
        "smoke": smoke,
        "calibration": {
            "dataset": PILE_REPO, "seed": args.seed,
            "stat_samples": sizes["stat_size"], "stat_seqlen": sizes["stat_seqlen"],
            # quantize_model sets train_size := mse_init_size and val_size := 0 when epochs == 0
            "train_size": args.train_size, "val_size": args.val_size,
            "training_seqlen": sizes["training_seqlen"], "batch_size": BATCH_SIZE,
            "epochs": sizes["epochs"], "optimizer_steps_per_block": steps_per_block,
            "mse_init_size": sizes["mse_init_size"], "quant_lr": args.quant_lr, "weight_lr": args.weight_lr,
        },
        "prefixed_tokens_text": tokenizer.decode(prefixed_tokens) if prefixed_tokens else None,
        "quantize_seconds": round(quant_seconds, 1),
        "artifact": "fake-quant fp16 weights + weight scales + activation/V clipping factors + fp16 prefix KV; "
                    "reload with quant_agent_inference_adapter.py",
    }
    quant_config = pq_main.save_quant_model(model, tokenizer, prefixed_key_values, prefixed_tokens, args, logger,
                                            extra_config=extra)
    print("prefix " + json.dumps({"tokens": prefixed_tokens, "text": extra["prefixed_tokens_text"],
                                  "ratio_without": quant_config.get("outlier_ratio_without_prefix"),
                                  "ratio_with": quant_config.get("outlier_ratio_with_prefix")}), flush=True)

    # one held-out Pile window for the reload self-check
    _, probe_loader = get_loaders("pile", tokenizer, 0, 1, seed=args.seed + 1, seqlen=min(256, sizes["training_seqlen"]))
    probe_ids = probe_loader[0][0]
    del model, prefixed_key_values
    gc.collect()
    torch.cuda.empty_cache()

    check = _reload_check(output_dir, probe_ids)
    print("reload_check " + json.dumps(check), flush=True)
    if not check["finite"]:
        raise SystemExit("reloaded artifact produced a non-finite loss")
    summary = dict(quant_config)
    summary["reload_check"] = check
    summary["publish_exclude"] = []
    with open(os.path.join(output_dir, "quantization_config.json"), "w") as handle:
        json.dump(summary, handle, indent=2, default=str)
    shutil.rmtree(work, ignore_errors=True)
    print(f"done in {time.time() - started:.1f}s; artifact files: {sorted(os.listdir(output_dir))}", flush=True)


if __name__ == "__main__":
    main()

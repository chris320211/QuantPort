#!/usr/bin/env python
# QUANT_AGENT_OVERLAY_DIR=<overlay>
"""PrefixQuant-O1 W4A4KV4 for Qwen/Qwen3-8B, packed to real integers (plan kernel_triton).

Same quantization as the verified diagnose_fix e0 job (20261107T001415Z-5642d6: PrefixQuant-O1,
W4 per-channel sym, per-token dynamic A4 sym with learned clipping, group-wise dynamic KV4,
R1/R2 pre-rotation, R4 online Hadamard on down_proj, R3 Hadamard on q/k after RoPE, prefix KV of
3 tokens, EPOCHS = 0 i.e. mse_init only, seed 2). That job's artifact stores fp16 fake-quant
weights and runs every quantizer in fp16 torch ops (PPL 11.14, 2650 tok/s, 18.5 GB).

This wrapper writes the integer artifact (format ``w4a4_int8_storage``):
1. source: the verified fake-quant e0 artifact at SOURCE_DIR when it exists and its
   prefixequant_config matches these settings; otherwise (and always in the smoke run) the
   repo pipeline quantizes again into a temporary fake-quant directory first.
2. ``utils.packed_runtime.convert_artifact``: W4 codes as int8 ``weight`` + fp16 per-channel
   ``weight_scale`` (every code is checked to reproduce its fp16 weight bit for bit), clipping
   factors, fp16 prefix KV and configs copied; embeddings / lm_head / norms stay fp16.
3. reload through the overlay adapter (packed path: meta-device build, integer weights copied
   straight to the GPU, Triton int8 GEMM with per-token A4 codes, fused RMSNorm+A4 kernel,
   Triton KV4 fake quant, prefix-aware flash SDPA), compare logits with the fake-quant source on
   a held-out Pile window, greedy generate, write quantization_config.json with the keys
   ``kernels.py match`` reads.
The repo's own ``--real_quant`` path is weight-only W4A16 (it asserts input/k/v bits >= 16 and
needs asymmetric zero points), so it cannot carry this W4A4KV4 setting.
"""
import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

MODEL_ID = "Qwen/Qwen3-8B"
OUTPUT_DIR = "./quantized/prefixquant-qwen3-8b-vastrtx3090"
# verified fake-quant artifact of job 20261107T001415Z-5642d6 (never modified here)
SOURCE_DIR = "./quantized/prefixquant-qwen3-8b-vastrtx3090-o1-e0"
REPO_URL = "https://github.com/ChenMnZ/PrefixQuant"
REPO_COMMIT = "60b74237cf2cf20e3354569d998ab708ff598ce3"

TRAIN_SIZE = 512
VAL_SIZE = 64
TRAINING_SEQLEN = 1024
BATCH_SIZE = 4
EPOCHS = 0  # mse_init only, as the verified e0 run
MSE_INIT_SIZE = 8
STAT_SIZE = 64
STAT_SEQLEN = 512
SEED = 2
SMOKE_EPOCHS = 2
PROBE_SEQLEN = 256
# a packed probe loss this far above the fake-quant one means a kernel bug, not rounding noise
MAX_PROBE_LOSS_GAP = 0.5

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
        "--seed", str(SEED),
    ]


def _sizes(smoke):
    calib = _env_int("QUANT_AGENT_SMOKE_CALIB_SAMPLES")
    seqlen = _env_int("QUANT_AGENT_SMOKE_SEQ_LEN")
    sizes = {
        "train_size": TRAIN_SIZE, "val_size": VAL_SIZE, "training_seqlen": TRAINING_SEQLEN,
        "epochs": SMOKE_EPOCHS if smoke else EPOCHS, "mse_init_size": MSE_INIT_SIZE,
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
    sizes["mse_init_size"] = max(BATCH_SIZE, sizes["mse_init_size"] // BATCH_SIZE * BATCH_SIZE)
    return sizes


def _find_source():
    """The verified e0 fake-quant artifact, if present and built with these settings."""
    for root in _workspace_roots():
        path = (root / SOURCE_DIR).resolve()
        cfg_path = path / "prefixequant_config.json"
        if not cfg_path.is_file():
            continue
        cfg = json.loads(cfg_path.read_text())
        calib = cfg.get("calibration") or {}
        want = {"wbits": 4, "w_group_size": -1, "input_bits": 4, "input_mode": "dynamic", "k_bits": 4,
                "v_bits": 4, "kv_group_size": 128, "kv_mode": "dynamic", "activation_clipping": True,
                "down_online_had": True, "qk_online_had": True, "set_prefixed_tokens": True,
                "base_model": MODEL_ID, "smoke": False}
        diff = {k: cfg.get(k) for k, v in want.items() if cfg.get(k) != v}
        if calib.get("epochs") != EPOCHS or calib.get("seed") != SEED:
            diff["calibration"] = {"epochs": calib.get("epochs"), "seed": calib.get("seed")}
        if diff:
            print(f"source {path} does not match these settings ({diff}); re-quantizing", flush=True)
            continue
        return path
    return None


def _quantize_fake(model_src, fake_dir, smoke):
    """The repo pipeline (as the verified e0 wrapper) -> fake-quant artifact in fake_dir."""
    import main as pq_main
    from utils import train_utils

    sizes = _sizes(smoke)
    os.environ["PREFIXQUANT_PILE_JSONL"] = str(_pile_jsonl())
    work = tempfile.mkdtemp(prefix="prefixquant-")
    args = pq_main.build_parser().parse_args(
        _argv(model_src, MODEL_ID.split("/")[-1], fake_dir, os.path.join(work, "cache"), os.path.join(work, "log"), sizes)
    )
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    pq_main.seed_everything(args.seed)
    logger = train_utils.create_logger(Path(args.output_dir))
    logger.info(args)
    print(f"quantizing {model_src} -> {fake_dir} sizes={json.dumps(sizes)}", flush=True)
    started = time.time()
    model, tokenizer, prefixed_key_values, prefixed_tokens = pq_main.quantize_model(args, logger)
    model.half()
    pq_main.model_utils.restore_rope_fp32(model)
    torch.cuda.empty_cache()
    extra = {
        "method": "PrefixQuant",
        "setting": "PrefixQuant-O1 W4A4KV4: per-channel W4 sym, per-token dynamic A4 sym with learnable "
                   "per-layer clipping, group-wise (128 = head_dim) dynamic KV4 sym (V clipped, K unclipped)",
        "base_model": MODEL_ID, "repo_url": REPO_URL, "repo_commit": REPO_COMMIT, "smoke": smoke,
        "calibration": {
            "dataset": PILE_REPO, "seed": args.seed, "stat_samples": sizes["stat_size"],
            "stat_seqlen": sizes["stat_seqlen"], "train_size": args.train_size, "val_size": args.val_size,
            "training_seqlen": sizes["training_seqlen"], "batch_size": BATCH_SIZE, "epochs": sizes["epochs"],
            "mse_init_size": sizes["mse_init_size"], "quant_lr": args.quant_lr, "weight_lr": args.weight_lr,
        },
        "prefixed_tokens_text": tokenizer.decode(prefixed_tokens) if prefixed_tokens else None,
        "quantize_seconds": round(time.time() - started, 1),
    }
    pq_main.save_quant_model(model, tokenizer, prefixed_key_values, prefixed_tokens, args, logger, extra_config=extra)
    del model, prefixed_key_values
    gc.collect()
    torch.cuda.empty_cache()
    shutil.rmtree(work, ignore_errors=True)


def _adapter():
    import importlib.util

    adapter_path = Path(os.environ["QUANT_AGENT_METHOD_REPO"]) / "quant_agent_inference_adapter.py"
    spec = importlib.util.spec_from_file_location("quant_agent_inference_adapter", adapter_path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    return adapter


def _probe_ids(model_dir):
    from transformers import AutoTokenizer
    from utils.data_utils import get_loaders

    os.environ.setdefault("PREFIXQUANT_PILE_JSONL", str(_pile_jsonl()))
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    seqlen = min(PROBE_SEQLEN, _env_int("QUANT_AGENT_SMOKE_SEQ_LEN") or PROBE_SEQLEN)
    _, loader = get_loaders("pile", tokenizer, 0, 1, seed=SEED + 1, seqlen=seqlen)
    return loader[0][0]


def _forward(model_dir, ids, generate):
    adapter = _adapter()
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer = adapter.load_model_and_tokenizer(
        model_path=str(model_dir), model_id=MODEL_ID, dtype="float16", device="cuda", trust_remote_code=False
    )
    torch.cuda.synchronize()
    out = {"load_peak_vram_gb": torch.cuda.max_memory_allocated() / 2 ** 30}
    ids = ids.to("cuda")
    with torch.inference_mode():
        res = model(input_ids=ids, labels=ids)
        out["loss"] = float(res.loss.float().item())
        out["logits"] = res.logits[0].float().cpu()
        if generate:
            prompt = ids[:, :32]
            gen = model.generate(input_ids=prompt, max_new_tokens=8, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            if not torch.equal(gen[:, : prompt.shape[1]], prompt):
                raise RuntimeError("generate output does not start with the caller's input_ids")
            out["generated"] = tokenizer.decode(gen[0, prompt.shape[1]:], skip_special_tokens=True)
    out["runtime"] = getattr(model, "quant_agent_runtime", None)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out


def _weight_bytes(path):
    return sum(p.stat().st_size for p in Path(path).glob("*.safetensors") if p.name != "prefixed_key_values.safetensors")


def main():
    method_repo = os.environ.get("QUANT_AGENT_METHOD_REPO")
    if not method_repo:
        raise SystemExit("QUANT_AGENT_METHOD_REPO is not set; run through the launcher")
    print(f"QUANT_AGENT_OVERLAY_DIR={os.environ.get('QUANT_AGENT_OVERLAY_DIR')}", flush=True)
    sys.path.insert(0, method_repo)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required (fast_hadamard_transform, Triton int8 GEMM)")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    smoke_model = os.environ.get("QUANT_AGENT_SMOKE_MODEL_DIR")
    output_dir = os.environ.get("QUANT_AGENT_SMOKE_OUTPUT_DIR") or OUTPUT_DIR
    smoke = os.environ.get("QUANT_AGENT_SMOKE") == "1" or bool(smoke_model)
    started = time.time()

    from utils import packed_runtime

    source = None if smoke else _find_source()
    temp_fake = None
    if source is None:
        parent = Path(output_dir).resolve().parent
        parent.mkdir(parents=True, exist_ok=True)
        temp_fake = Path(tempfile.mkdtemp(prefix=".prefixquant-fake-", dir=str(parent)))
        _quantize_fake(smoke_model or MODEL_ID, str(temp_fake), smoke)
        source = temp_fake
    print(f"source fake-quant artifact: {source} (reused={temp_fake is None})", flush=True)

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    conversion = packed_runtime.convert_artifact(str(source), output_dir)

    probe = _probe_ids(str(source))
    fake = _forward(source, probe, generate=False)
    packed = _forward(output_dir, probe, generate=True)
    a, b = packed.pop("logits"), fake.pop("logits")
    check = {
        "probe_tokens": int(probe.shape[-1]),
        "probe_loss": packed["loss"],
        "fake_quant_probe_loss": fake["loss"],
        "finite": bool(torch.isfinite(torch.tensor(packed["loss"]))),
        "rel_logit_error_vs_fake_quant": ((a - b).norm() / b.norm()).item(),
        "top1_agreement_vs_fake_quant": (a.argmax(-1) == b.argmax(-1)).float().mean().item(),
        "generated": packed.get("generated"),
        "load_peak_vram_gb": packed["load_peak_vram_gb"],
        "fake_quant_load_peak_vram_gb": fake["load_peak_vram_gb"],
        "runtime": packed.get("runtime"),
    }
    print("reload_check " + json.dumps(check), flush=True)
    if not check["finite"]:
        raise SystemExit("packed artifact produced a non-finite loss")
    if packed["loss"] > fake["loss"] + MAX_PROBE_LOSS_GAP:
        raise SystemExit(f"packed probe loss {packed['loss']:.3f} far above fake-quant {fake['loss']:.3f}")

    qcfg = packed_runtime.read_config(output_dir)
    summary = dict(qcfg)
    summary.update(packed_runtime.kernel_quant_config(qcfg))
    summary["source_artifact"] = str(source) if temp_fake is None else "re-quantized (temporary fake-quant dir)"
    summary["weight_file_bytes"] = _weight_bytes(output_dir)
    summary["source_weight_file_bytes"] = _weight_bytes(source)
    summary["conversion"] = conversion
    summary["reload_check"] = check
    summary["artifact"] = ("int8 W4 codes + fp16 per-channel scales for every decoder projection, fp16 "
                           "embeddings/lm_head/norms, learned clipping factors, fp16 prefix KV; reload with "
                           "quant_agent_inference_adapter.py (packed_runtime)")
    summary["publish_exclude"] = []
    with open(os.path.join(output_dir, "quantization_config.json"), "w") as handle:
        json.dump(summary, handle, indent=2, default=str)
    if temp_fake is not None:
        shutil.rmtree(temp_fake, ignore_errors=True)
    print(f"done in {time.time() - started:.1f}s; weights {summary['weight_file_bytes'] / 1e9:.2f} GB "
          f"(fake-quant {summary['source_weight_file_bytes'] / 1e9:.2f} GB); files: {sorted(os.listdir(output_dir))}",
          flush=True)


if __name__ == "__main__":
    main()

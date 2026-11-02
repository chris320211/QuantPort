#!/usr/bin/env python
# QUANT_AGENT_OVERLAY_DIR=<overlay>
"""FlatQuant W4A4KV4 for Qwen/Qwen2.5-1.5B-Instruct (plan qwen2_packed_deploy).

The repo's own Qwen2 pipeline, entered directly (D1: model_utils.get_qwen2 /
qwen_utils.apply_flatquant_to_qwen, no model-name dispatch):

  wikitext2 train (local parquet) -> cali_flat_quant (Qwen-2.5 script flags, D8:
  W4 per-channel sym, A4 per-token sym, KV4 asym, --lwc --lac --cali_trans --add_diag
  --deactive_amp --direct_inv, flat_lr 5e-3, 15 epochs, 128 x 2048) ->
  save_flat_matrices -> reparameterize_model (fold in fp32) -> RTN on the 7 decoder
  projections only (no InvSingleTransMatrix.linear) -> packed int4 export
  (--quantized_save path: pack_i4 + fp16 scales, D2) -> reload through
  deploy.transformers.modeling_qwen2.FlatQuantQwen2ForCausalLM (Linear4bit int4 GEMM,
  online Kronecker transforms, P_h after RoPE, torch P_o^T (D3), KV4, re-tied lm_head
  (D5)) and compare its logits with the in-memory fake-quant model.

The artifact reloads through quant_agent_inference_adapter.py.
"""
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "./quantized/flatquant-qwen25-15b-vastrtx3090"
REPO_COMMIT = "9d88ffcb7d2c6bda59fb5c44dad36adc101aadb1"

CALIB_SAMPLES = 128
CALIB_SEQ_LEN = 2048
EPOCHS = 15  # full run: 15 x (128 // 4) = 480 optimizer steps per layer (repo recipe)
# Smoke (D10): Qwen2.5's k_proj bias (absmax ~316 in layer 0) swamps per-token KV4 keys until the
# learned P_h and K clip have trained ~200+ steps per layer; 2-layer slice W4A4KV4 vs fp16:
#   ~4 steps -> rel 0.76;  240 steps (8 x 256) -> ~0.25;  480 steps (128 x 2048) -> ~0.17.
SMOKE_STEPS_PER_LAYER = 240
CALI_BSZ = 4
CHECK_TOKENS = 256
CALIB_REL_DIR = ".cache/calib/wikitext2"


def _env_int(name, default):
    raw = os.environ.get(name)
    return int(raw) if raw else default


def _calib_dir():
    candidates = [Path.cwd() / CALIB_REL_DIR]
    overlay = os.environ.get("QUANT_AGENT_OVERLAY_DIR")
    if overlay:
        # out/overlays/<slug>/<plan>/<hash> -> workspace root
        parents = Path(overlay).resolve().parents
        if len(parents) > 4:
            candidates.append(parents[4] / CALIB_REL_DIR)
    for path in candidates:
        if (path / "train.parquet").is_file():
            return path.resolve()
    raise FileNotFoundError("wikitext2 train.parquet not found at: " + ", ".join(str(p) for p in candidates))


def main():
    method_repo = os.environ.get("QUANT_AGENT_METHOD_REPO")
    if not method_repo:
        raise SystemExit("QUANT_AGENT_METHOD_REPO is not set; run through the launcher")
    print(f"QUANT_AGENT_OVERLAY_DIR={os.environ.get('QUANT_AGENT_OVERLAY_DIR')}", flush=True)
    sys.path.insert(0, method_repo)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required (calibration and the int4 kernels run on GPU)")

    model_src = os.environ.get("QUANT_AGENT_SMOKE_MODEL_DIR") or MODEL_ID
    output_dir = os.environ.get("QUANT_AGENT_SMOKE_OUTPUT_DIR") or OUTPUT_DIR
    smoke = os.environ.get("QUANT_AGENT_SMOKE") == "1"
    nsamples = _env_int("QUANT_AGENT_SMOKE_CALIB_SAMPLES", CALIB_SAMPLES)
    seqlen = _env_int("QUANT_AGENT_SMOKE_SEQ_LEN", CALIB_SEQ_LEN)
    cali_bsz = min(CALI_BSZ, nsamples)
    calib_dir = _calib_dir()
    # data_utils.get_wikitext2 reads these parquet files instead of ./datasets/wikitext.
    os.environ["FLATQUANT_WIKITEXT2_DIR"] = str(calib_dir)
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger("flatquant")

    from transformers import AutoTokenizer

    import flatquant.utils as fq_utils
    from flatquant import data_utils, flat_utils, model_utils, qwen2_packed, train_utils
    import gptq_utils

    epochs = qwen2_packed.epochs_for_steps(SMOKE_STEPS_PER_LAYER, nsamples, cali_bsz) if smoke else EPOCHS
    steps_per_layer = epochs * (nsamples // cali_bsz)
    print(
        f"model={model_src} output={output_dir} samples={nsamples} seqlen={seqlen} epochs={epochs} "
        f"optimizer_steps_per_layer={steps_per_layer} smoke={smoke} calib={calib_dir}",
        flush=True,
    )

    fq_utils.seed_everything(seed=0)
    dev = fq_utils.DEV
    started = time.time()

    # D1: the Qwen2 path directly (get_model's name check rejects this model id).
    model, apply_flatquant_to_qwen = model_utils.get_qwen2(model_src, None)
    model.eval()
    model.seqlen = seqlen
    tokenizer = AutoTokenizer.from_pretrained(model_src)

    args = qwen2_packed.qwen25_w4a4kv4_args(
        exp_dir=output_dir, nsamples=nsamples, epochs=epochs,
        cali_bsz=cali_bsz, model_name=model_src,
    )
    logger.info("calibration: %d samples x %d tokens, %d epochs, %d optimizer steps per layer",
                nsamples, seqlen, epochs, steps_per_layer)
    trainloader = data_utils.get_loaders(
        args, args.cali_dataset, tokenizer, nsamples=nsamples, seqlen=seqlen, eval_mode=False,
    )
    model = apply_flatquant_to_qwen(args, model)
    train_utils.cali_flat_quant(args, model, trainloader, dev, logger=logger)
    flat_utils.save_flat_matrices(args, model)
    print(f"calibrated {model.config.num_hidden_layers} layers in {time.time() - started:.1f}s", flush=True)

    # Fold and RTN in fp32 so the int4 codes and scales come from unrounded weights.
    model.float()
    flat_utils.reparameterize_model(model)
    quantizers = gptq_utils.rtn_fwrd(model, dev, args, name_filter=qwen2_packed.is_packed_projection)

    quant_config = qwen2_packed.quant_config_for(args, model, extra={
        "base_model": MODEL_ID,
        "repo_url": "https://github.com/ruikangliu/FlatQuant",
        "repo_commit": REPO_COMMIT,
        "smoke": smoke,
    })
    quant_config["calibration"]["seqlen"] = seqlen
    quant_config["calibration"]["optimizer_steps_per_layer"] = steps_per_layer
    summary = qwen2_packed.export_qwen2_packed(args, model, quantizers, tokenizer, output_dir, quant_config)
    print("saved " + json.dumps(summary), flush=True)

    # Packed reload vs the in-memory fake-quant model on one calibration window.
    probe = trainloader[0][0][:, : min(CHECK_TOKENS, seqlen)]
    model.to(dev)
    ref = qwen2_packed.logits_of(model, probe)
    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    from deploy.transformers.modeling_qwen2 import FlatQuantQwen2ForCausalLM

    packed = FlatQuantQwen2ForCausalLM.from_pretrained(output_dir, device="cuda")
    check = qwen2_packed.compare_logits(ref, qwen2_packed.logits_of(packed, probe))
    check["tokens"] = int(probe.shape[-1])
    check["reference"] = "in-memory fake-quant FlatQuant model (fp32, eager)"
    check["runtime"] = packed.flatquant_runtime
    qwen2_packed.update_quant_config(output_dir, {"packed_self_check": check})
    print("packed_vs_fakequant " + json.dumps(check), flush=True)
    if not check["finite"]:
        raise SystemExit("packed int4 model produced non-finite logits")
    print(f"done in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()

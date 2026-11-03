#!/usr/bin/env python
# QUANT_AGENT_OVERLAY_DIR=<overlay>
"""OmniQuant W4A4 for microsoft/Phi-3.5-mini-instruct (plan diagnose_fix, from phi3_layer_down_let).

Fix for job 20261102T224718Z-e79ead (non-finite loss, layer 5 epoch 4 step 16): the dynamic
activation / Q/K/V quantizers evaluate in fp32 inside the fp16-AMP block (their fp16 backward
overflowed whatever the loss scale and drove the GradScaler to 2.9e-39, then NaN parameters), and a
non-finite step or epoch is skipped / rolled back to the best epoch instead of aborting the job
(differences.json D15). The reload adapter uses the same fp32 quantizer arithmetic.

The repo's own pipeline (quantize/omniquant.py block-wise LWC + LET training) with a Phi-3
decoder layer added by the overlay (models/int_phi3_layer.py, dispatched on model_type phi3):

  WikiText-2 train, 128 random 2048-token windows, seed 2 (datautils.get_wikitext2, read from the
  local parquet) -> act scales (generate_act_scale_shift.get_act_scales, max |x| per input
  channel of every linear) -> omniquant(): per layer, SmoothQuant-initialised LET scales
  ([ln1, qkv_proj], [v rows, o_proj], [Q, K] tied over RoPE pairs, [ln2, gate_up_proj],
  [up rows of gate_up_proj, down_proj] (exact for SwiGLU; beyond the paper, D8))
  and LWC bounds trained with AdamW (let_lr 1e-3, lwc_lr 1e-2, wd 0, batch 1, AMP) on the MSE to
  the fp layer output, keeping the parameters of the lowest-loss epoch; W4 asymmetric per output channel, A4 asymmetric per token on every linear
  input and on Q/K/V, softmax output 16-bit -> LETs folded, weights fake-quantized in place ->
  native Phi3ForCausalLM fp16 checkpoint (auto_map stripped) + quantization_config.json.

The artifact reloads through quant_agent_inference_adapter.py, which re-attaches the A4
activation quantizers (W4 weights alone would be W4A16).
"""
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
OUTPUT_DIR = "./quantized/omniquant-phi35-mini-vastrtx3090"
REPO_COMMIT = "feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14"

W_BITS = 4
A_BITS = 4
GROUP_SIZE = None  # per output channel, as the paper's W4A4 rows
ALPHA = 0.75  # scripts/Llama-2/*/w4a4.sh
LET_LR = 1e-3  # scripts/Llama-2/*/w4a4.sh (Llama-family W4A4)
LWC_LR = 1e-2  # main.py default
WEIGHT_DECAY = 0.0
SEED = 2  # main.py / datautils default
BATCH_SIZE = 1
CALIB_SAMPLES = 128
CALIB_SEQ_LEN = 2048
EPOCHS = 20  # full run: 20 x 128 = 2560 optimizer steps per layer (see differences.json)
# Smoke: a learned method needs enough optimizer steps per layer for the LET/LWC parameters to
# move; 320 steps = 20 epochs over the 16 smoke windows (the paper's epoch count).
SMOKE_STEPS_PER_LAYER = 320
DOWN_LET = True
# Keep each layer's LET/LWC parameters from the epoch with the lowest mean block loss
# (differences.json D9: on Phi-3.5 layer 0 the block loss rises after ~10 epochs at the repo lr).
KEEP_BEST_EPOCH = True
SNAPSHOT_DIR_REL = ".cache/hf-snapshots"
WIKITEXT_GLOBS = (
    ".cache/calib/wikitext2/train.parquet",
    ".cache/eval/hf/datasets--Salesforce--wikitext/snapshots/*/wikitext-2-raw-v1/train-*.parquet",
)
CHECK_TOKENS = 256


def _env_int(name, default):
    raw = os.environ.get(name)
    return int(raw) if raw else default


def _roots():
    roots = [Path.cwd()]
    overlay = os.environ.get("QUANT_AGENT_OVERLAY_DIR")
    if overlay:
        parents = Path(overlay).resolve().parents  # out/overlays/<slug>/<plan>/<hash> -> workspace
        if len(parents) > 4:
            roots.append(parents[4])
    return roots


def _wikitext_train():
    looked = []
    for root in _roots():
        for pattern in WIKITEXT_GLOBS:
            looked.append(str(root / pattern))
            hits = sorted(root.glob(pattern))
            if hits:
                return hits[0].resolve()
    raise FileNotFoundError("WikiText-2 train parquet not found; looked at: " + ", ".join(looked))


def _model_source():
    smoke_dir = os.environ.get("QUANT_AGENT_SMOKE_MODEL_DIR")
    if smoke_dir:
        return smoke_dir
    for root in _roots():
        snapshot = root / SNAPSHOT_DIR_REL / MODEL_ID.replace("/", "_")
        if (snapshot / "config.json").is_file():
            return str(snapshot.resolve())
    return MODEL_ID


def _quant_args(nsamples, epochs, output_dir):
    args = SimpleNamespace(
        net="phi3", model=MODEL_ID, nsamples=nsamples, batch_size=BATCH_SIZE, epochs=epochs,
        wbits=W_BITS, abits=A_BITS, group_size=GROUP_SIZE, alpha=ALPHA, let_lr=LET_LR, lwc_lr=LWC_LR,
        wd=WEIGHT_DECAY, let=True, lwc=True, aug_loss=False, symmetric=False, disable_zero_point=False,
        deactive_amp=False, resume=None, real_quant=False, output_dir=output_dir, down_let=DOWN_LET,
        keep_best_epoch=KEEP_BEST_EPOCH,
    )
    args.weight_quant_params = {
        "n_bits": W_BITS, "per_channel_axes": [0], "symmetric": False, "dynamic_method": "per_channel",
        "group_size": GROUP_SIZE, "lwc": True, "disable_zero_point": False,
    }
    act = {"n_bits": A_BITS, "per_channel_axes": [], "symmetric": False, "dynamic_method": "per_token"}
    args.act_quant_params = dict(act)
    args.q_quant_params = dict(act)
    args.k_quant_params = dict(act)
    args.v_quant_params = dict(act)
    args.p_quant_params = {"n_bits": 16, "metric": "fix0to1"}
    return args


def _logits(model, input_ids):
    with torch.no_grad():
        return model(input_ids.to(model.get_input_embeddings().weight.device)).logits.float().cpu()


def main():
    method_repo = os.environ.get("QUANT_AGENT_METHOD_REPO")
    if not method_repo:
        raise SystemExit("QUANT_AGENT_METHOD_REPO is not set; run through the launcher")
    print(f"QUANT_AGENT_OVERLAY_DIR={os.environ.get('QUANT_AGENT_OVERLAY_DIR')}", flush=True)
    sys.path.insert(0, method_repo)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required (OmniQuant trains each block on GPU)")

    model_src = _model_source()
    output_dir = os.environ.get("QUANT_AGENT_SMOKE_OUTPUT_DIR") or OUTPUT_DIR
    smoke = os.environ.get("QUANT_AGENT_SMOKE") == "1"
    nsamples = _env_int("QUANT_AGENT_SMOKE_CALIB_SAMPLES", CALIB_SAMPLES)
    seqlen = _env_int("QUANT_AGENT_SMOKE_SEQ_LEN", CALIB_SEQ_LEN)
    steps_per_epoch = max(1, nsamples // BATCH_SIZE)
    epochs = max(1, math.ceil(SMOKE_STEPS_PER_LAYER / steps_per_epoch)) if smoke else EPOCHS
    calib_path = _wikitext_train()
    os.makedirs(output_dir, exist_ok=True)
    print(
        f"model={model_src} output={output_dir} samples={nsamples} seqlen={seqlen} epochs={epochs} "
        f"optimizer_steps_per_layer={epochs * steps_per_epoch} down_let={DOWN_LET} smoke={smoke} calib={calib_path}",
        flush=True,
    )

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
    logger = logging.getLogger("omniquant")

    from transformers import AutoTokenizer
    from transformers.models.phi3.modeling_phi3 import Phi3ForCausalLM

    from generate_act_scale_shift import get_act_scales
    from models.int_phi3_layer import QuantPhi3DecoderLayer
    from quantize.omniquant import omniquant
    from quantize.phi3_export import export_phi3_fakequant, get_wikitext2_local

    started = time.time()
    # Native class, never the checkpoint's remote code (auto_map); fp16 like LMClass.
    tokenizer = AutoTokenizer.from_pretrained(model_src, trust_remote_code=False)
    model = Phi3ForCausalLM.from_pretrained(model_src, torch_dtype=torch.float16, attn_implementation="eager")
    model.eval()
    model.config.use_cache = False
    for param in model.parameters():
        param.requires_grad = False
    bad = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
    if bad:
        raise SystemExit(f"non-finite weights after the fp16 cast: {bad[:5]}")
    num_layers = model.config.num_hidden_layers

    dataloader = get_wikitext2_local(str(calib_path), tokenizer, nsamples, SEED, seqlen)
    print(f"calibration: {len(dataloader)} windows x {seqlen} tokens ({time.time() - started:.0f}s)", flush=True)

    model.to("cuda")
    with torch.no_grad():
        act_scales = get_act_scales(model, dataloader, nsamples)
    model.to("cpu")
    torch.cuda.empty_cache()
    for i in range(num_layers):
        for leaf in ("self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj"):
            key = f"model.layers.{i}.{leaf}"
            if key not in act_scales or not torch.isfinite(act_scales[key]).all():
                raise SystemExit(f"missing or non-finite activation scales for {key}")
    print(f"act scales for {len(act_scales)} linears ({time.time() - started:.0f}s)", flush=True)

    args = _quant_args(nsamples, epochs, output_dir)
    lm = SimpleNamespace(model=model, device=torch.device("cuda"), _device=torch.device("cuda"), seqlen=seqlen)
    omniquant(lm, args, dataloader, act_scales, None, logger)
    n_quant = sum(isinstance(m, QuantPhi3DecoderLayer) for m in model.model.layers)
    if n_quant != num_layers:
        raise SystemExit(f"omniquant converted {n_quant} of {num_layers} layers")
    print(f"omniquant done ({time.time() - started:.0f}s)", flush=True)

    # Reference logits of the in-memory quantized model (QuantLinear/QuantMatMul runtime).
    probe = dataloader[0][0][:, : min(CHECK_TOKENS, seqlen)]
    model.to("cuda")
    for layer in model.model.layers:
        layer.set_quant_state(weight_quant=False, act_quant=True)  # weights are already fake-quantized
    ref_logits = _logits(model, probe)
    model.to("cpu")
    torch.cuda.empty_cache()
    if not torch.isfinite(ref_logits).all():
        raise SystemExit("in-memory quantized model produced non-finite logits")

    quant_config = {
        "quant_method": "omniquant",
        "method_name": "OmniQuant",
        "format": "fake_quant_fp16",
        "w_bits": W_BITS,
        "a_bits": A_BITS,
        "qkv_bits": A_BITS,
        "softmax_bits": 16,
        "group_size": GROUP_SIZE,
        "weight_quant": "per_channel",
        "w_sym": False,
        "act_quant": "per_token",
        "act_dynamic": True,
        "a_sym": False,
        "lwc": True,
        "let": True,
        "let_pairs": ["input_layernorm->qkv_proj", "v rows of qkv_proj->o_proj",
                      "q rows/k rows of qkv_proj (tied over RoPE pairs)", "post_attention_layernorm->gate_up_proj"]
        + (["up rows of gate_up_proj->down_proj"] if DOWN_LET else []),
        "let_shift": False,
        "weights_stored": "dequantized fp16 (4-bit levels), LETs folded",
        "runtime_transforms": ["per-token A4 fake quant on qkv_proj/o_proj/gate_up_proj/down_proj inputs",
                               "per-token-per-head A4 fake quant on Q/K (after RoPE) and V"],
        "inference_adapter": "quant_agent_inference_adapter.py",
        "calibration": {
            "dataset": "wikitext-2-raw-v1 train (local parquet)",
            "seed": SEED, "num_samples": nsamples, "seq_len": seqlen, "epochs": epochs,
            "optimizer_steps_per_layer": epochs * steps_per_epoch, "batch_size": BATCH_SIZE,
            "let_lr": LET_LR, "lwc_lr": LWC_LR, "alpha": ALPHA, "keep_best_epoch": KEEP_BEST_EPOCH,
            "act_quantizer_compute": "float32 (FP32ActQuantizer), cast back to fp16",
            "smoke": smoke,
        },
        "base_model": MODEL_ID,
        "repo_url": "https://github.com/OpenGVLab/OmniQuant",
        "repo_commit": REPO_COMMIT,
        "publish_exclude": ["omni_parameters.pth"],
    }
    summary = export_phi3_fakequant(model, tokenizer, output_dir, quant_config)
    print("saved " + json.dumps(summary), flush=True)
    del model
    torch.cuda.empty_cache()

    # Reload through the inference adapter and compare with the in-memory quantized model.
    import importlib.util

    spec = importlib.util.spec_from_file_location("omniquant_adapter", os.path.join(method_repo, "quant_agent_inference_adapter.py"))
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    reloaded, _ = adapter.load_model_and_tokenizer(
        model_path=output_dir, model_id=MODEL_ID, dtype="float16", device="cuda", trust_remote_code=False
    )
    got = _logits(reloaded, probe)
    rel = ((got - ref_logits).norm() / ref_logits.norm().clamp(min=1e-12)).item()
    top1 = (got.argmax(-1) == ref_logits.argmax(-1)).float().mean().item()
    check = {"tokens": int(probe.shape[-1]), "rel_logit_error_vs_inmemory": rel, "top1_agreement": top1,
             "finite": bool(torch.isfinite(got).all()), "runtime": reloaded.omniquant_runtime}
    with open(os.path.join(output_dir, "quantization_config.json")) as handle:
        cfg = json.load(handle)
    cfg["reload_self_check"] = check
    with open(os.path.join(output_dir, "quantization_config.json"), "w") as handle:
        json.dump(cfg, handle, indent=2)
    print("reload_vs_inmemory " + json.dumps(check), flush=True)
    if not check["finite"]:
        raise SystemExit("reloaded artifact produced non-finite logits")
    if rel > 0.05:
        raise SystemExit(f"adapter reload disagrees with the in-memory quantized model: rel {rel:.3f}")
    print(f"done in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()

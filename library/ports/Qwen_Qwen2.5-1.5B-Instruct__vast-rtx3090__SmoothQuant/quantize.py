#!/usr/bin/env python
# QUANT_AGENT_OVERLAY_DIR=<overlay>
"""SmoothQuant W8A8 for Qwen/Qwen2.5-1.5B-Instruct (strategy kernel_triton).

Quantization is identical to the qwen2_llama_path winner (c47f3e78ceb1fa88): the
overlay's repo-side hunks are byte-identical, so the artifact is the same. Only the
inference adapter changes: Triton per-token int8 activation quant + Triton
int8 x int8 -> int32 GEMM with a fused dequant/bias epilogue, SDPA attention.

Uses the repo's own Llama-like pipeline, extended to Qwen2 by the overlay:
get_act_scales (Pile val, seed 42) -> smooth_lm (alpha 0.85; input_layernorm ->
q/k/v, post_attention_layernorm -> gate/up) -> quantize_model (all 7 decoder
linears to W8A8Linear, per-channel int8 weights, per-token dynamic int8
activations, keep_int8=True) -> save_w8a8_int8 (int8 weight + fp16 per-channel
weight_scale, fp16 biases/norms/tied embeddings, quantization_config.json).
The artifact reloads through quant_agent_inference_adapter.py (int8 Triton path).
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "./quantized/smoothquant-qwen25-15b-vastrtx3090"
REPO_COMMIT = "c61476d728e42ae0d8a35e7e78494edcac3237b5"

ALPHA = 0.85
CALIB_SAMPLES = 512
CALIB_SEQ_LEN = 512
CALIB_REL_PATH = ".cache/calib/val.jsonl.zst"
CALIB_DONE_NAME = "pile.done"

QUANT_CONFIG = {
    "quant_method": "smoothquant",
    "method_name": "SmoothQuant",
    "format": "w8a8_int8",
    "w_bits": 8,
    "a_bits": 8,
    "weight_quant": "per_channel",
    "act_quant": "per_token",
    "act_dynamic": True,
    "w_sym": True,
    "a_sym": True,
    "quantize_bmm_input": False,
    "weight_dtype": "int8",
    "weight_scale_dtype": "float16",
    "weight_scale_shape": "[out_features, 1]",
    "dequant": "weight_fp16 = weight.int8 * weight_scale",
    "kept_fp16": [
        "linear biases",
        "input_layernorm / post_attention_layernorm (smoothed)",
        "model.norm",
        "model.embed_tokens (lm_head tied, not stored)",
    ],
    "smoothing_folded_offline": True,
    "smoothing_sites": [
        "input_layernorm -> self_attn.q_proj, self_attn.k_proj, self_attn.v_proj",
        "post_attention_layernorm -> mlp.gate_proj, mlp.up_proj",
    ],
    "runtime_transforms": [],
    "repo_url": "https://github.com/mit-han-lab/smoothquant",
    "repo_commit": REPO_COMMIT,
    "inference_adapter": "quant_agent_inference_adapter.py",
}


def _env_int(name, default):
    raw = os.environ.get(name)
    return int(raw) if raw else default


def _calib_path():
    candidates = [Path.cwd() / CALIB_REL_PATH]
    overlay = os.environ.get("QUANT_AGENT_OVERLAY_DIR")
    if overlay:
        # out/overlays/<slug>/<plan>/<hash> -> workspace root
        candidates.append(Path(overlay).resolve().parents[4] / CALIB_REL_PATH)
    for path in candidates:
        if path.is_file() and (path.parent / CALIB_DONE_NAME).is_file():
            return path.resolve()
    raise FileNotFoundError(
        f"Pile validation file not found (or its {CALIB_DONE_NAME} marker is missing) at: "
        + ", ".join(str(p) for p in candidates)
    )


def main():
    method_repo = os.environ.get("QUANT_AGENT_METHOD_REPO")
    if not method_repo:
        raise SystemExit("QUANT_AGENT_METHOD_REPO is not set; run through the launcher")
    print(f"QUANT_AGENT_OVERLAY_DIR={os.environ.get('QUANT_AGENT_OVERLAY_DIR')}", flush=True)
    sys.path.insert(0, method_repo)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

    from smoothquant.calibration import get_act_scales
    from smoothquant.fake_quant import W8A8Linear, quantize_model
    from smoothquant.int8_export import save_w8a8_int8
    from smoothquant.smooth import smooth_lm

    model_src = os.environ.get("QUANT_AGENT_SMOKE_MODEL_DIR") or MODEL_ID
    output_dir = os.environ.get("QUANT_AGENT_SMOKE_OUTPUT_DIR") or OUTPUT_DIR
    num_samples = _env_int("QUANT_AGENT_SMOKE_CALIB_SAMPLES", CALIB_SAMPLES)
    seq_len = _env_int("QUANT_AGENT_SMOKE_SEQ_LEN", CALIB_SEQ_LEN)
    if os.environ.get("QUANT_AGENT_SMOKE") == "1":
        # SmoothQuant has no optimizer loop; smoke only shrinks calibration (above).
        print("smoke mode: no iterations to cut", flush=True)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required (the model is calibrated in fp16 on GPU)")

    calib_path = _calib_path()
    print(
        f"model={model_src} output={output_dir} alpha={ALPHA} samples={num_samples} "
        f"seq_len={seq_len} calib={calib_path}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_src)
    model = AutoModelForCausalLM.from_pretrained(
        model_src, torch_dtype=torch.float16, attn_implementation="sdpa"
    ).to("cuda")
    model.eval()

    layers = [m for m in model.modules() if isinstance(m, Qwen2DecoderLayer)]
    if not layers or len(layers) != model.config.num_hidden_layers:
        raise SystemExit(f"expected {model.config.num_hidden_layers} Qwen2DecoderLayer, found {len(layers)}")

    started = time.time()
    with torch.no_grad():
        act_scales = get_act_scales(model, tokenizer, str(calib_path), num_samples, seq_len)
    print(f"collected act scales for {len(act_scales)} linears in {time.time() - started:.1f}s", flush=True)

    norms_before = [
        (layer.input_layernorm.weight.detach().clone(), layer.post_attention_layernorm.weight.detach().clone())
        for layer in layers
    ]
    smooth_lm(model, act_scales, ALPHA)
    unsmoothed = [
        i for i, (layer, (attn_w, ffn_w)) in enumerate(zip(layers, norms_before))
        if torch.equal(attn_w, layer.input_layernorm.weight) or torch.equal(ffn_w, layer.post_attention_layernorm.weight)
    ]
    if unsmoothed:
        raise SystemExit(f"smooth_lm left layers {unsmoothed} unsmoothed (Qwen2 dispatch missing?)")
    del norms_before

    model = quantize_model(
        model,
        weight_quant="per_channel",
        act_quant="per_token",
        quantize_bmm_input=False,
        keep_int8=True,
    )
    n_w8a8 = sum(isinstance(m, W8A8Linear) for m in model.modules())
    if n_w8a8 != 7 * len(layers):
        raise SystemExit(f"expected {7 * len(layers)} W8A8Linear, got {n_w8a8}")
    print(f"smoothed {len(layers)} layers, replaced {n_w8a8} linears with W8A8Linear", flush=True)

    with torch.no_grad():
        probe = tokenizer("The capital of France is", return_tensors="pt").input_ids.to("cuda")
        logits = model(probe).logits
    if not torch.isfinite(logits).all():
        raise SystemExit("fake-quant model produced non-finite logits")

    quant_config = dict(QUANT_CONFIG)
    quant_config.update(
        {
            "alpha": ALPHA,
            "calibration": {
                "dataset": "mit-han-lab/pile-val-backup val.jsonl.zst (local copy)",
                "shuffle_seed": 42,
                "num_samples": num_samples,
                "seq_len": seq_len,
            },
            "base_model": MODEL_ID,
        }
    )
    summary = save_w8a8_int8(model, tokenizer, output_dir, quant_config)
    print("saved " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

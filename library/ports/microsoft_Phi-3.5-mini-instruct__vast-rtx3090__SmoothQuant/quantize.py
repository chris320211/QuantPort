#!/usr/bin/env python
# QUANT_AGENT_OVERLAY_DIR=<overlay>
"""SmoothQuant W8A8 for microsoft/Phi-3.5-mini-instruct (plan phi3_fused_path).

The repo's own pipeline, extended to Phi-3 by the overlay (D1-D3):
get_act_scales (Pile val, seed 42) -> smooth_lm (alpha 0.75; input_layernorm ->
self_attn.qkv_proj, post_attention_layernorm -> mlp.gate_up_proj, each fused
linear taking the whole scale) -> quantize_model (qkv_proj, o_proj, gate_up_proj,
down_proj to W8A8Linear: per-channel int8 weights, per-token dynamic int8
activations, keep_int8=True) -> save_w8a8_int8 (int8 weight + fp16 per-channel
weight_scale, fp16 norms / embeddings / untied lm_head, quantization_config.json,
auto_map stripped) (D4, D5). The model is loaded natively in fp16 (D5, D6).
The artifact reloads through quant_agent_inference_adapter.py, which runs the
shared w8a8_int8 Triton kernels.
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
OUTPUT_DIR = "./quantized/smoothquant-phi35-mini-vastrtx3090"
REPO_COMMIT = "c61476d728e42ae0d8a35e7e78494edcac3237b5"

# D7: per-linear W8A8 error on Phi-3.5 is flat-minimal over 0.65-0.75 (0.85 is worse).
ALPHA = 0.75
CALIB_SAMPLES = 512
CALIB_SEQ_LEN = 512
CALIB_DIR_REL = ".cache/calib"
CALIB_DONE_NAME = "pile.done"
SNAPSHOT_DIR_REL = ".cache/hf-snapshots"
LINEARS_PER_LAYER = 4  # qkv_proj, o_proj, gate_up_proj, down_proj

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
        "input_layernorm / post_attention_layernorm (smoothed)",
        "model.norm",
        "model.embed_tokens",
        "lm_head (untied, stored)",
    ],
    "smoothing_folded_offline": True,
    "smoothing_sites": [
        "input_layernorm -> self_attn.qkv_proj",
        "post_attention_layernorm -> mlp.gate_up_proj",
    ],
    "runtime_transforms": [],
    "repo_url": "https://github.com/mit-han-lab/smoothquant",
    "repo_commit": REPO_COMMIT,
    "inference_adapter": "quant_agent_inference_adapter.py",
}


def _env_int(name, default):
    raw = os.environ.get(name)
    return int(raw) if raw else default


def _roots():
    roots = [Path.cwd()]
    overlay = os.environ.get("QUANT_AGENT_OVERLAY_DIR")
    if overlay:
        # out/overlays/<slug>/<plan>/<hash> -> workspace root
        parents = Path(overlay).resolve().parents
        if len(parents) > 4:
            roots.append(parents[4])
    return roots


def _calib_path():
    """Local Pile validation copy: val.jsonl, or val.jsonl.zst when zstandard is installed."""
    try:
        import zstandard  # noqa: F401  (datasets needs it to read .zst)
        names = ("val.jsonl", "val.jsonl.zst")
    except ImportError:
        names = ("val.jsonl",)
    looked = []
    for root in _roots():
        folder = root / CALIB_DIR_REL
        for name in names:
            path = folder / name
            looked.append(str(path))
            if path.is_file() and (folder / CALIB_DONE_NAME).is_file():
                return path.resolve()
    raise FileNotFoundError(
        "Pile validation file not found (or its pile.done marker is missing). Fetch "
        "https://huggingface.co/datasets/mit-han-lab/pile-val-backup/resolve/main/val.jsonl.zst "
        f"into {CALIB_DIR_REL}/, decompress it to val.jsonl when the venv lacks zstandard, "
        f"then touch {CALIB_DIR_REL}/{CALIB_DONE_NAME}. Looked at: " + ", ".join(looked)
    )


def _model_source():
    """The smoke slice, else the gather snapshot of MODEL_ID, else the Hub id."""
    smoke_dir = os.environ.get("QUANT_AGENT_SMOKE_MODEL_DIR")
    if smoke_dir:
        return smoke_dir
    for root in _roots():
        snapshot = root / SNAPSHOT_DIR_REL / MODEL_ID.replace("/", "_")
        if (snapshot / "config.json").is_file():
            return str(snapshot.resolve())
    return MODEL_ID


def main():
    method_repo = os.environ.get("QUANT_AGENT_METHOD_REPO")
    if not method_repo:
        raise SystemExit("QUANT_AGENT_METHOD_REPO is not set; run through the launcher")
    print(f"QUANT_AGENT_OVERLAY_DIR={os.environ.get('QUANT_AGENT_OVERLAY_DIR')}", flush=True)
    sys.path.insert(0, method_repo)

    from transformers import AutoTokenizer
    from transformers.models.phi3.modeling_phi3 import Phi3DecoderLayer, Phi3ForCausalLM

    from smoothquant.calibration import get_act_scales
    from smoothquant.fake_quant import W8A8Linear, quantize_model
    from smoothquant.int8_export import save_w8a8_int8
    from smoothquant.smooth import smooth_lm

    model_src = _model_source()
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

    # D5: native class, never the checkpoint's remote code. D6: fp16 like the
    # repo's calibration (bf16 checkpoint; activations stay far inside fp16 range).
    tokenizer = AutoTokenizer.from_pretrained(model_src, trust_remote_code=False)
    model = Phi3ForCausalLM.from_pretrained(
        model_src, torch_dtype=torch.float16, attn_implementation="sdpa"
    ).to("cuda")
    model.eval()
    bad = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
    if bad:
        raise SystemExit(f"non-finite weights after the fp16 cast: {bad[:5]}")

    layers = [m for m in model.modules() if isinstance(m, Phi3DecoderLayer)]
    if not layers or len(layers) != model.config.num_hidden_layers:
        raise SystemExit(f"expected {model.config.num_hidden_layers} Phi3DecoderLayer, found {len(layers)}")

    started = time.time()
    with torch.no_grad():
        act_scales = get_act_scales(model, tokenizer, str(calib_path), num_samples, seq_len)
    print(f"collected act scales for {len(act_scales)} linears in {time.time() - started:.1f}s", flush=True)
    for i in range(len(layers)):
        for leaf in ("self_attn.qkv_proj", "mlp.gate_up_proj"):
            key = f"model.layers.{i}.{leaf}"
            if key not in act_scales:
                raise SystemExit(f"no activation scales for {key}")
            if not torch.isfinite(act_scales[key]).all():
                raise SystemExit(f"non-finite activation scales for {key} (fp16 overflow?)")

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
        raise SystemExit(f"smooth_lm left layers {unsmoothed} unsmoothed (Phi-3 dispatch missing?)")
    del norms_before

    model = quantize_model(
        model,
        weight_quant="per_channel",
        act_quant="per_token",
        quantize_bmm_input=False,
        keep_int8=True,
    )
    n_w8a8 = sum(isinstance(m, W8A8Linear) for m in model.modules())
    if n_w8a8 != LINEARS_PER_LAYER * len(layers):
        raise SystemExit(f"expected {LINEARS_PER_LAYER * len(layers)} W8A8Linear, got {n_w8a8}")
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
                "dataset": "mit-han-lab/pile-val-backup val.jsonl (local copy)",
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

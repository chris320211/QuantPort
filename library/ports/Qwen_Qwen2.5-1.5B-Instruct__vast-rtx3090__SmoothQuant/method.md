# SmoothQuant (arXiv 2211.10438), method model

Repo: mit-han-lab/smoothquant @ c61476d. W8A8: int8 weights, int8 activations, offline smoothing that moves activation outliers into the weights.

## Steps

1. **Collect activation statistics** (`smoothquant/calibration.py:13-51`, `get_act_scales`). Forward hooks on every `nn.Linear` record the per-input-channel max `|X_j|` over calibration tokens. Inputs: fp16 model, calibration text. Output: dict `linear name -> [in_features]` act-scale vector. Paper §5.1 ("Activation smoothing").
2. **Compute smoothing factors** (`smoothquant/smooth.py:49-67`, `smooth_ln_fcs_llama_like`). For a norm and the linears that read its output, `s_j = max|X_j|^α / max|W_j|^(1-α)`, where `max|W_j|` is taken over the output rows of all reader linears together. Clamped at 1e-5. Paper §4, Eq. 4.
3. **Fold the smoothing offline** (`smooth.py:69-71`). Divide the preceding norm's weight by `s` (and its bias, for LayerNorm, `smooth.py:41-42`), and multiply each reader linear's weight columns by `s`. Applied to two sites per decoder layer (`smooth.py:126-141`, Llama path): input norm → q/k/v projections, and post-attention norm → MLP gate/up projections. o_proj and down_proj are not smoothed, because their inputs have no norm to fold into.
4. **Quantize weights** (`smoothquant/fake_quant.py:7-13`, `quantize_weight_per_channel_absmax`). Symmetric int8, per output channel: `scale = max|W_row| / 127`, `W_q = round(W / scale)`.
5. **Quantize activations at runtime** (`fake_quant.py:27-34`, `quantize_activation_per_token_absmax`). Symmetric int8, dynamic per token: `scale = max|x_token| / 127`. This is the paper's O1-O3 efficiency level with per-token dynamic activations (§3, Table 2).
6. **Replace linears** (`fake_quant.py:183-232`, `quantize_llama_like`). Every decoder linear (q, k, v, o, gate, up, down) becomes `W8A8Linear` (`fake_quant.py:48-135`). The bias is kept in fp16 and added after the matmul. `quantize_bmm_input=False` by default, so attention BMM inputs stay fp16.

## Tensor roles

- **Residual stream / norm weights (RMSNorm)**: the input norm and the post-attention norm. Step 3 divides them by `s`. Kept in fp16, not quantized.
- **Attention input projections (q, k, v weights)**: readers of the input norm. Step 3 scales their input columns by `s`; step 4 quantizes them to int8 per channel.
- **Attention output projection (o_proj weight)**: not smoothed; step 4 int8 per channel.
- **MLP input projections (gate, up weights)**: readers of the post-attention norm. Step 3 scales their input columns by `s`; step 4 int8 per channel.
- **MLP output projection (down_proj weight)**: not smoothed; step 4 int8 per channel.
- **Linear biases**: the method keeps them in fp16 and adds them after the int8 matmul (`fake_quant.py:133-134`). Smoothing never touches them.
- **Activations entering every quantized linear**: int8 per token, dynamic (step 5).
- **Embeddings and LM head**: untouched, kept in fp16 (`quantize_llama_like` walks `model.model` decoder modules only).
- **KV cache and attention BMM inputs**: fp16 (`quantize_bmm_input=False`).

## Invariants

- **Smoothing preserves the function.** `(X·diag(s)^-1)·(diag(s)·W^T) = X·W^T`, so before quantization the smoothed model computes the same function. This requires that `s` is folded into the operator that **produces** X (the norm weight, which is elementwise) and into **every** linear that reads that same X.
- **Each norm output feeds exactly the linears smoothed with it.** If any reader of the norm output is not scaled by `s`, its output changes.
- **Linear biases are unaffected.** Smoothing rescales the input side of W (its columns); the bias is added to `X·W^T` after the matmul, so it stays exactly as it was. Biases need no change.
- **Embeddings/LM head are untouched**, so tied embeddings stay tied and consistent.
- **The norm must be a pure per-channel scale** (RMSNorm weight or LayerNorm weight+bias) so dividing by `s` is exact.

## Calibration

- **Data**: the Pile validation set (`mit-han-lab/pile-val-backup`, `val.jsonl.zst`), shuffled with seed 42 (`calibration.py:39-40`, `examples/generate_act_scales.py:34-53`). Paper §5.1 calibrates on a Pile validation subset.
- **Size**: 512 samples, sequence length 512, truncated (`generate_act_scales.py:37-38`).
- **What is measured**: per-input-channel max |activation| per linear. Nothing is optimized; no gradients or iterations.
- **Migration strength α**: 0.5 for OPT/BLOOM (paper §5.1); the repo uses 0.85 for Llama-2/3 and 0.8 for Mistral (`README.md` PPL table). Grid search is the paper's way to pick it.

## Save and reload

- **What the repo writes**: the reference pipeline (`ppl_eval.py`) keeps a fake-quant fp16 model in memory and does not save it. The real int8 path (`smoothquant/opt.py`, `examples/export_int8_model.py`) exists only for OPT and needs `torch-int` CUTLASS kernels.
- **Classes**: `W8A8Linear` stores the dequantized fp16 weight, so a plain save of the fake-quant model is dense fp16 (no VRAM win).
- **What a port must save to be useful**: int8 `weight` per linear plus a float per-output-channel `weight_scale`, fp16 biases, and the smoothed fp16 norms, with per-token dynamic activation quantization at runtime. That is the layout the pipeline's vLLM export maps to compressed-tensors `int-quantized` (`export_vllm.py` profile `w8a8_int8`).
- **Reload**: needs an inference adapter that rebuilds W8A8 linears from int8 weight + scale and quantizes activations per token at runtime. No eval-only flags; no transforms at runtime because smoothing is folded into the weights.

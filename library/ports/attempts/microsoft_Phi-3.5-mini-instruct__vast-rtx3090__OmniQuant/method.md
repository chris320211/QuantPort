# OmniQuant (arXiv 2308.13137), OpenGVLab/OmniQuant @ feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14

Weight-activation (W4A4) setting: learnable weight clipping (LWC) plus learnable equivalent
transformation (LET), trained block by block to match each full-precision transformer block.
Model independent; module names below are roles, and the repo's Llama code is quoted only as the
reference implementation of each step.

Entry point of the repo's own pipeline: `main.py` (`--wbits 4 --abits 4 --lwc --let --epochs 20
--alpha 0.75 --let_lr 1e-3`, `scripts/Llama-2/*/w4a4.sh`) -> `quantize/omniquant.py:omniquant()`.
Generic building blocks: `quantize/quantizer.py` (UniformAffineQuantizer, LWC, STE rounding),
`quantize/int_linear.py` (QuantLinear: weight quantizer + per-token input quantizer),
`quantize/int_matmul.py` (QuantMatMul: quantizers on both operands of an activation matmul),
`quantize/omni_norm.py` (norms whose weight can be swapped for a temporary one),
`models/transformation.py` (fold a LET scale into a norm / linear pair, temporary or in place),
`quantize/utils.py` (parameter groups, truncation, temporary / in-place smooth+quant),
`generate_act_scale_shift.py:get_act_scales` (per-channel max |x| of every linear input),
`utils.NativeScalerWithGradNormCount` (AMP loss scaling). Family glue: the `net`-name switch in
`omniquant()` (`quantize/omniquant.py:57-99`), `models/int_{llama,opt,falcon}_layer.py`, the
`isllama` branches of `quantize/utils.py:62-136`, and `models/LMClass.py`.

## Steps

1. **Calibration data.** 128 random windows of 2048 tokens from the WikiText-2 train split joined
   with "\n\n", python `random` seeded with 2 (paper §4.1 Training; `datautils.py:35-54`,
   `main.py:198-204,327-339`).
2. **Activation statistics for LET initialisation.** Run the full-precision model on the
   calibration windows and record, for every linear, the per-input-channel max |x|
   (`generate_act_scale_shift.py:25-57`) and, for models that use shifting, a running channel
   midpoint (`:59-94`). Paper §4.1: "scaling initialized with SmoothQuant, shifting with Outlier
   Suppression+".
3. **Catch the inputs of the first block** (`quantize/omniquant.py:109-139`): embeddings run on the
   calibration windows; the block input, attention mask and position ids are cached (fp16).
4. **Per block, in order** (paper §3.1 Eq. 1, `quantize/omniquant.py:192-313`):
   1. Wrap every linear in QuantLinear and every activation matmul (QK^T, PV) in QuantMatMul;
      norms in an Omni norm (`models/int_llama_layer.py:20-211`).
   2. Full-precision target: the block output on the cached fp inputs (`:207-215`).
   3. LET parameters (`:216-237`): for each LET pair a per-channel scale initialised SmoothQuant
      style, `s = max|x|^alpha / max|W_col|^(1-alpha)`, clamped at 1e-5, alpha 0.75 for W4A4;
      a QK scale initialised to ones; shifts initialised from the act shifts, or zero (shifting
      disabled) for RMSNorm models and weight-only runs (`:219-221`).
   4. LWC parameters: per weight row (or group) two logits initialised to 4.0; the clipped range is
      `[sigmoid(beta)*min(W), sigmoid(gamma)*max(W)]` (paper §3.2 Eq. 2; `quantize/quantizer.py:67-79,143-145`).
   5. Train (`:243-276`): AdamW, weight decay 0, LET lr 1e-3 (Llama-family W4A4 scripts; repo
      default 5e-3), LWC lr 1e-2, batch 1, 20 epochs over the 128 windows, AMP (fp16 autocast +
      GradScaler) unless 8-bit. Each step re-derives the transformed weights from the frozen fp
      weights (`quantize/utils.py:62-101`): truncate |scale| < 1e-2 (`:44-60`), fold every LET
      into its norm / weights (`models/transformation.py:24-69`), fake-quantize every weight with
      LWC, then run the block with per-token activation quantizers on, and minimise
      MSE(fp block output, quantized block output). Rounding uses the straight-through estimator.
   6. Fold and freeze (`:277-279`; `quantize/utils.py:111-136`): apply every LET to the norms /
      weights in place, replace each weight by its fake-quantized value (fp16), keep the
      activation quantizers.
   7. Propagate: the next block's quantized-path input is this quantized block's output on the
      quantized-path input (`:280-290`); the fp-path input is the fp output from 4.2.
5. **Save** (`main.py:354-369`): delete LET/LWC parameters and `save_pretrained` the model: dense
   fp16 weights holding the 4-bit levels, transformed norms. `--real_quant` packs weights with
   AutoGPTQ but asserts `abits >= 16` (weight-only; `quantize/omniquant.py:294-311`).

## Tensor roles

- **Residual stream**: never transformed (LET only touches norm outputs and linear pairs).
- **Norm weights (and biases)** feeding attention and MLP: divided by the LET scale of their
  reader (and shifted by the LET shift, which becomes a norm bias). Kept fp16.
- **Attention input projections (q, k, v)**: input columns multiplied by the [norm1 -> qkv] LET
  scale; weights W4 per output channel (asymmetric, LWC); input A4 per token.
- **Query/key pair**: Q output channels divided and K output channels multiplied by the QK LET
  scale `s_a` (paper §3.3 Eq. 5); Q and K (after the rotary embedding) A4 per token per head.
- **Value/output pair**: v output rows divided by the [v -> out] LET scale (and shifted), out
  projection input columns multiplied by it; V A4 per token per head.
- **Attention output projection**: W4 (LWC), input A4 per token, LET input side as above.
- **Attention probabilities (softmax output)**: kept 16-bit (paper §4.1, Table A7).
- **MLP input projections (fc1 / gate, up)**: input columns multiplied by the [norm2 -> fc1] LET
  scale; W4 (LWC); input A4 per token.
- **MLP output projection (fc2 / down)**: no LET (paper §3.3, Table A5: features after the
  nonlinearity are sparse, gradients unstable); W4 (LWC); input A4 per token.
- **Linear biases**: transformed with their weights (`fc.bias + W @ shift`, `bias / s` on the v
  side); RMSNorm models without biases get none because shifting is off.
- **KV cache**: K and V are the quantized A4 tensors of the attention matmuls.
- **Embeddings, final norm, LM head**: untouched, fp16.
- **Activations in general**: dynamic, asymmetric MinMax, per token (last dimension), 4 bits; no
  learned activation clipping.

## Invariants

- A LET must be an exact equivalence before quantization: a scale divided out of an activation
  is multiplied into every reader's weights on the same channels, and a shift subtracted from an
  activation is added back through `W @ shift` in each reader's bias. A norm can absorb a scale
  only if it is the sole producer of that activation and all its readers take the same scale.
- A scale on the output rows of one linear (v) must be undone on exactly the input columns that
  read those rows (out projection), respecting any head grouping.
- The QK scale must cancel inside `Q K^T`: it may be applied to the rows of the q and k
  projections only if nothing between the projection and the matmul mixes channels. A rotary
  embedding mixes channel pairs; the scale commutes with it only if both channels of each rotary
  pair carry the same scale value.
- A scale on a linear's input is only equivalent if the input enters linearly: before a
  nonlinearity it cannot be moved (fc2 has no LET in the paper; a scale on a branch that enters
  an elementwise product linearly is exact).
- LWC only changes how a weight row is quantized; the fp weights stay frozen. Per-row quantization
  of a fused weight equals per-row quantization of its parts.
- What is trained is what is saved: the folded, fake-quantized weights plus the same activation
  quantizers must reproduce the trained block.

## Calibration

WikiText-2 train, 128 windows x 2048 tokens, seed 2; act scales (and shifts when used) from the
same windows. Learned per block: LET scales (+ shifts), QK scale, LWC logits. AdamW, wd 0, LET lr
1e-3 for Llama-family W4A4 (scripts/Llama-2/*/w4a4.sh; main.py default 5e-3), LWC lr 1e-2
(main.py:212-215), batch 1, 20 epochs = 2560 optimizer steps per block, loss MSE to the fp block
output, fp16 AMP. alpha 0.75 for the SmoothQuant initialisation. Paper §4.1 lists the two
learning rates the other way round; the repo's scripts are what produced its numbers. Reported
cost: one A100, ~1-2 h for 7B W4A4.

## Save and reload

The repo writes `save_pretrained` fake-quant fp16 weights (4-bit levels per row, transformations
folded) plus transformed norms; the LET/LWC parameters go to `omni_parameters.pth` for resume.
Activation quantization (A4 per token on linear inputs, A4 per token-head on Q/K/V) lives only in
the in-memory QuantLinear/QuantMatMul modules and is not saved, so a plain reload is W4A16.
There is no A4 kernel and no packed W4A4 format in the repo; the only packed path is AutoGPTQ for
weight-only runs (and MLC-LLM for deployment of weight-only models). A faithful reload must
re-attach the activation quantizers with the same arithmetic (dynamic asymmetric MinMax in the
activation dtype) and keep the softmax output 16-bit.

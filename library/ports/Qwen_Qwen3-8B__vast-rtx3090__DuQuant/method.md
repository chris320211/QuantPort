# DuQuant method model (model-independent)

Sources: paper arXiv 2406.01721 (`.cache/papers/2406.01721.txt`); repo Hsu1023/DuQuant @ d56cfc6
(`.venvs/duquant-qwen3-8b-vastrtx3090/repo`, OmniQuant-derived, pins transformers==4.35.2).
Paths below are relative to the repo root. Recipe of record: `run.sh` = `main.py --block_size 128
--max_rotation_step 256 --epochs 0 --wbits 4 --abits 4 --lwc --alpha 0.6 --smooth --lac 0.9 --swc 0.8`
("DuQuant", RTN). `run_lwc.sh` = `--epochs 20 --alpha 0.5`, no `--swc` ("DuQuant+LWC").

DuQuant is a W4A4 (+ 4-bit Q/K/V) fake-quant method. Every quantized linear computes
`Y = (X Λ^-1 R1 P R2)_q · (W Λ R1 P R2)_q^T`, where Λ is a SmoothQuant scale, R1/R2 are
block-diagonal orthogonal rotations (one 128x128 block repeated along the channel dim), and P is a
zigzag channel permutation. The transform is LOCAL to each linear's input: the residual stream is never
rotated, so the activation side of R1 P R2 runs ONLINE at inference.

## Steps

1. **Preprocess: base rotation table.** `get_rot.py:8-41` builds, for n = 2..4096, an orthogonal matrix
   R~ as a product of n-1 Givens rotations (cos=1/sqrt(i+2)), then swaps row/col 0 and n-1. The result spreads
   channel 0 uniformly over all n channels. It is pickled to `./Rot.pkl` (cwd-relative, needs CUDA). At use
   time `utils.py:99-113 get_rot(n)` loads R~[n] and right-multiplies diag(1, Q'), with Q' a random
   (n-1)x(n-1) orthogonal from `torch.linalg.qr(randn)` (seeded by `main.py:315-318`, seed 2).
   Paper §3.2 Eq.(2) and App. G Alg. 1 lines 7-11. `Rot.pkl` is not in the repo checkout. Port: generate it,
   or compute R~ for block_size only.
2. **Preprocess: activation statistics.** `generate_act_scale_shift.py:25-57` hooks every `nn.Linear` of
   the fp16 model (`device_map=auto`, :99-103). Over 128 x 2048-token WikiText2 samples (seed 2, :116-118) it
   records per-input-channel max|x| as `act_scales[<module name>]`. `:59-94` records the shift
   (max+min)/2 with EMA 0.99/0.01 as `act_shifts`. Files: `./act_scales/<net>.pt` and
   `./act_shifts/<net>.pt`, with <net> = basename of `--model` (`main.py:410-414`). They are loaded only
   when `--smooth` is set (`main.py:436-438`). They are not shipped, so the script must be run per model.
3. **Capture layer-0 inputs.** `quantize/duquant.py:90-122` runs a Catcher on layers[0] and collects
   `inps[nsamples, seqlen, hidden]` plus the `attention_mask` and `position_ids` kwargs. Calibration
   loader: `main.py:421-433`, `datautils.py:35-55` (128 random 2048-token windows of WikiText2 train,
   seed 2). `rotate_inps = inps.mean(dim=0)` (`duquant.py:135`) is ONE averaged sequence. The paper (App. H)
   calls it the "mean embeddings" that drive the rotation and permutation search.
4. **Per layer, wrap the layer.** `duquant.py:163-173`: every linear becomes `QuantLinear`
   (`quantize/int_linear.py:11-76`), with a weight `UniformAffineQuantizer` and an act
   `UniformAffineQuantizer` (`quantize/quantizer.py:21-110`). Attention matmuls become `QuantMatMul`
   (`quantize/int_matmul.py:7-44`) for QK^T and PV. Norms become `DuLlamaRMSNorm` (`quantize/du_norm.py:40-63`,
   an RMSNorm with an optional bias). Q/K matmul quantizers reuse the activation params
   (`duquant.py:167-168`).
5. **Smoothing scale Λ (SmoothQuant).** `duquant.py:202-223`, for each pair key
   (`pairs` `duquant.py:59-64`: q_proj→`qkv`, o_proj→`out`, up_proj→`fc1`, down_proj→`down`):
   `s = act_scales[name]^α / max_col|W_name|^(1-α)`, clamped at 1e-5 (`:211-213`). Paper §3.2:
   Λ_j = max|X_j|^α / max|W_j|^(1-α). α is 0.6 for run.sh and 0.5 for run_lwc.sh (App. C). The weight
   statistic uses only the keyed linear (q_proj for qkv, up_proj for fc1), not the max over its siblings.
   The shift is forced to zero for llama-like nets (`use_shift=False`, `:192-195`, `:214-217`).
   `qkt_smooth_scale` is initialized to ones (`:206`) and clamped to ≥0.5 (`:264-268`), so it is a no-op
   without `--smooth_epochs` (default 0).
6. **Fuse Λ offline.** `quantize/utils.py:182-203 smooth_and_let_inplace`, using `models/transformation.py`:
   - pre-attn norm weight /= s_qkv; q,k,v weight columns *= s_qkv (`smooth_ln_fcs_inplace` :87-106).
     Zero bias buffers are registered on the norm and on q/k/v (bias = W@shift = 0).
   - pre-MLP norm weight /= s_fc1; up,gate columns *= s_fc1.
   - up_proj rows (and bias) /= s_down; down_proj columns *= s_down (`smooth_fc_fc_inplace` :108-121).
     This is valid because down's input is act(gate) ⊙ up.
   - v→o: when v_proj out == o_proj in (MHA), v rows /= s_out and o columns *= s_out, plus q/k smoothing
     (`smooth_q_k_inplace` :123-129). Under GQA the v/o shape mismatch raises inside the `try`, and the
     `except` (`utils.py:197-199`) instead calls `smooth_fc_inplace` (:131-134): o columns *= s_out, and
     `o_proj.act_quantizer.smooth_scales = s_out`, so o_proj's input is divided ONLINE
     (`quantizer.py:344-345`, in-place `x /= s`).
7. **Greedy block rotation R1 (computed from activations).** `duquant.py:272-283` sets
   `init_duquant_params=False` everywhere, turns on weight+act quant, and runs ONE forward of `rotate_inps`
   through the smoothed layer. On first call each act quantizer runs `online_duquant_cali`
   (`quantizer.py:279-303`), which calls `rotation()` (`:208-269`):
   reshape X to (-1, block_size); repeat up to `max_rotation_step` (256) times:
   find argmax/argmin entries, take `exchange_id`, set R = R~ with row/col 0 swapped with
   `exchange_id % block_size` (`utils.py:86-96 exchange_row_col`), X ← X·R. Then keep the prefix of steps
   whose recorded peak (max − min) is smallest (`:257-265`) and return the product R_ (block_size x block_size).
   Paper §3.2 Eq.(2)-(3), App. G Alg. 1.
   ONE block matrix is shared by all K = C_in/block_size blocks (paper Remark 2). Repo vs paper:
   - (a) The repo swaps on the ROW index of the flattened (tokens*K, bs) view mod bs (`:231-237`). The paper
     swaps on the outlier COLUMN.
   - (b) The selection metric is signed max−min, not max|X|.
   - (c) Q' is drawn once per `rotation()` call (`:222`), not once per step.
   Reproduce the repo behaviour if parity matters.
8. **Zigzag permutation P.** `quantizer.py:173-206 permutation_zigzag`: sort channels by max|x| (after R1),
   deal them across K = C/block_size blocks in a back-and-forth (zigzag) order, and sort each block
   descending, giving `perm` (length C). Paper §3.2 "The Permutation Transformation", Eq.(4), Thm 2. Runs
   `permutation_times` (=1) times (`:285-296`). `permutation_random` (`:143-171`) is unused.
9. **Second rotation R2.** `quantizer.py:298-302`: `rotation()` again on the permuted X. The stored transform
   is `R = stack([R1, R2])` of shape [perm_times+1, bs, bs] plus `permutation_list` [perm_times, C].
   Paper Eq.(5): G = Λ R̂(1) P R̂(2).
10. **Share transforms between siblings.** Weight quantizers copy R/perm from their own act quantizer
    (`int_linear.py:55-58`). k_proj and v_proj copy q_proj's quantizers (`models/int_llama_layer.py:119-124`).
    up_proj copies gate_proj's (`:44-46`). o_proj and down_proj compute their own. So there are 4 distinct
    transforms per layer: qkv-input, o-input, gate/up-input, down-input. `register_duquant_params`
    (`quantizer.py:426-435`) turns R/perm into buffers. The averaged input is propagated as
    `rotate_inps = qlayer(rotate_inps)` (`duquant.py:281`), with quantization on, to calibrate the next layer.
11. **Apply transform + fake-quant.** `quantizer.py:305-340 init_duquant`:
    `x = x.reshape(-1,bs) @ R[i]`, then `x = x[:, perm]` (after `.squeeze(0)`, so batch 1 is assumed,
    `:325-333`), then `@ R[-1]`. The same code path runs on the weight (out,in), transforming the input dim,
    and on the activation (…, C). Then `per_token_dynamic_calibration` (`:387-418`) reduces over the last
    dim: per token for activations, per output row for weights (group_size None). It is asymmetric min/max
    (`:413-417`, qmin 0, qmax 2^b−1) with clipping precedence `swc` → `lwc` → `lac` (`:398-406`). Then
    `fake_quant` (`:116-141`) does round-STE, clamp and dequant.
12. **Optional LWC training (epochs>0).** `duquant.py:304-356`. fp targets come from the un-smoothed fp
    layer (`:181-187`). AdamW over `bound_factor` (LWC, `:311-312`, lr `--lwc_lr` 1e-2 default; the paper
    says 5e-3) plus `post_scale` (only with `--let`, off in both recipes). MSE loss is computed on the layer
    output per sample (batch 1) for `epochs` passes, with NaN rollback (`:347-353`). LWC:
    xmax ← σ(up)·xmax and xmin ← σ(low)·xmin, one factor per output channel, init 4.0 (σ≈0.982)
    (`quantizer.py:72-83`, `:401-403`). Paper §4 "Implementation Details", OmniQuant LWC.
13. **Freeze weights.** `duquant.py:358-363`: `quant_inplace` (`quantize/utils.py:205-209`) replaces
    `module.weight` with Q(W Λ R1 P R2), still fp16 and IN THE TRANSFORMED BASIS. Weight quant is then
    turned off and act quant stays on. With epochs>0, `quant_inps` is propagated through the quantized
    layer (`:365-371`). `register_scales_and_zeros` (`:371/:377`) moves the weight quantizer's
    scale/zero to buffers. The layer goes to CPU (`:372/:378`).
14. **Attention matmuls (Q/K/V quant).** `int_llama_layer.py:93-98`: `qkt_matmul` and `pv_matmul` use
    `rotate=None`, so their quantizers carry a block_size Hadamard (`quantizer.py:104-105`; applied in
    `init_duquant` `:308-312` only when `quant_method` is set).
    - Q and K, after RoPE: Hadamard over head_dim, then per-(token,head) asymmetric `abits` quant with lac
      clip (`int_llama_layer.py:132-133`). K is quantized BEFORE it enters the cache (`:137-142`).
    - P (softmax output): 16 bit, so not quantized (`main.py:401-404`, `quantizer.py:375-376`).
    - V: per-(token,head) asymmetric `abits`, no rotation, no clip (`main.py:395-400`). Because
      `quant_method` is absent there, `init_duquant` returns early. V is quantized AFTER the cache concat
      (`int_llama_layer.py:167-168`).
    - `QuantMatMul.forward` itself never quantizes (`int_matmul.py:42-44`). The attention calls
      `quant_x1`/`quant_x2` explicitly.
    Paper App. C ("Hadamard rotation for Query·Key and attention-output·Value"), §4 ("quantize all
    intermediate activations, excluding the SoftMax output"), App. F Table F20 (query per-token asym).
15. **Evaluate in process.** `main.py:448` → `evaluate` (`main.py:38-192`): PPL on wikitext2/c4 at seqlen
    2048 plus vendored lm_eval tasks. No export.

## Tensor roles

- **Residual stream / hidden states between layers**: never transformed. fp16. Not quantized (only linear
  inputs are quantized).
- **Embeddings, final norm, LM head**: untouched, fp16. Only decoder layers are wrapped
  (`duquant.py:163-172`). Tied embeddings are irrelevant because there is no global rotation.
- **Pre-attention norm weight**: input side ÷ Λ_qkv (offline). It gains a zero bias buffer. Not quantized.
- **Attention input projections (q,k,v)**: weight columns × Λ_qkv, then input-side R1·P·R2 (one transform
  shared by q/k/v, computed on q_proj's input). Weight quant is per-output-channel asym `wbits` with swc
  0.8 (RTN) or trained LWC. Their input activation: online ÷ nothing (Λ is in the norm), then R1 P R2,
  then per-token asym `abits` with lac 0.9. Zero bias buffers are added. Pre-existing biases are
  unaffected by input-side Λ/R.
- **Query/Key states (post-projection, post-RoPE, and post q/k-norm if the model has one)**: online
  block_size Hadamard along head_dim on both Q and K, then per-(token,head) asym `abits` with lac 0.9.
  This is the **KV cache K**: stored already fake-quantized and Hadamard-rotated (`int_llama_layer.py:133,137-142`).
- **Value states / KV cache V**: stored fp in the cache. Quantized per-(token,head) asym `abits` when read
  (after concat). No rotation.
- **Attention probabilities (softmax output)**: kept at 16 bit.
- **Value/output pair**: smoothing Λ_out. Under MHA it is fused offline into v_proj rows/bias and o_proj
  columns. Under GQA (v out ≠ o in), o_proj columns × Λ_out and an ONLINE division of o's input by Λ_out.
- **Attention output projection (o_proj)**: own R1 P R2, computed on its (smoothed) input. Per-channel
  weight quant. Per-token act quant.
- **Pre-MLP norm weight**: ÷ Λ_fc1 (offline). Zero bias buffer.
- **MLP input projections (gate, up)**: columns × Λ_fc1. Shared R1 P R2 computed on gate_proj's input.
  up_proj rows (and bias) also ÷ Λ_down. Per-channel weight quant. Per-token act quant.
- **MLP output projection (down_proj)**: columns × Λ_down. Own R1 P R2 computed on the smoothed
  intermediate activation, which is where the paper's "massive outliers" live (§2). Per-channel weight
  quant. Per-token act quant (in-dim must be divisible by block_size).
- **Linear biases**: no learned role. Smoothing adds `W@shift` (zero for llama-like) as bias buffers on
  q,k,v,up,gate,down. A real v bias (MHA) is ÷ Λ_out. Shifts are disabled for llama/mistral.
- **Activations (all linear inputs)**: dynamic per-token asymmetric, scales computed at runtime. No static
  activation scales are stored.
- **Weight scales/zeros**: buffers after `register_scales_and_zeros`. They are not used at runtime because
  the weight is already dequantized fp16.

## Invariants

- **Smoothing (Λ)**: for each linear, X·W^T = (X/s)·(W·diag(s))^T. 1/s must be moved into the PRODUCER of
  X, and only through elementwise ops: RMSNorm weight, up_proj rows (gating is elementwise), and v_proj
  rows only when each o_proj input channel corresponds to exactly one v_proj output channel (MHA). If that
  is impossible (GQA repeat_kv, q/k-norm, non-linear producers), the repo falls back to an online division
  of the activation (o_proj under GQA). Λ may never be pushed through act(gate), q_norm/k_norm, softmax or RoPE.
- **Rotation/permutation**: G = R1·P·R2 is orthogonal, so (X G)(W G)^T = X W^T. Each linear's weight
  must be transformed with exactly the same G its input activation gets online: same R stack, same
  `perm`, same block_size, applied on the input-channel axis (last dim of X, dim 1 of W[out,in]).
  Linears sharing an input (q/k/v, gate/up) may share G. Nothing else is required to change, because the
  residual stream is untouched.
- **Block structure**: in_features % block_size == 0 for every quantized linear. Zigzag fills
  K = C/block_size blocks (`quantizer.py:183,203-204`), and a remainder would silently drop channels.
  `block_size=-1` means 4096 (`:99-102`).
- **Q/K Hadamard**: the same H must be applied to Q and K on the head_dim axis, AFTER every per-head op
  (RoPE, q/k norm). Then Q H (K H)^T = Q K^T. Because it reshapes to (-1, block_size), block_size must
  divide head_dim, otherwise rows of different tokens are mixed and QK^T breaks.
- **Activation transform order**: ÷smooth_scales (if any) → R1 → P → R2 → quantize. The weight is stored
  already multiplied by Λ and by G and then fake-quantized. Applying G again or skipping it at runtime
  computes the wrong function.
- **K before cache, V after**: K entries in the cache are rotated and quantized. Any cache replacement must
  keep Q and cached K in the same Hadamard basis.
- **Permutation axis**: the repo's `x[:, perm]` after `squeeze(0)` (`quantizer.py:325-333`) is correct only
  for batch 1 (3-D input). A port must permute the last dim.
- **Masking**: the repo attention adds an explicit additive 4-D mask (`int_llama_layer.py:157-163`). If
  the mask is None it computes NON-causal attention.

## Calibration

- **Data**: WikiText2 train, 128 random windows of 2048 tokens, seed 2 (`main.py:203-210,346`,
  `datautils.py:35-55`; paper App. C). The same data feeds `generate_act_scale_shift.py` (Λ statistics,
  computed on the original fp16 model, all layers in one pass).
- **Rotation/permutation search**: uses the MEAN of the 128 layer inputs (one 2048 x hidden sequence,
  `duquant.py:135`), propagated layer by layer through the already-quantized layers (`:281`). Each
  QuantLinear act quantizer searches ≤ `max_rotation_step`=256 greedy steps per rotation, block_size 128,
  permutation_times 1 (App. C). There is one forward per layer and no gradient. The transform depends on
  the torch RNG (QR of randn).
- **epochs=0 (run.sh, "DuQuant")**: no optimization. RTN weights with symmetric clip ratio swc=0.8 on both
  min and max. Activations clipped with lac=0.9. `--lwc` is accepted, but swc takes precedence in
  `per_token_dynamic_calibration`, so the LWC factors exist and are unused. `fp_inps`/`quant_inps` are not
  computed or propagated (`duquant.py:181,365`).
- **epochs>0 (run_lwc.sh, "DuQuant+LWC")**: 20 epochs x 128 samples, batch 1, AdamW (lwc_lr 1e-2 repo
  default, wd 0). MSE between the fp layer output (fp inputs) and the quantized layer output (quantized
  inputs) under AMP. Only the LWC bound factors are trained (LET post-scales only with `--let`; smoothing
  only with `--smooth_epochs`). The rotation is fixed before training. `main.py:320-321` requires `--lwc`
  or `--let` when epochs>0.
- **Precision**: the calibration forward uses fp16 autocast. The model is loaded fp16 (`models/LMClass.py:29`).
  `deactive_amp` only applies for 8-bit (`main.py:323-324`).

## Save and reload

- **What the repo writes**: only `<save_dir>/duquant_parameters.pth`, re-saved after every layer
  (`duquant.py:372-381`). It is a dict {layer_idx: OrderedDict} from `duquant_state_dict`
  (`quantize/utils.py:55-64`), holding:
  - parameters matching smooth/bound_factor/trans/post: `qkv|fc1|down|out|qkt _smooth_scale/_shift`, LWC
    `upbound_factor/lowbound_factor`;
  - buffers matching `R`/`permutation_list`/`init_duquant_params`: each act and weight quantizer's R stack
    [2,bs,bs] and perm [1,C].

  It does NOT save quantized weights, the integer codes, weight scales/zeros, the online `smooth_scales`
  attribute (GQA o_proj), the model config, or a HF checkpoint. There is no packing, no int kernel and no
  `save_pretrained`. The README's "save the quantized models" means these parameters only.
  `mtbench_generate.py` also only re-saves act stats.
- **Reload path**: `--resume <dir>` re-runs the whole `duquant()` on the fp16 base model
  (`duquant.py:158-161,197-204,275-276,304-306`). `load_state_dict(strict=False)`, `load_smooth_params`,
  `load_duquant_params` (exec-assigns R/perm, `int_llama_layer.py:385-388`) and `load_lwc_params` skip the
  rotation search and LWC training, then re-fuse Λ and re-fake-quantize the weights. It still needs the
  calibration loader (Catcher), the act_scales file when `--smooth` (`main.py:436-438`), and the same
  family glue. Bug: with `--let` it calls nonexistent `qlayer.load_post_parameter` (`duquant.py:288`;
  the method is `load_post_params`).
- **In-memory artifact**: decoder layers become `QuantLlamaDecoderLayer` holding fp16 fake-quant weights
  Q(WΛG) in the transformed basis, with act quant ON. It is evaluated in-process only (`main.py:448`).
- **Runtime needs after any reload**: custom modules are mandatory. Per linear input: optional ÷Λ_out
  (GQA o_proj), block-diag R1, perm, R2, per-token asym act fake-quant (lac 0.9). Per attention: Hadamard
  plus abits quant on Q and K, and abits quant on V. A stock HF/vLLM model loaded with these weights
  computes the wrong function, because its activations are not rotated. The transforms cannot be folded
  into norms or neighbouring layers: they differ per linear and the norm is elementwise. The paper's
  speedups (§4.2, App. E.1) use QuaRot's W4A4 kernels, which are not in this repo.
- **Port options**:
  - (i) Save the fp16 fake-quant transformed weights plus R/perm/smooth_scales buffers, and rebuild
    QuantLinear-like modules on reload.
  - (ii) Keep the repo's approach: save `duquant_parameters.pth` and recompute from the base model.
  - (iii) For a stock fp16 HF checkpoint that keeps only the WEIGHT-quant error, fold back
    W_eff = Q(WΛG)·G^T·Λ^-1 into the original basis. Activation and KV quant error are then not
    represented. This is the only stock-loadable form.

## Porting notes

**Entry point / pipeline**: `get_rot.py` (once) → `generate_act_scale_shift.py --model M` (per model) →
`main.py main()` (`run.sh`) → `quantize/duquant.py:duquant()` (`main.py:439`) → `evaluate()` (`main.py:448`).

**Generic building blocks (reuse as-is):**
- `quantize/quantizer.py UniformAffineQuantizer`:
  - `rotation` :208-269, `permutation_zigzag` :173-206, `online_duquant_cali` :279-303;
  - `init_duquant` :305-340 (fix the batch-1 perm axis);
  - `per_token_dynamic_calibration`/`fake_quant` :387-418/:116-141, which include LWC/swc/lac;
  - `register_duquant_params`/`copy_duquant_params` :426-445.
- `utils.py get_rot / exchange_row_col / get_hadamard` (:86-123) and `get_rot.py`.
- `quantize/int_linear.py QuantLinear`, `quantize/int_matmul.py QuantMatMul`.
- `models/transformation.py` smoothing fusers: `smooth_ln_fcs_inplace`, `smooth_fc_fc_inplace`,
  `smooth_fc_inplace`, `smooth_q_k_inplace`.
- `quantize/du_norm.py DuLlamaRMSNorm` (generic RMSNorm plus bias).
- `quantize/utils.py`: `set_quant_state`, `quant_inplace`, `set_init_duquant_params_state`,
  `duquant_state_dict`, `register_scales_and_zeros`, `lwc_parameters`.
- `generate_act_scale_shift.py` (hooks any nn.Linear) and `datautils.get_loaders`.

**Family glue:**
- `models/int_llama_layer.py`, `models/int_mistral_layer.py`: copies of HF-4.35 Llama/Mistral
  attention/MLP/decoder forward, with quant hooks and sibling-sharing logic.
- `quantize/utils.py smooth_and_let_inplace` / `smooth_and_quant_temporary`: hard-coded llama attribute
  names, an OPT `else` branch, and exception-driven GQA detection.
- `quantize/duquant.py`: family switch `:52-80` and `:127-131`, keyed on `args.net.lower()` containing
  "llama"/"vicuna" (→ QuantLlamaDecoderLayer) or "mistral" (→ QuantMistralDecoderLayer), else
  `ValueError("Only support for llama/Llama-2/Llama-3/Vicuna/Mistral now")`. Also the `pairs` dict, the
  `layer_name_prefix="model.layers"` act_scales key prefix, and Catcher kwargs.
- `main.py net_choices` (:22-35, llama-7b..65b, Llama-2/3, Vicuna-1.5, mistral-7b). The assert is
  commented out (:343). `args.net` = model basename. `model_family` is used only in cache file names.
  "Qwen3-8B" hits the ValueError.
- `models/LMClass.py` (fp16 load, slow tokenizer) and the vendored old `lm_eval`.

**Version-sensitive (pinned transformers 4.35.2 → target 4.51.3):**
- Rotary: the glue deep-copies `org_module.rotary_emb` (`int_llama_layer.py:73`) and calls
  `rotary_emb(v, seq_len=)` and `apply_rotary_pos_emb(q,k,cos,sin,position_ids)` (:129-130). In 4.51 the
  rotary module lives on the model (`model.model.rotary_emb(x, position_ids)`), layers receive
  `position_embeddings=(cos,sin)`, and `apply_rotary_pos_emb(q,k,cos,sin)` takes no position_ids.
- Layer call signature: 4.51 calls `decoder_layer(h, attention_mask=, position_ids=, past_key_value=,
  output_attentions=, use_cache=, cache_position=, position_embeddings=, **kwargs)`. The repo layer accepts
  none of the last three, so it needs `**kwargs`. Catcher (`duquant.py:102-108`) must also capture
  `position_embeddings` (and `cache_position`), not just mask and position_ids.
- Returns: 4.51 attention returns `(out, weights)` and the decoder layer returns `(hidden,)` (+weights).
  The repo returns `(out, weights, past_kv)`. Only `[0]` is read by the model loop, but keep the arity
  compatible.
- Cache: the repo uses tuple concat `past_key_value[0]` (:127-142). 4.51 passes a `Cache`
  (`DynamicCache.update(k,v,layer_idx,…)`). Eval and calibration use `use_cache=False`, but generation
  needs the Cache API and a `layer_idx`. Keep "quantize K before update, quantize V after" if parity is wanted.
- Mask: the repo requires a 4-D additive mask of shape (bsz,1,q,kv) and raises otherwise (:157-163). In
  4.51, `_update_causal_mask` returns None for sdpa/flash when it can rely on `is_causal`, which makes the
  repo attention silently non-causal. Force `attn_implementation="eager"` (the LMClass default passes
  it via config, `models/LMClass.py:23-25`) or build the causal mask.
- head_dim: the repo derives head_dim = hidden/num_heads and raises on mismatch (:62,67-71). Newer
  configs carry `config.head_dim` (Qwen3: explicit). Use the config value.
- Models with q_norm/k_norm (e.g. Qwen3): the glue has no slot for them. They sit between projection and
  RoPE. The Q/K Hadamard and quant must come after them, and q/k smoothing (`smooth_q_k_inplace`) is
  invalid across them (it is skipped anyway under GQA, and its scale is ones).
- dtype: fp16 is forced (`LMClass.py:29`, `duquant.py:88,263,361`, `int_llama_layer.py:260 .half()`).
  bf16-native models may overflow in fp16 on massive outliers, so check activations.
- Torch 2.x: `torch.cuda.amp.autocast`/`GradScaler` are deprecated (warnings only).
- `pickle.load("Rot.pkl")` is cwd-relative. `AutoTokenizer(use_fast=False)` requires a slow tokenizer.
  `datasets==2.17.1` dataset ids.

**Online-transform consequence**: any reload that keeps activation/KV quantization needs custom runtime
modules (a quantized linear wrapper with R/perm buffers and a per-token act quantizer, plus the
attention Q/K Hadamard and Q/K/V quantizers). There is no repo path to a stock or vLLM-loadable artifact.

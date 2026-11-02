# FlatQuant method model (arXiv 2410.09426, repo @ 9d88ffcb)

Paper: `fq-src/paper.txt`. Repo paths are relative to `fq-src/repo`. "Paper §x" refers to the section headings in
paper.txt (§3.1 at line 181, §3.2 at 235, §3.3 at 257, §4.1 at 267, App. B.1 at 1490, B.2 at 1575, C.3 at 2406).

FlatQuant is a W4A4(KV4) post-training method. Each group of linear layers that reads the same activation gets its
own learnable invertible affine transform `P`. For the full hidden or intermediate width, `P` is Kronecker-factored as
`P1 ⊗ P2`, preceded by a learnable per-channel scale `diag(c)`. Learnable clipping is applied to weights and
activations. All of these parameters are trained block by block to minimise the MSE between the fp block output and
the fake-quantized block output. Afterwards the inverse transforms, scales and weight clips are folded into the
weights and norms. The forward Kronecker transforms stay **online** at runtime, each followed by per-token activation
quantization. Unlike QuaRot/SpinQuant, the residual stream is **never rotated**: every transform sits after a norm or
inside a block, so embeddings, the final norm and the LM head are untouched.

Entry point used by the repo's own pipeline: `main.py:12-91` (`main()`): `get_model` → `apply_flatquant_to_model` →
(`cali_flat_quant` | `load_flat_matrices` | `load_flat_parameters`) → optional `save_flat_matrices` →
`reparameterize_model` → RTN or GPTQ weight quantization → optional `--quantized_save` → PPL and lm-eval.

### Generic building blocks vs model-family glue

- **Generic (model-agnostic):**
  - `flatquant/flat_linear.py` (`FlatQuantizedLinear`: weight transform, LWC, fake quantization, reparameterize).
  - `flatquant/trans_utils.py` (`SVDSingleTransMatrix`, `SVDDecomposeTransMatrix`, `InvSingleTransMatrix`,
    `InvDecomposeTransMatrix`).
  - `flatquant/flat_utils.py` (`kronecker_matmul`, `reparameterize_ln`, matrix save/load, packed safetensors save).
  - `flatquant/quant_utils.py` (`ActivationQuantizer` with LAC, `WeightQuantizer`, STE rounding).
  - `flatquant/function_utils.py` (decompose-dim choice, random orthogonal init, inverse, SmoothQuant-style scale
    init, parameter selection by name).
  - `flatquant/train_utils.py::cali_flat_quant`. Its loop is generic but assumes `model.model.layers`,
    `model.model.embed_tokens`, and `layer.self_attn` / `layer.mlp` exposing `_ori_mode` and `init_diag_scale`.
  - `gptq_utils.py`: GPTQ and RTN. The GPTQ sequential list hard-codes `self_attn.{q,k,v,o}_proj.linear` and
    `mlp.{up,gate,down}_proj.linear` (`gptq_utils.py:203-208`).
  - `deploy/` kernels (`deploy._CUDA` int4 GEMM, quant/dequant, FlashInfer 4-bit KV) plus the Triton
    `kron_matmul` / `block_matmul`, and `deploy/nn` (`Linear4bit`, `Quantizer`, `OnlineTrans`, `RMSNorm`).
- **Glue (per model family):**
  - `flatquant/model_tools/{llama,llama31,qwen,deepseekv3}_utils.py`. These replace Attention and MLP with FlatQuant
    versions, decide where each transform sits, and write the custom attention forward.
  - `flatquant/model_utils.py`: dispatch on model-name string.
  - `flat_utils.reparameterize_model` (`flat_utils.py:30-40`): hard-codes `input_layernorm`, `post_attention_layernorm`,
    `self_attn.ln_trans` and `mlp.up_gate_trans`.
  - `deploy/transformers/modeling_llama.py`: the packed-int4 HF model, **Llama only**.
  - `vllm_custom/.../{llama,qwen2}_flatquant.py`: vLLM fake-quant models.

## Steps

1. **Wrap the block's linears and attach transform modules** (glue).
   - Paper §3.2 "Self-Attention" and "Feed-forward Network". Repo `flatquant/model_tools/qwen_utils.py:323-331`
     (Llama: `llama_utils.py`); per-linear wrapper `flatquant/flat_linear.py:8-26`.
   - Every attention or MLP linear becomes `FlatQuantizedLinear`. That class holds a per-output-channel symmetric
     `WeightQuantizer` and a per-token `ActivationQuantizer`.
   - Six transforms per block (`qwen_utils.py:36-47`, `141-161`):
     - `P_a` = `self_attn.ln_trans`: Kronecker `P1⊗P2` on the shared q/k/v input, plus an optional `diag(c_a)`.
     - `P_o` = `self_attn.o_trans`: a single `H×H` matrix over the **head axis** (H = num attention heads) of the
       o_proj input.
     - `P_h` = `self_attn.kcache_trans`: a single `d×d` matrix (d = head_dim), applied per head to queries and keys.
     - `P_v` = `self_attn.vcache_trans`: a single `d×d` matrix applied per head to values.
     - `P_ug` = `mlp.up_gate_trans`: Kronecker on the shared up/gate input, plus `diag(c_ug)`.
     - `P_d` = `mlp.down_trans`: Kronecker on the down_proj input, plus `diag(c_d)`.
   - When each transform is created:
     - `P_a`, `P_o`, `P_ug` and `P_d` exist when `w_bits<16 or a_bits<16`.
     - `P_h` exists when `k_bits<16 or q_bits<16`.
     - `P_v` exists when `v_bits<16 or w_bits<16 or a_bits<16` (`qwen_utils.py:146-161`).
   - Only `P_a`, `P_ug` and `P_d` are Kronecker-decomposed. `P_o` (over heads) combines with `P_v` (over head_dim)
     into `P_o⊗P_v` on the o_proj input (paper §3.2: "we further fuse P_o with P_v").

2. **Choose the Kronecker factor sizes.**
   - Paper §3.1 "Kronecker Decomposition", eq. 3: `n1*,n2* = argmin(n1+n2), n1·n2=n, n1≤n2`.
   - Repo `flatquant/function_utils.py:11-21` (`get_decompose_dim`). It searches `a²−b²=n` upward from `ceil(√n)` and
     returns `(a−b, a+b)`.
   - Applying `P1⊗P2` to a row vector: `flatquant/flat_utils.py:6-17` (`kronecker_matmul`). It reshapes x to
     `(…, n1, n2)` and computes `P1ᵀ · X · P2`, i.e. `vec(X)(P1⊗P2)` in row-major order.

3. **Parameterise each transform and its inverse.**
   - Paper App. B.1, "SVD and AMP Training" vs "Direct Inversion and FP32 Training".
   - SVD form (default):
     - `P = U·diag(σ)·Vᵀ`, where U and V use PyTorch's Cayley orthogonal parametrization and σ is learnable.
     - The inverse-transpose `P⁻ᵀ = U·diag(1/σ)·Vᵀ` comes in closed form.
     - Repo `trans_utils.py:8-46` (single) and `57-116` (decomposed).
   - `--direct_inv` form:
     - The parameter is a free `nn.Linear` weight.
     - The inverse is `torch.inverse` in float64, transposed (`function_utils.py:35-37`; `trans_utils.py:128-213`).
   - Initialisation is a random orthogonal matrix (QR of a Gaussian, `function_utils.py:24-32`). Paper §4.1:
     "random affine transformation matrices as the starting point".
   - The diag scale is applied first. `forward(x) = kron(x*c)`, and `forward(W, inv_t=True) = kron_inv(W/c)`
     (`trans_utils.py:85-103`, `190-204`).

4. **Initialise the per-channel scale `c` (SmoothQuant style).**
   - Paper §3.1 "Per-channel Scaling", formula in §2.1. Repo `train_utils.py:98-111`,
     `qwen_utils.py:65-73, 97-106, 174-177, 304-310`, and `function_utils.py:7-8`.
   - During the fp pass of each block, record per-channel `max|X|` for three inputs: the qkv input (`ln_smax`), the
     up/gate input (`up_smax`) and the down_proj input (`down_smax`).
   - Then set `c = max|W|^(1−α) / max|X|^α`, clamped at 1e-5. α = `--diag_alpha`, default 0.3 (`args_utils.py:91`).
   - `max|W|` is taken over the concatenated weights of every consumer: `[q;k;v]` for `c_a`, `[up;gate]` for `c_ug`,
     and `down` for `c_d`.
   - With `--diag_init one_style`, `c` starts at ones.

5. **Learnable clipping thresholds.**
   - Paper §3.1 "Learnable Clipping Thresholds".
   - Weights (LWC, `--lwc`; `flat_linear.py:18-33`):
     - Two per-output-row logits, `clip_factor_w_max` and `clip_factor_w_min`, initialised to 4.0.
     - Clip `W` to `[σ(β_min)·rowmin, σ(β_max)·rowmax]`.
   - Activations (LAC, `--lac`; `quant_utils.py:61-67, 96-98`):
     - One scalar pair, `clip_factor_a_max` and `clip_factor_a_min`, per `ActivationQuantizer`, initialised to 4.0.
     - Scale the per-token max/min by their sigmoids.
     - This also covers the K and V cache quantizers (`qwen_utils.py:125-133`, `lac=args.lac`).
   - Clipping is applied **in the transformed space**: after `P⁻¹` on W and after `P` on X (`flat_linear.py:48-52`).

6. **Fake-quantized block forward used during training.** Paper §3.1 eq. 2-3 and §3.2.
   - Linear (`flat_linear.py:45-67`):
     - Weight path: `W' = clip(kron_inv(W / c))`. For v_proj, also `W' ← (W'ᵀ·blockdiag(P_v))ᵀ` and the bias
       `b' = b·blockdiag(P_v)`.
     - Quantize W': per-output-channel, symmetric, signed range [-8,7] at 4 bits (`quant_utils.py:150-212`).
     - Quantize the already-transformed input per token (`quant_utils.py:85-119`), with STE rounding
       (`quant_utils.py:3-7`).
   - Attention (`qwen_utils.py:163-172, 204-281`):
     - `x_t = ln_trans(x)` is computed once and shared by q, k and v.
     - RoPE is applied to q and k. **Then** `q ← q·P_h⁻ᵀ` and `k ← k·P_h`, and k is quantized
       (`quant_kcache`, `190-202`); q is quantized only if `q_bits<16`.
     - v is already in `P_v` space (from v_proj) and is quantized (`quant_vcache`, `183-188`).
     - Run attention, then reshape the output to `(…, H, d)`.
     - Left-multiply by `P_oᵀ`. The output side already carries `P_v`, so the result is `X·(P_o⊗P_v)`.
     - o_proj consumes it with weight `W·(P_o⊗P_v)⁻ᵀ`, via `qa_trans=[P_o⁻ᵀ, P_v⁻ᵀ]` (`qwen_utils.py:269-275`).
   - MLP (`qwen_utils.py:49-63`): `x_t = up_gate_trans(x)`, then up and gate, then `silu(g)*u`, then `down_trans`,
     then down.

7. **Block-wise calibration.**
   - Paper §3.1 "The Training Objective" eq. 4, §4.1 "Implementation Details". Repo `flatquant/train_utils.py:14-171`.
   - Capture the inputs of layer 0 for `nsamples` sequences with a `Catcher`. The Catcher also records
     `attention_mask` and `position_ids` (`train_utils.py:38-69`).
   - For each layer i:
     1. Cast to fp32 (`:96`).
     2. Compute fp targets with `_ori_mode=True` (original linears, no transforms, no quantization), recording the
        activation maxima (`:98-104`).
     3. Initialise `c` (`:105-111`).
     4. Collect the parameter groups (`:116-127`): `trans.linear` (all P factors) and `trans.diag_scale` at
        `flat_lr`; `clip_factor_w` and `clip_factor_a` at `10×flat_lr`.
     5. Train with AdamW and `CosineAnnealingLR(T_max=epochs·nsamples/bsz, eta_min=flat_lr·1e-3)`, with optional
        linear warmup (`:129-135`).
     6. Loss = `MSE(fp_out, quant_out)`, normalised as `loss/loss.detach()` (unit-magnitude loss, `:142-151`).
   - The input to layer i+1 is the **fp output** of layer i, not the quantized output (`fp_inps, fp_outs` swap at
     `:155`).
   - Trained parameters are checkpointed per layer to `exp_dir/flat_parameters.pth` (`:157-158`).

8. **(Optional) save the transform matrices.** Paper: n/a (artifact). Repo `main.py:36-37`, `flat_utils.py:65-78`.
   - `rep_matrix_only` materialises P, P⁻ᵀ and the factors (`to_eval_mode`).
   - Then every parameter whose name contains `trans.matrix`, `trans.diag_scale`, `clip_factor_w` or `clip_factor_a`
     is saved to `exp_dir/flat_matrices.pth` (dict: layer_idx → state-dict slice).

9. **Reparameterise: fold everything offline that can be folded.**
   - Paper §3.1 (merge c), §3.2 (merge `c_d` into W_u, fuse P_v into P_o), Fig. 3(b). Repo `main.py:38`,
     `flat_utils.py:20-40`, `flat_linear.py:82-97`, `qwen_utils.py:80-95, 283-302`. All work is done in float64.
   - `to_eval_mode()` freezes `matrix_left/right` and `matrix_left_inv/right_inv` (= P_i⁻ᵀ), or `matrix` and
     `matrix_inv_t`, and deletes the SVD factors.
   - q/k/v/gate/up/down: `W ← kron(W/c, P1⁻ᵀ, P2⁻ᵀ) = W·diag(1/c)·(P1⊗P2)⁻ᵀ`, then LWC-clipped and stored
     (`flat_linear.py:82-97`).
   - v_proj additionally gets the output-side `P_v`, applied to both weight and bias (`flat_linear.py:91-94`).
   - o_proj: `W ← kron(W, P_o⁻ᵀ, P_v⁻ᵀ)` (`qwen_utils.py:298-301`).
   - `c_a` → `input_layernorm.weight *= c_a`; `c_ug` → `post_attention_layernorm.weight *= c_ug`
     (`flat_utils.py:20-27, 36-39`). Both transforms then set `use_diag=False`.
   - `c_d` → `up_proj` output rows `*= c_d` (`qwen_utils.py:89-95`), so that `silu(g)·(u·c_d) = (silu(g)·u)·c_d`.
   - Afterwards each `FlatQuantizedLinear` has `_eval_mode=True` and its forward becomes `act_quant(x) → linear`
     (`flat_linear.py:75-80`).

10. **Quantize the weights (RTN by default).**
    - Paper §4.1 "Quantization": RTN by default, GPTQ optional. Repo `main.py:41-47`, `gptq_utils.py:272-310` (RTN),
      `151-269` (GPTQ).
    - RTN covers every `nn.Linear` inside the decoder layers. It is per-output-channel min-max (symmetric unless
      `--w_asym`), applied to the already clipped and transformed weights. Clipping was learned, so no MSE grid search
      is needed (`--gptq_mse` is documented as incompatible with LWC, `args_utils.py:55-57`).
    - `lm_head` is skipped. It is outside `model.model.layers` anyway.
    - The fp weights are overwritten by their fake-quantized values. The `quantizers` dict holds the per-row scales.

11. **Runtime (online) computation.** Paper §3.2 (first paragraph) and §3.3; Fig. 3(b), blue = online.
    - Online per block:
      - `ln_trans` Kronecker on the qkv input (`qwen_utils.py:164-165`).
      - Per-token activation quantization before every linear.
      - `P_h⁻ᵀ` and `P_h` on q and k after RoPE, then KV quantization.
      - `P_oᵀ` over heads on the attention output (`qwen_utils.py:269-271`).
      - `up_gate_trans` and `down_trans` Kronecker (`qwen_utils.py:50-61`).
    - Kept in fp16: norms, transforms, RoPE, attention scores (paper §3.2).
    - The packed implementation fuses each Kronecker transform with int4 quantization in one Triton kernel
      (`deploy/kernels/kron_matmul.py:24-130`, `deploy/functional/online_trans.py:113-141`). Matmuls then use a
      CUTLASS int4 GEMM, and the KV cache uses FlashInfer 4-bit pages.

12. **(Optional) export packed int4.**
    - Paper §3.3 (kernels). Repo `main.py:49-51`, `flat_utils.py:97-204`, `deploy/functional/quantization.py:49-56`.
    - Recompute the integer weights from the fake-quant weights and per-row scales, then `pack_i4` them (two
      signed nibbles per uint8 along the input dim).
    - Every other parameter is written as fp16 safetensors, together with `quantizer.<layer>.scale/zero/maxq`.

## Tensor roles

Notation: H = attention heads, Hkv = KV heads, d = head_dim; "input side" means `W ← W·M⁻ᵀ` paired with `x ← x·M`.

| Role | Steps that touch it | How |
|---|---|---|
| Residual stream (hidden states between blocks) | none | Never rotated or scaled. Transforms start after each norm. Paper §3.2 "Layer Normalization" calls this out as a design choice. |
| Pre-attention norm weight (`input_layernorm` γ) | 4, 9 | Absorbs `c_a` (`γ ← γ·c_a`). Stays fp16. The norm itself is kept. |
| Pre-MLP norm weight (`post_attention_layernorm` γ) | 4, 9 | Absorbs `c_ug`. Stays fp16. |
| Final norm, token embeddings, LM head | none | Untouched, kept in the original dtype. RTN/GPTQ skip `lm_head` (`gptq_utils.py:227-229, 292-294`). No fusion into the embedding or the head. |
| Attention input projections (q, k, v weights) | 5, 6, 9, 10 | Input side: `diag(1/c_a)` then `(P1⊗P2)_a⁻ᵀ`. LWC clip. W4 per-channel sym. |
| Value projection (v weight) | additionally 9 | **Output side** as well: right-multiplied per head by `P_v` (block-diagonal over Hkv heads). |
| Linear biases of q/k | none | Unchanged. Only input-side transforms act on q/k (`P_h` is applied online after RoPE, never folded into k_proj). |
| Linear bias of v | 6, 9 | Multiplied per head by `P_v`, the same output-side transform as the v weight (`flat_linear.py:62-63, 93-94`). |
| Queries / keys (post-RoPE) | 6, 11 | Online `q·P_h⁻ᵀ` and `k·P_h`, per head. K is quantized per (token, head) to 4-bit asymmetric with LAC. Q stays fp unless `q_bits<16`. |
| Values / V cache | 6 | Already in `P_v` space (folded into v_proj). Quantized per (token, head) to 4-bit asymmetric with LAC. |
| Attention-output / o_proj pair | 6, 9, 11 | Activation: online `P_oᵀ` over the head axis (it already carries `P_v`), then A4 per-token sym. Weight: input side `(P_o⊗P_v)⁻ᵀ`, LWC, W4. |
| MLP input projections (up, gate weights) | 4, 5, 9, 10 | Input side `diag(1/c_ug)·(P1⊗P2)_ug⁻ᵀ`. LWC. W4. |
| up_proj output rows | 9 | Scaled by `c_d`, the down-transform's per-channel scale. This needs a projection whose output enters the gated product linearly, i.e. up, not gate. |
| MLP output projection (down weight) | 4, 5, 9, 10 | Input side `diag(1/c_d)·(P1⊗P2)_d⁻ᵀ`. LWC. W4. |
| Linear activations (inputs of every quantized linear) | 6, 11 | Transformed online, then A4 per-token (symmetric by default) with a learnable scalar clip per quantizer. |
| Transform matrices, scales, clip logits | 3-9, 12 | Trained in fp32. Stored as fp32 Parameters after `to_eval_mode`; cast to fp16 in the packed export. |

## Invariants

- **Input-side transform**: for a linear `y = xWᵀ + b` and any invertible `M`, `(xM)(WM⁻ᵀ)ᵀ + b = xWᵀ + b`. The bias is
  unchanged. All linears that read the **same** activation must share one `M` (q/k/v share `P_a`, up/gate share
  `P_ug`), because the activation is transformed and quantized once.
- **Diagonal scale**: `M = diag(c)·(P1⊗P2)`. `x·diag(c)` must be produced by the preceding op and `diag(1/c)` folded into
  the consumer weights. The producer must be elementwise-linear in that channel: an RMSNorm γ (no bias here), or the
  output rows of a linear whose output enters the next op multiplicatively and linearly (`silu(g)·u`, so scale u).
  The scale cannot be pushed through a nonlinearity (gate) or across a residual add.
- **Output-side transform**: right-multiplying a linear's output by `M` requires `W ← (WᵀM)ᵀ` **and** `b ← bM`.
  `P_v` qualifies because attention is linear in V: `softmax(·)·(VM) = (softmax(·)·V)M`, per head. With GQA the same
  `P_v` must be used for every KV head and for every query head that shares it. The repo uses a single `P_v` for all
  heads of a layer. The consumer (o_proj) must then undo it on its input, together with `P_o`: o_proj weight
  `← W·(P_o⊗P_v)⁻ᵀ`, activation `← P_oᵀ·X_heads`.
- **Q/K transform**: `qkᵀ` is unchanged under `q→qP_h⁻ᵀ`, `k→kP_h`. This must happen **after** RoPE because RoPE does
  not commute with a general `P_h`. Consequently `P_h` cannot be folded into q/k weights or biases and stays online.
- **No residual-stream transform**: embeddings, the final norm, the LM head and the residual adds see the original
  basis. Hence there is no norm-into-head fusion and no embedding rotation, so tied embeddings are unaffected.
- **Folding precision**: all folds happen in float64 and are then cast back to the original dtype (`flat_linear.py:85`,
  `flat_utils.py:24`, `qwen_utils.py:93`).
- **Clip in transformed space**: LWC/LAC thresholds were learned on transformed tensors. They must be applied after the
  fold (LWC is baked in by `reparameterize`) and after the online transform (LAC), never on original-basis tensors.
- **Quantizer after transform**: activation fake-quant is always the step after the online transform. In the packed
  path the two are one fused kernel, so per-token scales are computed on transformed values.
- **Reload must recreate identical modules**: `load_state_dict(strict=False)` silently drops keys whose modules do not
  exist (`flat_utils.py:92`). The same `--add_diag/--lwc/--lac/--direct_inv/--separate_vtrans` and bit flags must be
  used at reload.

## Calibration

- **Data**:
  - Default `--cali_dataset wikitext2` (`args_utils.py:65`), also `c4` and `pile`.
  - 128 random contiguous windows (`--nsamples 128`, `args_utils.py:67`), each 2048 tokens (`model.seqlen = 2048`,
    `model_utils.py:24,38,58`), drawn from the joined, tokenized train split (`data_utils.py:12-30`).
  - Datasets load from local paths `./datasets/wikitext` and `./datasets/allenai/c4` (`data_utils.py:14,18,36,43`).
  - Seed 0 (`args_utils.py:34`, `main.py:14`).
  - Paper §4.1: 128 sentences × 2048 tokens from WikiText-2. Paper App. C.3 says results are robust to the
    calibration set.
- **What is optimised**:
  - Per block, the MSE between fp block output and fake-quantized block output (paper eq. 4), using the normalised
    loss.
  - Parameters: all transform factors (`--cali_trans`), per-channel scales (`--add_diag`), weight clips (`--lwc`) and
    activation/KV clips (`--lac`).
  - Training runs only if at least one of these is set (`main.py:34`).
- **Schedule (argparse defaults vs repo scripts)**:

  | Flag | argparse default | Scripts |
  |---|---|---|
  | `--epochs` | 15 (`args_utils.py:64`) | `--epoch 15` |
  | `--cali_bsz` | 4 (`:69`) | 4 |
  | `--flat_lr` | **1e-5** (`:71`) | **5e-3** (paper §4.1), clip lr = 10× = 5e-2 (`train_utils.py:123,126`) |
  | `--diag_init` | `sq_style` (`:89`) | n/a |
  | `--diag_alpha` | 0.3 (`:91`) | n/a |
  | `--warmup` | off | off |

  - Optimiser and schedule: AdamW, cosine annealing to `flat_lr·1e-3`.
  - Steps per layer = 15 × (128/4) = 480.
- **Precision**:
  - AMP is on by default: autocast fp16 for `LlamaForCausalLM`, **bf16 for every other model class**
    (`train_utils.py:24-29`), with the SVD parametrization. This is the paper's default (App. B.1).
  - The repo's Qwen-2.5 scripts add `--deactive_amp --direct_inv`: FP32 training with a float64 direct inverse.
    The paper's App. B.1 notes that AMP can be numerically unstable for some models.
  - The block is cast to fp32 for training (`train_utils.py:96`) and back to its original dtypes afterwards
    (`:160-163`).
- **Bits in the repo's W4A4KV4 scripts** (`scripts/*/w4a4kv4.sh`; argparse defaults are all 16, so nothing is
  quantized by default, `args_utils.py:38-120, 148`):
  - Weights: `w_bits 4`, symmetric, per-channel, RTN (no `--gptq`).
  - Activations: `a_bits 4`, symmetric, per-token. `a_groupsize>-1` raises `NotImplementedError` (`args_utils.py:144-145`).
  - KV cache: `k_bits 4 --k_asym`, `v_bits 4 --v_asym`, `--k_groupsize 128 --v_groupsize 128`.
    - The group-size args are **not** read by the HF fake-quant path. Its KV quantizers are built with `groupsize=-1`
      (`qwen_utils.py:126-133`) and quantize over the last dim, which is head_dim. That is per-head = group 128 when
      d = 128.
  - Queries: `q_bits` stays 16.
  - Paper §4.1 matches: per-channel/per-token symmetric W/A, group-wise asymmetric KV with group = head dim.
- **Cost**: paper App. B.1 Table 8 reports 0.9 h for LLaMA-3-8B on one GPU (AMP). FP32 roughly doubles time and uses
  1.28× memory (Table 7).

## Save and reload

**What the repo writes** (all under `exp_dir = output_dir/<model basename>/w{w}a{a}/<exp_name>`, `args_utils.py:153-155`):

1. `flat_parameters.pth`: raw trainable parameters per layer (SVD factors or free matrices, `diag_scale`, clip logits).
   - Written during calibration (`train_utils.py:157-158`). Reloaded with `--resume` (`main.py:30-31`,
     `flat_utils.py:52-62`).
   - `load_state_dict(strict=False)` loads it into freshly wrapped layers, then the normal reparameterize runs.
2. `flat_matrices.pth`: the `--save_matrix` output (`main.py:36-37`, `flat_utils.py:65-78`).
   - Per layer: materialised `*trans.matrix*` (`matrix_left`, `matrix_right`, `matrix_left_inv`, `matrix_right_inv`,
     or `matrix`, `matrix_inv_t`, where `*_inv` = P⁻ᵀ), `*trans.diag_scale`, `clip_factor_w_{max,min}` and
     `clip_factor_a_{max,min}` (including the KV quantizers'). All fp32.
   - Reloaded with `--reload_matrix --matrix_path <dir>` (`main.py:32-33`, `flat_utils.py:81-93`). The loader calls
     `rep_matrix_only()` (switches the transforms to eval mode) and then `load_state_dict(strict=False)`.
   - After a reload, `main.py` runs `reparameterize_model` and RTN/GPTQ again from the **original fp weights**. The
     fp weights are *not* saved; the artifact is "fp model + transforms".
   - The HF hub "Model Zoo" (README "Model Zoo") distributes this format.
3. Packed int4 safetensors from `--quantized_save` (`main.py:49-51`, `flat_utils.py:97-204`):
   - `model*.safetensors` (sharded at 5 GB) + optional `model.safetensors.index.json` + `quantization_config.json`
     (`{w_bits, model_name, symmetric, format:'packed_int4', sharded}`).
   - Weights of every linear in `quantizers` are stored as `pack_i4(round(W/scale))`, a uint8 of shape `[out, in/2]`.
     The export always uses the **symmetric** formula (`sym=True` default argument, `flat_utils.py:97, 125-129`).
   - Every other parameter is cast to **fp16**: norms with folded scales, embeddings, transform matrices and their
     inverses, clip logits, biases already transformed.
   - Per-linear `quantizer.<name>.scale`, `.zero` and `.maxq` are stored in the quantizer's own dtype, which is the
     weight dtype.
   - `model.named_parameters()` is used, so buffers are not saved.
   - REALQUANT.md documents this flow ("Add `--quantized_save` flag ...", REALQUANT.md:150) and reports Real vs Fake
     PPL/accuracy for Llama only.
   - The vLLM fake-quant models (`vllm_custom/.../*_flatquant.py:505-542`) expect a different artifact: an HF
     checkpoint that is already reparameterized and fake-quantized, plus a `flat_matrices.pth` in a per-TP-rank
     layout (`trans_list.{rank}` keys, clip factors indexed by rank). `main.py` does **not** produce that layout. It
     also never calls `save_parametrized_checkpoint` (`flat_utils.py:43-49`).

**Classes that rebuild it**:

- Fake-quant HF path: `FlatQuantQwen2Attention` / `FlatQuantQwen2MLP` (or the Llama equivalents) plus
  `FlatQuantizedLinear` and the `*TransMatrix` classes, reconstructed by `apply_flatquant_to_*`.
- Packed path:
  - Top-level classes: `deploy.transformers.modeling_llama.FlatQuantLlamaForCausalLM.from_pretrained`
    (`modeling_llama.py:346-542`), or `benchmarks/benchmark_lm_eval.py:207-248` (`get_model_quantized` +
    `rename_keys`, `:163-205`).
  - These build `FlatQuantLlamaAttention` / `FlatQuantLlamaMLP` from `deploy.nn.Linear4bit` (`linear.py:22-77`,
    bias supported), `deploy.nn.Quantizer` (`quantization.py:5-36`), `deploy.nn.OnlineTrans(trans="matmul")`
    (`online_trans.py:18-67`) and `deploy.transformers.MultiLayerPagedKVCache4Bit` (`kv_cache.py:170-391`).
  - Keys are renamed, for example `ln_trans.matrix_left` → `self_attn.left_matrix` (re-registered into
    `inp_trans_q/k/v`), `o_trans.matrix` → `o_proj_trans.right_matrix`, `kcache_trans.matrix` → `trans_matrix_k`,
    and `quantizer.*.linear` → `*.weight_scales` (`modeling_llama.py:479-517`).
  - Clip-factor tensors are turned into Python floats (`:532-538`); the kernels apply the sigmoid.
  - **There is no Qwen2 packed model class.**

**Runtime flags that matter**:

- `FlatQuantizedLinear._eval_mode` (`flat_linear.py:26, 69-73, 97`): False means the training forward (re-applies
  `qa_trans` and LWC to the weight on every call). True means the folded weights and only act fake-quant. It is set
  by `reparameterize()`. Loading folded weights into a module with `_eval_mode=False` would apply the transforms
  twice.
- `*TransMatrix._eval_mode` (`trans_utils.py:19, 83, 134, 188`): after `to_eval_mode()`, the module uses the frozen
  `matrix*` Parameters and deletes the trainable factors. That is required before loading `flat_matrices.pth`, which
  is why `rep_matrix_only()` is called first.
- `DecomposeTransMatrix.use_diag` (`trans_utils.py:77, 182`): must be False after `c` is folded into the
  norm/up_proj (`flat_utils.py:27`, `qwen_utils.py:88, 95`). Otherwise `c` is applied twice.
- `FlatQuantQwen2Attention._eval_mode` (`qwen_utils.py:136, 272-277, 302`): selects whether o_proj receives
  `qa_trans` online.
- `_ori_mode` on attention and MLP: fp bypass, used only for targets.
- `ActivationQuantizer.enable` / `WeightQuantizer.enable` (`quant_utils.py:69, 131`).
- `args.separate_vtrans`: changes where `P_v` lives, online vs folded into v_proj.
- Packed path: `config.quantization_config` keys `fuseLN` (default False), `trans` (default `"matmul"`) and
  `online_trans` (default `{qk, o_proj, down_proj, qkv_proj, up_gate_proj}`) (`modeling_llama.py:378-387`).
  `cache_dtype="int4"` (`:358`). `attn_implementation` must be `flash_attention_2` (`:286, 349, 377`).

**Kernels and dtype**:

- `deploy._CUDA`, built by `pip install -e .` (`setup.py:86-106`, sm75/80/86 only, `setup.py:22-27`, cutlass
  submodule). It provides:
  - `matmul`: CUTLASS int4 GEMM; K must be a multiple of 32 (`deploy/__init__.py:37-41`).
  - `sym_quant` / `sym_dequant`: both **assert fp16 inputs and fp16 scales** (`deploy/__init__.py:43-52`).
  - FlashInfer 4-bit and fp16 paged KV `init/append/batch_decode`, **head_dim hard-checked == 128**
    (`deploy/kernels/bindings.cpp:148,192,236,277,321,365`).
- `fast_hadamard_transform` (third-party submodule, installed by `setup.py:45-56`) is imported unconditionally by
  `deploy/functional/online_trans.py:2` and `deploy/transformers/kv_cache.py:7`. It is required even though
  FlatQuant uses `trans="matmul"`.
- Triton ≥ 3.2:
  - `kron_matmul` (fused `P1ᵀXP2` + int4 quantization; pads to the next power of two with masks).
  - `block_matmul` (per-head `X·P_o` + quantization). It uses `tl.arange(0, N)` with N = number of heads, so
    **N must be a power of two** (`block_matmul.py:57-58, 248-249`). REALQUANT.md:157 warns that hidden dim must be
    a power of 2.
- `flash_attn` (`_flash_attention_forward`, `modeling_llama.py:20, 130`) for prefill. Prefill attention uses
  unquantized q/k (REALQUANT.md:155).
- **Scales must be fp16.** The packed loader sets the default dtype to fp16, so `Linear4bit.weight_scales` buffers
  are fp16 and loaded bf16 scales are cast on copy. Activation scales are produced as fp16 (`quantization.py:28,30`).
  The whole packed model runs in fp16, including norms, embeddings and transform matrices
  (`modeling_llama.py:373-374`, `flat_utils.py:135`).
- KV-cache LAC is **not** applied in the packed cache. `asym_quantize_and_pack_i4` is called without `lac=True`
  (`kv_cache.py:288-289`), so the learned KV clip factors are loaded but unused there.

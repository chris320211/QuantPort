# method.md — PrefixQuant (arXiv 2410.05265), repo ChenMnZ/PrefixQuant @ 60b7423

The target is the W4A4KV4 "O2" static setting from the README: `--wbits 4 --input_bits 4 --input_mode static --k_bits 4 --v_bits 4 --kv_group_size 128 --kv_mode static --mse_init --pre_rotate --down_online_had --qk_online_had --set_prefixed_tokens`, with optional `--epochs 20`.

Citations use the form `file:line`, relative to the repo root. Paper citations name the section.

## Steps

1. **Load the fp16 model on CPU and seed everything.** Seed defaults to 2. The CPU placement matters because the rotation helpers write back to `"cpu"`. Paper §5.1. Repo `main.py:163-166`, `main.py:188-191`.

2. **Fuse norms into their readers.** Runs only with `--pre_rotate`. Paper App. B (computational invariance, "Absorbable Rotation"). Repo `main.py:192-193` → `utils/rotation_utils.py:49-89`.
   - Mean-centre the embedding rows (`rotation_utils.py:56-58`). This is inherited from QuaRot. For RMSNorm models it is only approximately function-preserving.
   - Multiply each residual-reading norm's γ into the input columns of its readers (`fuse_ln_linear`, `rotation_utils.py:15-30`):
     - `input_layernorm` → q/k/v (`:68`)
     - `post_attention_layernorm` → gate/up (`:67`)
     - final norm → lm_head (`:82`)
   - Replace **every module of the model's norm class** with a weightless `RMSN(hidden_size)` (`rotation_utils.py:83-89`, `model_utils.py:233-258`). Its eps is hard-coded to 1e-5 and ignores `config.rms_norm_eps`.

3. **Rotate the weights offline.** Paper App. B (R1, R2, plus the weight half of R4). Repo `main.py:194-195` → `rotation_utils.py:276-298`.
   - **R1** is a randomized Hadamard `Q` of size `hidden_size`. Its random signs come from `torch.randint` under the seed (`hadamard_utils.py:93-98`).
     - Residual writers become `Qᵀ·W`, including their biases: o_proj (`:163-177`) and down_proj (`:192-207`).
     - Residual readers become `W·Q`: embeddings (`:146-152`), q/k/v (`:155-160`), gate/up (`:179-190`) and lm_head (`:239-244`).
   - **R2** is a randomized Hadamard `Q_ov` of size `head_dim`, applied per head (`:247-272`). The v_proj output rows of each **KV** head become `Q_ovᵀ·W_v`, and so does the v bias. The o_proj input columns of each **attention** head become `W_o·Q_ov`.
   - **R4 (weight half):** an exact Hadamard of size `intermediate_size` is folded into the down_proj input side (`:203-204` → `hadamard_utils.py:116-150`). `get_hadK` (`hadamard_utils.py:9-63`) needs `n = 2^k·K` with K in {1, 12, 20, 28, 36, 40, 52, 60, 108, 140, 148, 156, 172}.
   - Finally the model is cast with `.half()` (`main.py:195`).

4. **Wrap modules into fake-quant equivalents.** Paper §3, Eq. 1-2. Repo `main.py:198` → `quant_utils.py:96-114`.
   - Every `nn.Linear` becomes `QuantLinear` (`quantize/int_linear_fake.py:11-73`), except names containing `lm_head`.
   - Every `RMSN`/`LlamaRMSNorm` becomes `QuantRMSNorm` (`quantize/quant_norm.py:5-54`), except names containing `model.norm`.
   - `QuantLinear.forward` runs in this order: optional online Hadamard on the input, weight fake-quant, input fake-quant, `F.linear`, output fake-quant (`int_linear_fake.py:41-69`).

5. **Register R4 online on the down_proj input.** Paper App. B ("Online Rotation"). Repo `main.py:200-201` → `quant_utils.py:117-124`. At runtime the down_proj input becomes `x ← had(x)` via fast-hadamard-transform with the K-matrix, in fp16 (`int_linear_fake.py:45-49`). This happens before the input quantizer.

6. **Wrap RoPE for R3 and the K quantizer.** Paper App. B (R3: "output activations of q_proj and k_proj after RoPE"). Repo `main.py:203-210` → `rotation_utils.py:349-360` and `quantize/monkeypatch.py:19-31`.
   - Each attention module gets a copy of its `forward` whose globals map `apply_rotary_pos_emb` to a `QKRotationWrapper` (`rotation_utils.py:315-345`). The wrapper is stored as submodule `self_attn.apply_rotary_pos_emb_qk_rotation_wrapper`.
   - The wrapper first calls the real RoPE and gets q, k in `[b, h, s, d]`.
   - If `qk_online_had` is set, it applies a normalized Walsh-Hadamard over `head_dim` to **both** q and k, in fp32 (`:337-339`). `head_dim` must be a power of 2 (`:325`).
   - It then fake-quantizes K only (`:342-344`). Q is never quantized.
   - Because the returned k is what the attention forward then writes to the cache, new tokens' K is stored in the cache already Hadamard-rotated and fake-quantized.

7. **Find the prefixed (outlier) tokens.** Paper §4.1 (Eq. 3, "Number", "Content") and §4.2 ("What token is added as a prefix"). Repo `main.py:227-242` → `utils/stat_utils.py:230-277`.
   - **Data:** 64 Pile samples × 512 tokens (`main.py:227-234`). Each is a random slice, so position 0 is an arbitrary token and normally not BOS.
   - **Hook:** forward hooks capture the **input of every down_proj** (`stat_utils.py:238-250`). Since the hook fires on `QuantLinear` input, it sees the activation **before** the online Hadamard. At this point quantization is off and R1 has no effect on the MLP intermediate.
   - **Outlier rule:** for each sample and layer, take the token-wise max |x| over channels and compute `R_i = M_i / median(M)`. Tokens with `R_i > η = 64` count as outliers (`stat_utils.py:90-113`; `--outlier_threshold`, `main.py:118`).
   - **Count:** `o = ceil(max over layers of mean over samples of count)` (`stat_utils.py:252`). It asserts `o > 0`.
   - **Content:** collect the token ids of outlier tokens, excluding position 0 (`stat_utils.py:149-182`). Take the top-`o` most frequent ids (`:256-267`). If there is only one candidate and it appears in fewer than 10% of samples, drop it (corner case).
   - **Append the start token last** (`:268-274`): `tokenizer.bos_token`, or `eos_token` if the name contains "qwen".
   - **Result:** prefix = `[top-o non-initial outlier tokens…, BOS]`, so the length is at most `o + 1`. Paper Table "Prefixed tokens": Llama-3-8B and Qwen-2-7B → `[BOS]` (1 token); Llama-2-7B → `. \n [BOS]` (3).
   - The prefix is stored as `model.config.prefixed_tokens` (`main.py:241`).

8. **Prefill the prefix KV once, with the full-precision transformed model.** Paper §4.2, Eq. 4: "computed … using the full-precision model … remain in full precision". Repo `main.py:243-250`.
   - The call is `model(tensor([prefix]), use_cache=True)` and the result is kept as `output.past_key_values`. In transformers 4.40 this is a legacy tuple: one `(K, V)` per layer, each `[1, n_kv_heads, P, head_dim]` in fp16.
   - It runs after steps 2-6 with all quantizers still off/absent. So **K is post-RoPE at positions 0..P-1 and post-R3 Hadamard; V is in the R2 basis; neither is quantized.**
   - It is never recomputed, including after fine-tuning.

9. **Collect activation statistics with the prefix prepended.** Paper §3 ("Dynamic and Static"). Repo `main.py:253-255` → `quant_utils.py:20-93`.
   - Uses the same 64×512 data with the prefix token ids concatenated in front (`:84-88`).
   - Hooks sit on Linear, QuantLinear, RMSN, QuantRMSNorm and QKRotationWrapper (`:79`). Statistics **drop the first P positions** (`:54-58`, `:66-70`).
   - For down_proj, the Hadamard is applied manually so the statistics describe the post-R4 input (`:71-72`).
   - Per-channel abs-max is merged across batches with an EMA of 0.99 (`:34-46`). In practice this is dominated by the first sample.

10. **Create the quantizers with max-based init.** Paper Table 1 (O2: weights per-channel; activations per-tensor static; KV per-head static) and §3 Eq. 1-2. Repo `main.py:260-279` → `quant_utils.py:126-272`; `quantize/quantizer.py:42-163`. All quantizers default to symmetric with no zero point, int4 range [-8, 7], and `scale = 2·max/(2^b-1)` (`quantizer.py:105-113`, `128-140`).
    - **Weights:** every QuantLinear (not lm_head) gets W4 with `w_group_size = -1`, i.e. one scale per output channel (`quant_utils.py:126-144`).
    - **Activations:** A4, per-tensor static (one scalar), `quantized_shape = (1, in)` with group = in (`quant_utils.py:146-193`), placed as follows:
      - The output of each `QuantRMSNorm` is quantized. This single quantizer covers the shared input of q/k/v and of gate/up.
      - The input of every other QuantLinear is quantized. In Llama that means o_proj and down_proj. The skip list for inputs is `q_proj, k_proj, v_proj, up_proj, gate_proj, lm_head` (`:152`).
    - **V cache:** the v_proj output quantizer has group `kv_group_size = 128`, which gives one static scalar per KV head when `head_dim = 128` (`quant_utils.py:197-219`).
    - **K cache:** the `k_quantizer` inside `QKRotationWrapper`, shape `(1, n_kv·head_dim)`, group = head_dim, one static scalar per KV head. It runs after RoPE and after R3 (`quant_utils.py:224-272`, assert at `:230`).
      - `--k_pre_rope` (off by default) would instead quantize the k_proj output (`:232-251`).
    - Not quantized: Q, attention probabilities, lm_head and its input, embeddings, and the prefix KV.

11. **Run the block-wise driver.** Paper §4.3. Repo `main.py:288-315` → `quantize/block_ap.py:120-252`. It runs whenever `--mse_init` or `--epochs > 0` is set.
    - A `Catcher` captures layer-0 inputs together with `attention_mask` and `position_ids`. The capture forward passes the prefix cache, so positions are `[P, P+L)` and the mask is `L×(P+L)` (`block_ap.py:155-197`).
    - Blocks are then processed sequentially.
    - Each block call receives a **fresh** `DynamicCache` built from the fp prefix KV (`model_utils.py:367-374`).
    - fp targets come from the fp block applied to fp inputs (`training_target = fp_input`, `block_ap.py:240-252`).

12. **Grid-search initialization (`mse_init`), per block.** Paper §3 ("Initialization of Quantization Parameters") and §5.1 ("Grid Search Initialization Setting": 8 Pile × 1024; layer-output MSE for per-channel/per-head, block-output MSE for per-tensor). Repo `block_ap.py:255-261` → `quant_utils.py:536-631`.
    - Uses 8 samples of the **quantized-path** inputs. Its targets are the fp block applied to those inputs (`:560-564`), with the prefix KV.
    - **Each quantizer is searched in isolation**, with all other quantizers off (`:572-631`).
    - **Per-tensor static activation scales:** block-output MSE (`block_mse_init_static`, `:450-489`). A coarse pass tries `scale_max × linspace(0.25, 0.95, 8)`, then a fine pass tries `±0.05` around the best with 10 points.
      - Quirk: the fine pass omits `attention_mask` (`:481`), which is a mask bug.
    - **V per-head:** MSE of the V tensor itself, with factors 1.0→0.55 in steps of 0.05 (`tensor_mse_init_static`, `:398-419`, called at `:590-592`).
    - **K per-head:** MSE of `softmax(Q·Kᵀ/√d)` over the GQA query group via `repeat_kv`, using the same 10 factors (`k_cache_mse_init`, `:340-395`, called at `:606-610`). This pass uses no causal mask and no prefix keys.
    - **Weights per-channel:** MSE of the layer output `X·Wᵀ` on captured inputs, with factors 1.0→0.55 (`weight_layer_mse_init`, `:276-337`, called at `:617-630`). q/k use the same path unless `--skip_qk_weight_init` or `--block_qk_weight_init` is set.

13. **Block-wise fine-tuning (only when `--epochs > 0`).** Paper §4.3 ("O2: quantization parameters inherently trainable"; weights trained following EfficientQAT), §5.1 ("Fine-Tuning Setting"), App. C. Repo `block_ap.py:266-308`, `block_ap.py:32-65`, `block_ap.py:94-117`; `quant_utils.py:655-698`.
    - Turn on all quantizers (`:267`) and cast the block to fp32 for AMP (`:271`).
    - Two AdamW parameter groups, both with weight decay 0 (`:274-284`):
      - quantization params: every param whose name contains `scale`, `zero_point` or `bound_factor`, i.e. weight, activation and K/V scales; lr `quant_lr = 5e-5`
      - weights: every param whose name contains `weight` but not `weight_quantizer`; lr `weight_lr = 5e-6`. `QuantRMSNorm.weight` and `QuantLinear.bias` are buffers, so they are not trained.
    - Both learning rates follow a cosine schedule down to lr/10 (`min_lr_factor`).
    - Loss is MSE between the quantized block's output (on quantized inputs, with a fresh prefix cache every step) and the fp target (`:32-65`).
    - Rounding uses a straight-through estimator (`quantizer.py:10-20`).
    - `--clip_grad 0.3` is parsed but **never passed** (`block_ap.py:60`), so there is no clipping.
    - Validation runs every epoch on 64 samples. Early stopping is off by default.

14. **Fold weight quantization in place and propagate.** Paper §4.3. Repo `block_ap.py:310-320`, `quant_utils.py:732-737`.
    - The block goes back to `.half()`, then `weight ← weight_quantizer(weight)`.
    - From then on weight quantization is off and activation quantization is on.
    - The block's quantized outputs become the next block's quantized inputs (`:314-317`).

15. **Save.** Repo `main.py:317-327`. Details are in Save and reload below.

16. **Inference with the prefix.** Paper §4.2 (stored KV "reused during inference by quantized models"). Repo `main.py:20-69`; `utils/data_utils.py:210-243`; `utils/model_utils.py:340-415`; `eval.py:52-107`.
    - **The prefix lives only in the KV cache, but it occupies positions 0..P-1.** No prefix tokens are re-fed. The model derives `cache_position`/`position_ids = [P, P+L)` from the cache length (transformers 4.40 LlamaModel), and the causal mask spans `P+L` keys.
    - **Perplexity:** `test_ppl` passes the same legacy tuple as `past_key_values` for every 2048-token window (`data_utils.py:235`). In 4.40 each forward builds a new `DynamicCache` from the tuple, so the source is never mutated. Loss covers only the 2048 window tokens.
    - **lm_eval:** wraps the model in `WrappedPrefixCausalLM` (`model_utils.py:392-415`). Whenever `past_key_values is None`, it injects the prefix repeated to the batch size (`kv_cache_repeat`, `:377-389`). Per-layer device placement uses `mv_kv_cache` (`:340-364`).
    - **`generate()` is not implemented in the repo.**
    - **What any loader or eval must do:**
      - Every forward that starts a new sequence (each PPL window, each lm_eval batch, each `generate` call) must start from a **fresh copy** of the stored fp16 prefix KV, batch-expanded. Never reuse a cache that `update()` has mutated.
      - Real tokens must take positions starting at P, and any attention mask must cover the P prefix columns (ones).
      - For `generate`, the robust HF pattern is: `input_ids = prefix_ids + prompt_ids`, `past_key_values = deepcopy(prefix_cache)`, mask of ones. generate then skips the cached tokens via `cache_position`; strip the prefix from the output.
      - Do not add a second BOS/sink token.
      - Never recompute the prefix KV through the quantized model.
      - Without the prefix, the static per-tensor A4 scales are invalid: paper Table 5 reports static without prefix at PPL 141 vs 7.93 with it.

## Tensor roles

| Role | Steps touching it | Treatment |
|---|---|---|
| Residual stream | 2, 3 | Lives in the R1 basis (`h·Q`). Never quantized directly; quantized only after a norm (see the residual-reading norms row). |
| Token embeddings | 2, 3 | Mean-centred, then `W_E·Q` (output side of the residual). fp16, not quantized. |
| LM head | 2, 3, 4 | Final-norm γ fused in, then `W·Q` (input side). fp16; excluded from weight and activation quantization (`quant_utils.py:102`, `:152`). |
| Residual-reading norms (pre-attention, pre-MLP) | 2, 4, 10, 12, 13 | γ fused into readers, replaced by weightless RMSN (eps 1e-5). The **output is A4 per-tensor static** and this one quantizer serves all q/k/v (or gate/up) readers. |
| Final norm | 2 | γ fused into lm_head, weightless RMSN, not quantized. |
| Attention input projections (q, k, v) | 2, 3, 10, 12-14 | Input side `·Q` (R1) with γ fused in. W4 per-output-channel symmetric. Their input is quantized by the norm. |
| Value/output pair | 3, 10 | v_proj output rows `Q_ovᵀ·` per KV head (R2); o_proj input columns `·Q_ov` per attention head. |
| Attention output projection (o_proj) | 3, 10, 12-14 | Output side `Qᵀ·` (R1) and input side per-head `·Q_ov` (R2). W4 per-channel; **input A4 per-tensor static**. |
| MLP input projections (gate, up) | 2, 3, 10, 12-14 | Input side `·Q` with γ fused in. W4 per-channel; input quantized by the norm. |
| MLP output projection (down_proj) | 3, 5, 10, 12-14 | Output side `Qᵀ·` (R1); input side gets the exact Hadamard folded in plus the online Hadamard (R4). W4 per-channel; **input A4 per-tensor static, after R4**. |
| Linear biases | 3 | Writer biases (o/down) are rotated `Qᵀ·b`; v bias gets `Q_ovᵀ`; reader biases untouched. Stored as fp buffers, not trained, not quantized. |
| Q activations | 6 | Post-RoPE, then R3 Hadamard online. **Not quantized.** |
| K cache (new tokens) | 6, 10, 12, 13 | Post-RoPE, then R3 Hadamard, then **4-bit per-KV-head static symmetric scale** (group = head_dim), stored fake-quant. |
| V cache (new tokens) | 3, 10, 12, 13 | v_proj output in the R2 basis, **4-bit per-KV-head static** (group 128). |
| Prefix KV (P tokens/layer) | 8, 11-16 | Computed once by the fp transformed model (K post-RoPE at positions 0..P-1, post-R3; V in R2). **Kept fp16, never quantized, never retrained.** |
| Attention probs, softmax, RoPE, SiLU, residual adds | — | fp16. |
| Quantizer parameters | 10, 12, 13 | Weight scales `[out,1]`; activation scales `[1,1]`; K/V scales `[n_kv_heads,1]`. Trainable in step 13. |

## Invariants

- **R1 cancellation.** `Q` is orthogonal. Every residual **writer** (embeddings, o_proj, down_proj, plus their biases) is left-multiplied by `Qᵀ` on its output side, and every **reader** (q/k/v, gate/up, lm_head) is right-multiplied by `Q` on its input side. A tensor that writes to or reads from the residual stream and misses this breaks equivalence.
- **Norms are fused before rotation.** An elementwise γ does not commute with `Q`, so γ must first be moved into the readers' input columns and the norm made weightless. RMSNorm without γ is invariant under orthogonal Q (`‖hQ‖ = ‖h‖`).
  - Only **residual-stream** norms may be fused or replaced. Any other norm (for example a per-head norm) must be left alone. The repo replaces norms by class (`rotation_utils.py:83-89`), which is unsafe when one class is used for both kinds.
- **R2 cancellation.** For every attention head, `W_o[:, head]·Q_ov` must pair with `Q_ovᵀ·W_v[kv(head)]`. Under GQA, the same `Q_ov` serves every query head that shares a KV head. This holds because attention is linear in V per head.
- **R3.** The same orthonormal Hadamard is applied to q and k **after RoPE** (and after anything else applied to q/k beforehand), so `qᵀk` is preserved. It cannot be folded into weights because RoPE sits in between. The prefix K must be stored in the same R3 basis, and it is, because it is computed with the wrapper active.
- **R4.** `W_down ← W_down·Mᵀ` offline and `x ← M·x` online, using the same transform M (`get_hadK(intermediate_size)` plus fast WHT). Dropping either half breaks the model.
- **One run, one rotation.** Rotation signs are random under the seed. Weights, quantizer scales and prefix KV are only valid together from the same run.
- **The prefix is part of the model.** Calibration statistics exclude prefix positions, every scale was searched or trained with the prefix KV present, and the prefix shifts positions by P. Inference must therefore reproduce: fp16 prefix KV in every layer, real tokens at positions ≥ P, and a mask covering the prefix.
- **Ordering inside QuantLinear:** online Hadamard, then weight quant, then input quant, then matmul, then output quant (`int_linear_fake.py:41-69`). K quantization happens before the cache write.
- **Weights are on the grid after step 14.** Re-applying the weight quantizer at reload is idempotent.
- **Sanity check:** the rotated fp model (before quantizers) should match the original PPL closely. The only deliberate deviation is the embedding mean-centering.

## Calibration

| Stage | Data | Shape | What is computed / optimized | Iterations |
|---|---|---|---|---|
| Prefix detection | Pile val (`mit-han-lab/pile-val-backup`, `data_utils.py:12-45`), seed 2 | 64 × 512 | Token-max/median ratio of the down_proj input, η=64; o = ceil(max over layers of mean count); top-o frequent non-initial tokens, then BOS | 1 pass (paper App. C: 0.2 min on Llama-3-8B) |
| Prefix KV | the P prefix ids | 1 × P | fp forward | 1 |
| Activation stats | same 64 × 512, with prefix prepended | — | per-channel abs-max (EMA 0.99), prefix positions excluded → max-init scales | 1 pass |
| Grid-search init (`--mse_init`) | 8 Pile samples (`--mse_init_size`) | × 1024 (`--training_seqlen`) | Per-tensor activations: block-output MSE, 8 coarse + 10 fine factors. K: softmax-score MSE, 10 factors. V: tensor MSE, 10 factors. Weights: layer-output MSE, 10 factors. | per block (App. C: 0.7 min on Llama-3-8B) |
| Fine-tuning (`--epochs`) | 512 train + 64 val Pile | × 1024; batch 4 | Block-output MSE vs fp target; AdamW; quant lr 5e-5, weight lr 5e-6, cosine to /10, wd 0, AMP + GradScaler | epochs × 512/4 steps per block |

- Dataloaders are cached at `{cache_dir}/dataloader_{model_name}_{calib}_{train}_{val}_{seqlen}_*.cache` (`main.py:297-314`).
- **What `--epochs 20` costs:**
  - **Steps:** 2,560 AdamW steps per block (`block_ap.py:268`). Each step is a forward and backward of one block in fp32 with AMP, on 4 × 1024 tokens plus the prefix, followed by a validation pass every epoch. For 36 blocks that is about 92k steps.
  - **Time:** the paper reports **2.2 h for Llama-3-8B** on one A100-80GB (App. C, Table 6), and 17 h for 70B. My estimate for Qwen3-8B is about 2-2.5 h on an A100: 36 layers with a smaller MLP than Llama-3's 14336. On an RTX 3090 I would expect roughly 2-3× that.
  - **CPU RAM:** the four `BlockTrainDataset`s at hidden 4096 in fp16 take about 9 GiB (2 × 4 GiB train + 2 × 0.5 GiB val). `--off_load_to_disk` moves them to disk.
  - **GPU memory:** one Qwen3-8B block is about 193M params. fp32 weights, gradients and Adam states come to about 3 GB, plus activations, which fits in 24 GB.
  - **Gain:** the paper's ablation (Table 5) shows W4A4KV4 O2 going from 7.93 to 7.41 Wiki PPL on Llama-3-8B.

## Save and reload

**What `--save_quant_dir` writes** (`main.py:317-327`; the whole model is cast to `.half()` first):

- `model.save_pretrained` writes safetensors plus `config.json`. The config is the original architecture plus an extra `prefixed_tokens` key. The state dict holds:
  - **fake-quantized fp16 weights**: rotated and fused, then replaced by `weight_quantizer(weight)` in step 14. If neither `--mse_init` nor `--epochs` ran, the weights are rotated but unquantized.
  - norm weights, all ones after fusion
  - mean-centred, rotated embeddings, and the rotated, final-norm-fused lm_head, both fp16
  - QuantLinear bias buffers
  - every quantizer's `scale` (fp16), under keys like `…input_layernorm.output_quantizer.scale`, `…o_proj.input_quantizer.scale`, `…v_proj.output_quantizer.scale`, `…self_attn.apply_rotary_pos_emb_qk_rotation_wrapper.k_quantizer.scale` and `…weight_quantizer.scale`
- **Not saved:** the Hadamard matrices. `had_K`/`K` are plain attributes, and R1/R2 are baked into the weights.
- `tokenizer.save_pretrained`.
- `prefixed_key_values.pth`: a `torch.save` of the legacy tuple of per-layer `(K, V)` fp16 tensors. They stay on whatever CUDA devices they were dispatched to, and `eval.py:55` loads them without `map_location`.
- `prefixequant_config.json` (`quant_utils.py:811-831` plus `prefixed_tokens`):
  - bits, group sizes, asym flags, modes and `k_pre_rope`
  - `down_online_had` and `qk_online_had` (each AND `pre_rotate`)
  - `set_prefixed_tokens`, `activation_clipping`, `real_quant`

**Reload (`eval.py:52-107`):**

1. Read the JSON and `torch.load` the prefix KV.
2. Build the stock HF model under `init_empty_weights`.
3. Call `wrap_to_quant_model`.
4. If `down_online_had`, call `register_online_had`. This is **required**, because the down_proj weights already contain the Hadamard.
5. Re-install `QKRotationWrapper` with `online_had = qk_online_had`.
6. Create placeholder quantizers with `minmax_init=False`, with the same shapes as at save time. Note that `init_weight_quantizer` turns `use_weight_quant` on again, which is idempotent.
7. `load_checkpoint_in_model`, then `.half()`, then `evaluate`: dispatch, `mv_kv_cache`, `test_ppl` with the prefix tuple, and lm_eval through `WrappedPrefixCausalLM`.

No `--pre_rotate` is needed at reload. `--real_quant` is parsed but unused in `eval.py`.

**Runtime needs at reload:**

- `fast_hadamard_transform` (CUDA), for R3 and R4.
- transformers 4.40 semantics for legacy tuple caches (`requirements.txt`: `transformers==4.40.1`, `accelerate==0.28.0`, `lm_eval==0.4.2`).

**Real quantization: there is no real low-bit kernel.**

- `quantize/int_linear_real.py` is a GPTQ-style packer: int32-packed `qweight`, `qzeros`, fp16 `scales` (`:97-153`).
- Its `forward` dequantizes the whole weight with Triton kernels and then calls `torch.matmul` in fp16 (`:155-167`). `quantize/triton_utils/kernels.py` contains only `dequant_kernel_dim0/1` and `silu`, with no GEMM.
- `--real_quant` is asserted to be weight-only (`block_ap.py:324`).
- It also needs `zero_point`, which is `None` for the default symmetric weights, so it crashes (`block_ap.py:328`).
- `load_quantized_model` via `--resume_quant` references the nonexistent `args.group_size` (`main.py:185`).
- The README states that inference kernels are unreleased (`README.md:13`), so the paper's W4A4 speedups (Fig. 6) cannot be reproduced from this repo. **For W4A4KV4 the only artifact is fake-quant fp16 plus scales.**

## Porting notes for Qwen3

### Generic building blocks (reusable as-is)
- **Quantization primitives:**
  - `UniformAffineQuantizer` with STE and static/dynamic modes (`quantize/quantizer.py:10-274`)
  - fake `QuantLinear` (`quantize/int_linear_fake.py`)
  - `QuantRMSNorm` math (`quantize/quant_norm.py`)
  - the recon losses (`quantize/recon_loss.py`)
- **Hadamard and rotation utilities:**
  - `hadamard_utils.py` in full: `get_hadK`, `matmul_hadU(_cuda)`, `random_hadamard_matrix`, `apply_exact_had_to_linear`
  - the math of `rotation_utils.fuse_ln_linear`, `random_orthogonal_matrix` and `QKRotationWrapper.forward`
  - the patching mechanism in `quantize/monkeypatch.py`
- **Calibration and init:**
  - the MSE-init math in `quant_utils.py`: `weight_layer_mse_init`, `tensor_mse_init_static`, `k_cache_mse_init`
  - the parameter selectors and `quant_inplace` (`quant_utils.py:655-737`)
- **Training:** the block-AP loop logic (`block_ap.py:22-117`, `:232-320`), the `BlockTrainDataset`, and the loaders.
- **Weight-only packing:** `int_linear_real` and the Triton dequant kernels. These are irrelevant for W4A4.

### Llama / transformers-4.40 glue (each must be generalized)
- **Model and module identity:**
  - Model classes are hard-coded to OPT/Llama/Mistral/Qwen2/InternLM2, with a `ValueError` otherwise (`model_utils.py:13-47`, `:108-132`, `:137-199`).
  - The RoPE function name is set only for Llama/Mistral/Qwen2 (`model_utils.py:53-56`).
- **Norm handling:**
  - `RMSN` eps is fixed at 1e-5 (`model_utils.py:240`; `rotation_utils.py:87`).
  - Norm replacement is by class (`rotation_utils.py:83-89`).
- **Hard-coded module names:**
  - `input_layernorm/post_attention_layernorm/q,k,v,o_proj/up,gate,down_proj` (`rotation_utils.py:66-68`, `:157`, `:165-166`, `:181-182`, `:194-195`, `:249-250`)
  - `lm_head`, `model.norm` and `down_proj` matched by name (`quant_utils.py:102-106`, `:119`, `:152`, `:200`, `:234`, `:590`, `:619`)
  - `model.layers.{i}.mlp.down_proj` (`stat_utils.py:24-32`, `:100-104`, `:158-161`)
- **`head_dim = hidden/num_heads`** (`rotation_utils.py:283`, `:324`; `quant_utils.py:25`, `:228`).
- **Imports from the Llama module:** `LlamaRMSNorm` and `repeat_kv` (`quant_utils.py:8`, `:16`), with class-based hooking and wrapping (`quant_utils.py:79`, `:111`).
- **Prefix-token selection by model name:**
  - The family comes from `model_name.split('-')[0]` (`stat_utils.py:193-227`, `:232`).
  - The start token is BOS, or EOS when the name contains "qwen" (`stat_utils.py:268-273`).
- **Block-AP model plumbing:**
  - Hard-coded paths `model.model.layers/embed_tokens/norm/rotary_emb` (`block_ap.py:138-144`, `:200-205`; `main.py:22`, `:222`).
  - The `Catcher` keeps only `attention_mask`/`position_ids` (`block_ap.py:164-171`).
- **4.40 calling conventions:**
  - Block calls use `layer(h, attention_mask=, position_ids=, past_key_value=)[0]`, which assumes the layer computes RoPE itself, the kwarg is named `past_key_value`, and the return is a tuple (`block_ap.py:29`, `:41`, `:76`; `quant_utils.py:467`, `:481`, `:509`, `:524`, `:564`).
  - Legacy-tuple caches: `DynamicCache.from_legacy_cache` (`model_utils.py:10`, `:371`), `kv_cache_repeat` producing a dict, `WrappedPrefixCausalLM` (`model_utils.py:377-415`), a tuple `past_key_values` in `test_ppl` (`data_utils.py:235`), and `output.past_key_values` saved as a tuple (`main.py:249`, `:323`).

### Qwen3-8B interactions (qwen3, transformers ≥ 4.51; 36 layers; hidden 4096; 32 Q / 8 KV heads; head_dim 128; intermediate 12288; untied; no attention bias; per-head q_norm/k_norm before RoPE)

1. **q_norm/k_norm collide with norm handling by class.**
   - **Why it breaks:** these norms share the `Qwen3RMSNorm` class with `input_layernorm` and `post_attention_layernorm`.
   - **Repo as written:** an unknown class means `get_model_type` raises.
   - **Naive "add Qwen3RMSNorm to the class list" fix:** `replace_modules` would swap q_norm/k_norm for `RMSN(4096)` with weight ones. That throws away their learned γ and gives a 4096-wide weight against 128-wide inputs, which is a broadcast error.
   - **Naive `wrap_to_quant_model` fix:** it would wrap q_norm/k_norm into `QuantRMSNorm`, and `init_input_quantizer` would then attach **A4 per-tensor output quantizers to pre-RoPE q and k**. That is wrong.
   - **Required:** select residual-stream norms by role (name), never by class. Leave q_norm/k_norm untouched: not fused, not replaced, not quantized.
   - They need no rotation fix, because R1 leaves the q_proj/k_proj outputs unchanged.
   - Decide explicitly whether to freeze them in block-AP. As plain modules, their `weight` parameter matches `set_weight_parameters` (`quant_utils.py:655-669`) and would be trained at 5e-6. The repo freezes norm γ by making it a buffer.
2. **Reload silently drops activation quantizers.** `eval.py`'s `wrap_to_quant_model` only wraps `RMSN`/`LlamaRMSNorm`. On reload the Qwen3 norms come back as `Qwen3RMSNorm`, so the q/k/v and gate/up input quantizers would be missing. Depending on accelerate's handling of unexpected keys, this either silently evaluates without input quantization or errors. A port's loader must recreate exactly the saved quantizer set.
3. **R3 and K-quant placement are compatible with Qwen3; `k_pre_rope` is not.**
   - Qwen3's attention order is: `q_proj → view → q_norm → transpose → apply_rotary_pos_emb → (cache.update) → attention`.
   - `apply_rotary_pos_emb` is a module global in `modeling_qwen3`, so the monkeypatch works once the RoPE-name lookup returns it. The Hadamard and K-quant then sit after q_norm/k_norm and after RoPE, before the cache write, which is correct.
   - **Never use `--k_pre_rope`.** It would quantize the k_proj output before k_norm and RoPE.
   - **Never fold R3 into weights:** both k_norm and RoPE intervene.
   - Because k_norm already normalizes each head per token, the paper's tiny-magnitude "lower outlier" K tokens (§4.1) may look different. Re-measure the K/Q token ratios after prefixing (this is a hypothesis, not established).
4. **head_dim must come from `config.head_dim`.** For 8B the formula `hidden/heads = 4096/32 = 128` happens to be right, but it is wrong for other Qwen3 sizes (for example 0.6B, where hidden 1024 / 16 heads ≠ head_dim 128). The value 128 is a power of 2 and equals `kv_group_size = 128`, which satisfies the asserts at `quant_utils.py:230` and `rotation_utils.py:325`.
5. **GQA (32 Q / 8 KV) is already handled.**
   - `rotate_ov_proj` reshapes v by KV heads and o by Q heads.
   - `k_cache_mse_init` uses `num_key_value_groups` and `repeat_kv`.
   - The V and K static scales are `[8, 1]`.
   - The prefix KV is `[1, 8, P, 128]` per layer.
6. **R4 works for intermediate 12288.** 12288 = 12 × 1024, so `get_hadK` returns K=12 (had12 plus a 1024-point fast WHT).
7. **Embeddings and biases need no extra handling.**
   - The embeddings are untied, so `rotate_embeddings` and `rotate_head` act on separate tensors.
   - If a model were tied, the repo would mean-centre, γ-scale and double-rotate the shared matrix, because nothing unties it. A port must untie first.
   - There is no attention bias, so the bias-rotation paths are unused.
8. **eps mismatch.** Qwen3's `rms_norm_eps` is 1e-6, but `RMSN` uses 1e-5 at quantization time while the reload uses the config's 1e-6. Pass the config eps.
9. **Prefix-token selection by name breaks.**
   - `model_name = "Qwen3-8B"` gives family `qwen3`, which hits `NotImplementedError` at `stat_utils.py:216-217`.
   - The "qwen" substring check picks `eos_token` as the start token. For the Qwen3-8B tokenizer that is `<|im_end|>`; for a Base tokenizer it is `<|endoftext|>`. Qwen3 has no `bos_token`, so the non-qwen branch would crash.
   - The paper uses "[BOS]" for Qwen-2-7B, a 1-token prefix.
   - **Port:** choose the start token generically: BOS if present, else a document-start token. Log the choice and verify that the max top-1/median ratio collapses after prefixing.
   - `assert outlier_num > 0` (`stat_utils.py:253`) will fire if Qwen3 has no η=64 outliers in the down_proj input.
10. **Newer transformers attention and cache API (≥ 4.51, pinned repo is 4.40.1).**
    - **position_embeddings:** decoder layers need `position_embeddings=(cos, sin)` from `model.model.rotary_emb`. The `Catcher` must capture it, and every block call must pass it: `update_dataset`, train/eval, `mse_init`, and the `block_mse_init_*` functions. Positions must be `[P, P+L)`.
    - **Cache kwarg name:** `past_key_value` was renamed `past_key_values` in later 4.5x releases.
    - **Layer return type:** layers return a tuple in 4.51-4.53 but a bare tensor in later versions, where `[0]` would silently take batch item 0. Check the return type.
    - **Legacy caches:** tuple caches are not accepted by `Qwen3Model`, and `from_legacy_cache` is deprecated. Build a `DynamicCache` per forward by calling `update(k, v, layer_idx)` for each layer from saved tensors, batch-expanded. Because `DynamicCache` mutates in place, use a fresh or deep-copied cache for every PPL window, lm_eval batch, `generate` call and training step.
    - **Saving the prefix:** `model(...).past_key_values` (`main.py:249`) is now a `DynamicCache` object. Save per-layer `(K, V)` tensors instead, with `map_location` on load.
    - **Masks:** with a prefix, `q_len ≠ kv_len`. Masks must be bottom-right aligned so the prefix is fully visible. SDPA's `is_causal` with `q ≠ kv` is top-left aligned, which is wrong. Pass a materialized mask, or use flash-attn's bottom-right causal. Also fix the missing mask at `quant_utils.py:481` (a repo bug).
    - **Mask format:** new masking utilities may return `None` or a different mask format from what the `Catcher` saw. Verify that the mask used in block calls includes the P prefix columns.
11. **Save format.** Since there is no real int4 kernel for W4A4, the deliverable is fake-quant fp16 weights plus scales, `prefixed_key_values`, and the config. The loader must rebuild:
    - the R4 online Hadamard (K=12)
    - the R3 wrapper on the Qwen3 attention
    - the quantizers on exactly the residual norms, o_proj, down_proj, v_proj output and the K wrapper
    - fresh prefix-cache injection on every forward, with positions starting at P

    lm_head and the final norm stay fp16; `model.norm` matches the repo's name-based skip.

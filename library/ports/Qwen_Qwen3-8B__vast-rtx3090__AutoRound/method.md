# AutoRound / SignRound — method model

Paper: arXiv 2309.05516, "Optimize Weight Rounding via Signed Gradient Descent for the
Quantization of LLMs" (`.cache/papers/2309.05516.txt`). Repo: intel/auto-round at
`34d16e916f6d459f685033b889fd078117edf22d` (installed as `auto_round` 0.17.0;
paths below are relative to `.venvs/autoround-qwen3-8b-vastrtx3090/repo`).

AutoRound is weight-only post-training quantization. Each transformer block is tuned
on its own: for every linear in the block it learns a per-element rounding offset `V`
and per-group clipping coefficients `alpha`/`beta` (repo: `max_scale`/`min_scale`) by
minimizing the MSE between the block output with fake-quantized weights and the
full-precision block output, using signed gradient descent (SignSGD). No weights are
rotated, smoothed or re-parameterized; the only thing that changes is which integer
each weight rounds to and the per-group scale (and zero point if asym). The result is a
standard GPTQ-layout int4 checkpoint.

## Steps

1. **Resolve scheme and recipe.** `AutoRound(...)` (`auto_round/autoround.py:134`) resolves
   `scheme="W4A16"` to `bits=4, group_size=128, sym=True, data_type="int", act_bits=16`
   (`auto_round/schemes.py:624-632`) and the default recipe `"auto-round"` to a single
   `SignRoundConfig(iters=200)` stage with calibration defaults `batch_size=8, seqlen=2048,
   nsamples=128` (`auto_round/recipes/recipe.py:20-21,90-96`). `lr=None` resolves to
   `1.0/iters` (= 5e-3 at 200 iters) and `minmax_lr` falls back to `lr`
   (`auto_round/algorithms/quantization/sign_round/config.py:424-469`). Paper §4.1: 200
   steps, lr 5e-3, linear decay, bs 8.
2. **Select layers.** All `torch.nn.Linear`/`Conv1D` inside the detected decoder blocks
   are quantized (`SUPPORTED_LAYER_TYPES`, `auto_round/utils/common.py:738`;
   `wrapper_block`, `auto_round/wrapper.py:1063-1105`). `lm_head` is dropped from the
   layer config unless `quant_lm_head=True` (default False,
   `auto_round/compressors/layer_config_resolver.py:49-68`); embeddings are only
   collected for GGUF/explicit layer_config (`layer_config_resolver.py:343-351`,
   `algorithms/quantization/base.py:134`). Paper §4.1: "Layers such as the embedding
   layer and typically the last linear layer like 'lm-head' are excluded".
3. **Build calibration inputs.** Tokenize `nsamples` texts of `seqlen` tokens from the
   default dataset `NeelNanda/pile-10k` (`auto_round/compressors/base.py:279`,
   `auto_round/calib_dataset.py:364-410`), run the model up to the first block and cache
   the block inputs plus positional kwargs (attention mask, position ids / RoPE
   embeddings) (`auto_round/compressors/orchestrator.py:829-1110`). Paper Alg. 1 line 4.
4. **Per block, collect the FP reference output** `y_f = block(x_fp)` with original
   weights (`auto_round/algorithms/composer.py:537,654`). Paper Alg. 1 line 5.
5. **Wrap each linear** in `WrapperLinear` (`auto_round/wrapper.py:71`): per-group
   `weight_min = min(W_g, 0)`, `weight_max = max(W_g, 0)` (`wrapper.py:187-198`);
   rounding offset `value` (= paper `V`) initialized to 0, fp32, shape of the
   group-reshaped weight (`wrapper.py:279-281`); `min_scale`/`max_scale` (= paper
   `beta`/`alpha`) initialized to 1.0, one per group (`wrapper.py:283-288`), clamped
   to `minmax_scale_bound=(0,1)` every forward (`wrapper.py:85,562-575`). Paper §3
   Eq. 3-4, Alg. 1 line 1.
6. **Fake-quantize** the weight in the wrapper forward (`wrapper.py:577-616,800-811`):
   for `sym=True` the repo uses *full-range* symmetric int
   (`auto_round/data_type/int.py:164-237`): `max_v = ±max(|wmin·min_scale|, wmax·max_scale)`
   (sign chosen so the larger-magnitude side maps to -8), `scale = max_v/8`,
   `q = clamp(round(W/scale + V), -8, 7)`, `W~ = scale·q`; rounding uses a
   straight-through estimator (`round_ste`, `auto_round/data_type/utils.py:412`).
   For `sym=False`: `scale = (wmax·max_scale − wmin·min_scale)/15`,
   `zp = round(−wmin/scale)`, `q = clamp(round(W/scale + V) + zp, 0, 15)`
   (`int.py:240-322`) — this is paper Eq. 2-4 exactly. Paper Alg. 1 line 6.
7. **Block-wise output MSE loss.** `y_q = block_q(x_q)` where `x_q` is the output of the
   *already quantized* previous blocks (`enable_quanted_input=True`,
   `quantizer.py:384`, `orchestrator.py:1210-1240`); `loss = MSE(y_q, y_f)` in fp32
   over a random batch of `batch_size` samples per step, with padding masked out when
   input ids are available (`auto_round/algorithms/quantization/sign_round/quantizer.py:232-263,629-686`).
   Paper §3 Eq. 5, Alg. 1 lines 7-8 (the paper feeds the same `x` to both; the repo
   feeds quantized inputs to the student, a deliberate error-propagation choice).
8. **SignSGD update.** Optimizer is fixed to `SignSGD` (`quantizer.py:996-1009`) with two
   param groups (rounding `V` at `lr`, min/max at `minmax_lr`; `quantizer.py:402-451`),
   `p ← p − lr_t·sign(g)` (`sign_round/sign_sgd.py:388`), linear LR decay 1→0 over
   `iters` (`quantizer.py:453-456`), `iters` steps (`quantizer.py:591`). The total
   movement per element is bounded by Σlr_t ≈ 0.5, i.e. V ∈ [−0.5, 0.5] and
   alpha/beta ∈ [0.5, 1] without explicit clamping of V. Paper §3 Eq. 6 and the
   paragraph after it; Alg. 1 line 14.
9. **Keep the best.** Track the lowest-loss iteration and snapshot its `V/min_scale/max_scale`
   (`quantizer.py:707-722`); unless `not_use_best_mse`. Paper Alg. 1 lines 10-12.
10. **Unwrap / bake.** Re-quantize each linear with the best params and store
    `weight = qdq(W)`, `scale`, `zp` on the original module (`wrapper.py:671-695`).
    For full-range sym the returned `zp` is the constant `maxq=8`.
11. **Pack and save** (repo-only, not in paper): `format="auto_round"` with a sym int
    scheme selects GPTQ-style packing (`auto_round/export/formats/autoround/format.py:66-67`,
    `integer.py:357-371`); each linear becomes a `QuantLinear` with `qweight`
    int32 `[in/8, out]`, `qzeros` int32 `[in/128, out/8]` (stored `zp−1`, GPTQ v1),
    `scales` fp16 `[in/128, out]`, trivial `g_idx` `[in]`
    (`auto_round/export/formats/gptq/packing.py:47-101`,
    `auto_round_extension/torch/qlinear_torch_zp.py:63-90,307-314`).

Generic building blocks: `WrapperLinear`/`wrapper_block`/`unwrapper_block`
(`wrapper.py`), int quantizers `quant_tensor_sym/asym` (`data_type/int.py`), `SignSGD`
(`sign_round/sign_sgd.py`), `SignRoundQuantizer.quantize_block`
(`sign_round/quantizer.py:327`), GPTQ/AWQ packers (`export/formats/gptq`, `awq`),
`QuantLinear` classes (`auto_round_extension/`), inference conversion
(`auto_round/inference/convert_model.py`, `backend.py`). Model-family glue: block-name
detection (`utils/model.py:1566 get_block_names`), `special_model_handler.py`
(multimodal/omni/TTS forwards only), rotation transforms' per-model mappings
(`algorithms/transforms/hadamard/inplace/model_config.py:141` registers `qwen3`, unused
for plain SignRound). There is **no Qwen3-dense-specific code** in the quantization path;
Qwen3 is handled by the generic "decoder blocks of Linears" walker. The repo's own
pipeline entry point is `AutoRound(...)` → `BaseOrchestrator.quantize_and_save`
(`auto_round/compressors/base.py:2188`) → `quantize()` (`orchestrator.py:657`) →
`_quantize_data_driven` (`orchestrator.py:829`) → `_quantize_blocks`
(`orchestrator.py:317`) → `AlgComposer.compress_block` (`composer.py:537`) →
`SignRoundQuantizer.quantize_block`.

## Tensor roles

- **Residual stream / block input-output activations**: cached as calibration inputs
  (Step 3), used as FP reference (Step 4) and as the quantized-student input (Step 7).
  Never quantized (A16); never transformed.
- **Attention input projections (Q, K, V)**: wrapped, `V`+alpha/beta tuned jointly with
  the rest of the block, quantized to int4 g128 sym along the input dim, packed GPTQ.
  Each projection has its own groups/scales; serving engines fuse them, which requires
  all three to share bits/group_size/sym (they do, single scheme).
- **Attention output projection (O)**: same as above.
- **MLP input projections (gate, up)**: same as above; fused at serving time.
- **MLP output projection (down)**: same as above (in_features = intermediate size).
- **Norm weights inside blocks** (pre-attention, pre-MLP RMSNorm, and any extra norms):
  untouched, kept in model dtype; only tuned if `enable_norm_bias_tuning=True`
  (default False, `sign_round/config.py:239`), which we do not use.
- **Linear biases**: kept fp; untouched unless norm/bias tuning is on.
- **Embeddings**: kept in model dtype (not a Linear; not in layer config for non-GGUF).
- **LM head**: kept in model dtype by default (`quant_lm_head=False`); paper §4.1
  also excludes it. Untied heads could be quantized via `quant_lm_head=True` (tuned
  outside blocks, `quantizer.py:769`), not used here.
- **Final norm**: kept in model dtype.
- **KV cache / attention activations**: not quantized (A16; static KV options are
  experimental and off).

## Invariants

- The model function changes **only** through `W → W~ = qdq(W; V, alpha, beta)` of
  in-block linears. There is no rotation, smoothing or scale migration, so no tensor
  outside a linear's weight needs a compensating transform; norms, biases, embeddings,
  LM head, RoPE and attention math stay bit-identical.
- Per group of 128 input-channel weights: one scale (and one zero point if asym). The
  group axis is the linear's input dimension, so `in_features % 128 == 0` is required
  for exact grouping (padding is supported otherwise, `reshape_pad_tensor_by_group_size`).
- With `V=0, alpha=beta=1` the scheme reduces exactly to RTN; tuning can only move each
  integer by at most one step (|V| ≤ 0.5) and shrink the clip range to ≥ 0.5×.
- The packed artifact must dequantize to the same `W~` that was tuned: for full-range
  sym the stored unsigned int is `q+8 ∈ [0,15]` with implicit zero 8 (GPTQ "uint4b8"),
  scale may be negative (sign encodes which side hit −8); `(u − 8)·scale` reproduces
  `W~` exactly. Scales are fp16 (`scale_dtype` default), so the fp16-rounded scale is
  used during tuning too (`int.py:229`).
- Block outputs consumed by the next block come from the quantized block (error
  propagation); the FP reference chain is computed separately.
- Fused serving projections (q/k/v, gate/up) must share bits, group_size and sym.

## Calibration

- **Data**: default `NeelNanda/pile-10k` train split, shuffled with `seed=42`
  (`calib_dataset.py:364-410`); samples shorter than `seqlen` are only concatenated
  when there are fewer than `nsamples` full-length samples (`calib_dataset.py:405-409`).
  Loaded from the HF Hub via `datasets.load_dataset` in a subprocess
  (`calib_dataset.py:2153-2246`); on failure it falls back to fineweb-edu, also from the
  Hub (`calib_dataset.py:2240-2246`). On the vast3090 host the Hub answered HTTP 200 for
  `NeelNanda/pile-10k`, but the dataset is **not cached** (`$HF_HOME/datasets` only holds
  wikitext). Offline alternatives: `dataset=` accepts a local `.json/.jsonl` path
  (`calib_dataset.py:1318-1375`, loads the whole file) or a `list[str]` of raw texts
  (`utils/dataset_utils.py:576-586`); `.cache/calib/pile-val-backup/val.jsonl` (1.4 GB,
  Pile validation, the AWQ calibration set) exists on the host and can feed a list of
  the first N texts. Chat template is not applied by default.
- **Sizes (repo default vs paper)**: nsamples 128 (paper 512, `auto-round-best` recipe
  uses 512), seqlen 2048 (paper 2048), batch_size 8 (paper 8; capped at nsamples,
  `compressors/base.py:280`), iters 200 per block (paper 200), lr 1/iters = 5e-3
  (paper 5e-3), minmax_lr = lr, linear decay, `gradient_accumulate_steps=1`,
  `nblocks=1`, AMP on (`amp=True`). Quantization granularity: paper reports asym; the
  repo default `W4A16` is **sym** (full-range), group 128.
- **What is optimized**: per block, `V` (one per weight element) and `min_scale`,
  `max_scale` (one per 128-group) of all 7 linears jointly, against block-output MSE.
- **Memory (24 GB GPU)**: blocks are moved to GPU one at a time; the rest of the model
  stays on CPU/offloaded (`low_cpu_mem_usage=True` default = immediate-save mode).
  Repo's measured cost for an 8B Llama at W4G128 (`docs/step_by_step.md:913-918`):
  torch.compile 12 min / 18 GB; `low_gpu_mem_usage=True` 19 min / 10 GB;
  `low_gpu_mem_usage=True, gradient_accumulate_steps=8, batch_size=1` 25 min / 6 GB
  (datacenter GPU). `low_gpu_mem_usage` keeps cached inputs/outputs on CPU
  (`quantizer.py:496-507`). `device_map=0` (default) uses one GPU;
  `calibration_data_device="cpu"` parks the input cache on host.

## Save and reload

- **What is written** (`format="auto_round"`, W4A16 sym): safetensors with per-linear
  `qweight`, `qzeros`, `scales` (fp16), `g_idx`; all non-quantized tensors in model
  dtype; `config.json` gets `quantization_config = {quant_method: "auto-round",
  packing_format: "auto_round:auto_gptq", bits: 4, group_size: 128, sym: true,
  data_type: "int", block_name_to_quantize: "model.layers", autoround_version, ...}`
  with defaults filtered out (`export/formats/autoround/integer.py:357-371`,
  `export/common/serialization.py:499-524`). Tokenizer is saved alongside.
- **Other formats** (`README.md:245,290-296`): `auto_gptq` (quant_method `gptq`, plus
  `desc_act=false, true_sequential=false, damp_percent=0.01, provider="auto-round"`,
  `export/formats/gptq/format.py:231-237`), `auto_awq` (asym 4-bit, AWQ packing,
  fp16), `llm_compressor` (compressed-tensors), `gguf:*`, `fake`. vLLM loads
  `auto_round` (its `auto-round` quant method), `auto_gptq` (gptq→gptq_marlin),
  `auto_awq` (awq_marlin) and `llm_compressor` (compressed-tensors W4A16 → Marlin);
  on Ampere sm_86 Marlin handles 4-bit, g ∈ {−1,32,64,128}, sym GPTQ (uint4b8) or AWQ
  with zero points. vLLM itself is not installed on this host, so the vLLM side was
  not read from source here.
- **Output directory quirk**: `quantize_and_save(output_dir=X)` writes into
  `X/<basename(model path)>-w4g128` (`compressors/base.py:2107-2160,2256-2258`) and
  returns that folder; `save_quantized(output_dir=X)` with a single format writes into
  `X` directly (`compressors/utils.py:630-655`).
- **Reload in transformers**: `AutoModelForCausalLM.from_pretrained(dir)` sees
  `quant_method="auto-round"` → `AutoRoundQuantizer` (transformers
  `quantizers/quantizer_auto_round.py:32-71`, needs only the `auto_round` package;
  default dtype bf16) → `auto_round.inference.convert_model.convert_hf_model`
  (`convert_model.py:847-938`) replaces Linears with backend QuantLinears. Backend
  `"auto"` picks the highest-priority installed backend for packing
  `auto_round:auto_gptq` (`inference/backend.py:237-239`): `gptqmodel:marlin_zp`
  (prio 6, needs `gptqmodel`), `auto_round:humming_zp` (prio 4, needs `humming`),
  `auto_round:tritonv2_zp` (prio 2, needs `triton>=2.0`), `auto_round:torch_zp`
  (prio 0, pure torch) (`backend.py:499-546,592-620`). In this venv only `triton 3.3.1`
  is present (no gptqmodel / auto-gptq / optimum / humming), so CUDA reload uses
  **`auto_round:tritonv2_zp`**; force it with
  `quantization_config=AutoRoundConfig(backend="tritonv2_zp")` if needed.
  A `quant_method="gptq"` checkpoint would instead go to transformers'
  `GptqHfQuantizer`, which requires `optimum` + `gptqmodel`/`auto-gptq` (absent here).
- **Python API at this commit**: `AutoRound(model, tokenizer=None, scheme="W4A16",
  alg_configs=None, bits=None, layer_config=None, dataset=UNSET, low_gpu_mem_usage=False,
  device_map=0, enable_torch_compile=None, seed=42, low_cpu_mem_usage=True,
  platform="hf", recipe="auto-round", **kwargs)` with kwargs `iters, lr, minmax_lr,
  nsamples, seqlen, batch_size, gradient_accumulate_steps, group_size, sym,
  quant_lm_head, format, scale_dtype, model_dtype, trust_remote_code, amp,
  calibration_data_device` (`autoround.py:134-211`). Returns a compressor with
  `quantize()` → `(model, layer_config)`,
  `save_quantized(output_dir=None, format=None, inplace=True, return_folders=False,
  max_shard_size=None, **kw)` (`compressors/base.py:1963`) and
  `quantize_and_save(output_dir="tmp_autoround", format=None, inplace=True,
  max_shard_size=None, **kw)` → `(model, folders)` (`compressors/base.py:2188`).

## Porting notes

- Qwen3-8B is a plain decoder of `model.layers.*` with 7 bias-free Linears per block; the
  generic block walker covers it with no model glue. q_norm/k_norm are RMSNorms on the
  per-head q/k after projection; they are not Linears, are not wrapped, stay bf16, and
  are inside the block whose output MSE is minimized, so their effect is accounted for.
  GQA (32 q / 8 kv heads, head_dim 128) only changes k/v out_features (1024); grouping
  is along in_features (4096 → 32 groups) so nothing changes. Fused qkv N = 6144 and
  gate_up N = 24576 are multiples of 64 for Marlin; down_proj K = 12288 is a multiple of 128.
- Use `scheme="W4A16"` unchanged (sym, g128, int). This is the repo default, differs from
  the paper's asym, and is what Marlin's GPTQ path wants (sym → no zero points).
- Save with `format="auto_round"`: reloads in transformers with only `auto_round` +
  triton, and vLLM serves it through its `auto-round` quant method on the GPTQ-Marlin
  kernel. `auto_gptq` would also be Marlin-loadable in vLLM but needs optimum+gptqmodel
  for the transformers reload; `auto_awq` requires asym; `llm_compressor` needs the
  `compressed-tensors` package for transformers reload.
- `quantize_and_save` nests the artifact one directory deeper; the wrapper must make
  `OUTPUT_DIR` itself contain `config.json` + safetensors (move the returned folder's
  contents up, or use `save_quantized(OUTPUT_DIR, ...)` after `quantize()`).
- Smoke hooks map directly: `nsamples`, `seqlen`, and `iters` (e.g. 2-5 under
  `QUANT_AGENT_SMOKE=1`; lr auto-scales to 1/iters).
- Estimated full run on an RTX 3090 (24 GB, torch 2.7.1, torch.compile on): roughly
  2.5-4× the repo's datacenter numbers, i.e. ~30-50 min tuning at ~18 GB peak with
  `low_gpu_mem_usage=False`, or ~50-75 min at ~10 GB with `low_gpu_mem_usage=True`;
  plus a few minutes for dataset download/tokenization, model load and packing.

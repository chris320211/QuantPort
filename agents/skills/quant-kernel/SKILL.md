---
name: quant-kernel
description: >-
  Rewrite hot paths to Triton only when WikiText-2 quality_ok is true and the
  parent quant skill passed recommended_action: kernel (dense fakequant, VRAM,
  tokens/s, or prefill_kernel_missing). Use only as a quant-kernel subagent
  launched by the parent. Do not run on a quality failure. Validate-only;
  parent runs the GPU job.
disable-model-invocation: true
---

# Quant Kernel

You are a **subagent**. Do only kernel optimize. Need request JSON, `job_id`,
and `jobs/<job_id>/benchmark.json` (quant-benchmark). Skip unless
`comparison.quality_ok` is true **and** the parent said `recommended_action`
is `kernel` (dense fakequant, VRAM, tokens/s, or `prefill_kernel_missing`). Do not run if quality failed.

This stage is method-agnostic: rewrite whatever hot path this repo uses on
this GPU type. Do not ask the parent mid-stage. If `HF_TOKEN` is unset and
`.env` exists, `source agents/skills/_shared/load_env.sh .env` (never print values).

`PY=$([ -x .venv/bin/python ] && echo .venv/bin/python || command -v python || command -v python3)`
`S="$PY agents/skills/_shared/scripts"`

## Do

0. Check the shared kernel library first
   ([`_shared/kernels/`](../_shared/kernels/README.md)):

   ```bash
   $S/kernels.py match --job-id <job_id> --compute-capability <major.minor>
   ```

   Entries are keyed by the artifact's saved format (its
   `quantization_config.json`), not by model or GPU. If one matches and is
   `supported`, build the overlay on it: copy a `module` entry's file into
   the adapter and call its entry points, or take the listed pieces from a
   `port` entry's `overlay.patch`. With `retune: true`, tune on this GPU
   before the parent runs the job (`tune_config` for `w8a8_int8`). Write a
   new kernel only for hot paths no entry covers.
1. Profile hot Python/CUDA ops from the completed job logs **and**
   `jobs/<job_id>/diagnose.json` `error_excerpt` / `notes` (do not edit the clone).
   Packed dtype/loader verify failures are parent `author_fix`, not this stage.
2. Reread the paper (`paper_path`) and repo for what the method is supposed to do,
   plus `gpu_arch` from the request JSON.
3. Rewrite those ops to **Triton** (small CUDA only if Triton cannot express the
   kernel) **in a new overlay** with strategy `kernel_triton`.

   **Packed / realquant first:** if the method repo already ships a packed
   export (`pack_i4`, `Linear4bit`, `quantized_save`, a `deploy/` int4 matmul,
   **or that repo's equivalent**), the overlay **must** convert the verified
   fakequant checkpoint onto that path (pack weights, quantized activations,
   official GEMM). Do not leave dense fp16 RTN as the artifact. Do not invent
   a new kernel when the repo already has one. Triton-from-scratch is only
   for hot paths the repo does not pack.

   **CUDA kernel dtype contract:** packed int4 bindings often require
   **float16** row/col scales. Empty packed modules default to float32, so
   `load_state_dict` keeps float32 even if the packed file stored fp16.
   After pack **and** after load, cast kernel-facing scale (and activation)
   tensors to the dtype the repo asserts. A verify `AssertionError` on
   `float16` scales is `cuda_kernel_dtype_mismatch` — fix the packed loader,
   do not revert to dense fakequant or invent Triton-on-fakequant.

   **Eval flags are not in state_dict:** many methods fuse transforms at
   reparameterize time and keep Python-only flags (`_eval_mode`, `use_diag`,
   or the repo's equivalents) off the checkpoint. The adapter must enter
   eval/fuse mode **before** load (so fused buffers exist) and restore those
   flags **after** load. Do not re-fuse already packed weights. Missing flags
   are `eval_runtime_flags_missing` (transforms apply twice; PPL explodes).

   **First kernel overlay is complete (any method):** land pack/realquant (if
   the repo has it), dtype casts, eval-flag restore, **and** prefill kernels
   in the **same** overlay. Do not spend one GPU job on packed GEMM and
   another on attention unless diagnose later says `prefill_kernel_missing`.

   **Prefill throughput after packed GEMM:** WikiText-2 tok/s is 2048-token
   prefill, not decode. Naive `torch.matmul` attention is memory-bound at
   that length — use `scaled_dot_product_attention` or the repo's flash path.
   `is_causal` only when `q_len == kv_len`; decode with KV cache is `q_len=1`.
   If this repo fuses transform+quant into one kernel, call that kernel
   instead of Python then a separate quantizer. Import the **kernel module
   directly**; do not import a convenience file that pulls optional extras
   (Hadamard, flash-attn, extra CUDA) unless that extra is required. Triton
   value arguments must be Python floats/ints — 0-d tensors compile as
   pointers.
4. Write and validate like a port worker:

   ```bash
   $S/overlay.py write --slug <slug> --strategy kernel_triton --model-id <id> \
     --base-commit <sha> --patch-file /tmp/kernel.patch --rationale "..."
   $S/overlay.py apply-check --overlay-dir <bundle> --repo .venvs/<slug>/repo
   $S/validate_script.py <script.py> --model-id <id> --output-dir ./quantized/<slug> \
     --overlay-dir <bundle>
   ```

5. Return the new overlay path (validate only). Do **not** launch a GPU job
   and do **not** return `revert` from metrics you did not measure. The parent
   runs `quant-run` + `quant-verify` + `quant-benchmark`, then diagnose. If
   generate fails or WikiText-2 / VRAM / tok/s did not improve, the parent
   keeps the previous overlay (`recommended_action: none` or another kernel
   only if diagnose says so). Return `cannot_author` only if you cannot write
   a valid overlay (no packed path, apply-check failed).

Never edit `.venvs/<slug>/repo`.

## Return

New overlay path, or `cannot_author` if you cannot write a valid overlay.

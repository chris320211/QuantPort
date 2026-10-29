# Pipeline contract (skills-first)

Human overview: root `README.md`. This file is the **agent** contract
(order, retries, layout).

User inputs: **method name**, **model name**, **GPU instance**. No method catalog.

Canonical helpers live in `agents/skills/_shared/scripts/`. Every stage skill
invokes those scripts; do not reimplement clone, overlay, launch, or verify.

Interpreter: Ubuntu GPU AMIs often ship `python3` only; CI may provide `python`.
Resolve once per shell, then use `$S/<script>.py`:

```bash
PY=$([ -x .venv/bin/python ] && echo .venv/bin/python || command -v python || command -v python3)
S="$PY agents/skills/_shared/scripts"
```

Order:

1. `quant-setup` (parent only, **first**). Prefer a mode-600 `.env` created
    once per machine from `.env.example`; load it every shell. Do not recreate
    `.env` if it exists. Session `export` is fallback only.
2. `quant-gather` subagent — paper + GitHub + clone + HF snapshot + GPU facts + **one** venv install
3. `quant-port` coordinator — up to three named strategy subagents; validate; pick a winner (no GPU yet)
4. `quant-run` subagent — one GPU launch per job; no inner overlay patches
5. `quant-verify` subagent — saved artifact must generate (not Hub fp16)
6. `quant-benchmark` subagent — **always** WikiText-2 LLM metrics vs the fp16 snapshot
7. `quant-diagnose` subagent — after a failed run (no verify), after every
   verify, and after every passed benchmark (including success) so stop is
   `recommended_action: none`. A crashed GPU job or failed WikiText-2 is not
   `none` while retries remain (unless OOM/auth/disk).
8. Parent **retry loop** (not a new skill): switch on diagnose `recommended_action` with a GPU budget
9. `quant-kernel` only if diagnose says `kernel`
10. `quant-serve` subagent after a **beneficial** STOP, before publish.
    `export_vllm.py` (CPU) converts the verified artifact into a checkpoint
    vLLM loads natively, detected from the saved tensors (today: int8
    per-channel W8A8 → compressed-tensors). `serve_check.py` (GPU, own
    `.venvs/vllm-serve`) loads it with vLLM and measures WikiText-2 PPL and
    prefill/decode tok/s vs vLLM fp16. Never re-quantizes. `unsupported`
    (e.g. A4 activations) or no GPU is not a failure: publish proceeds with
    the native artifact; serving status never changes the ranking.
    Writes `jobs/<id>/vllm_export.json` and `jobs/<id>/vllm.json`; publishing
    and recording them happen in steps 11 and 12.
11. `quant-publish` in the **parent** after verify + `quality_ok` **and** a
    VRAM or tok/s win versus fp16 (Hub model card + weights + WikiText-2
    metrics). Uses `HF_TOKEN` like `quant-setup`; never a subagent. Hugging
    Face Hub is the **artifact store** (one repo per method × model run). It
    is not the comparison UI. When `jobs/<id>/vllm.json` passed, the same
    stage publishes the export as its own repo `<native repo>-vllm`
    (`publish_vllm.py`: card with the vLLM numbers and GPU, raw
    `vllm.json`) and adds a "Serve with vLLM" section to the native card.
12. `quant-catalog` in the **parent** after a **beneficial** run (quality_ok and
    better VRAM or tok/s than fp16). Standard row: model, GPU instance, method,
    paper URL, method GitHub, Hugging Face URL. Writes
    `library/contributions/` and rebuilds `library/catalog.json`. On
    unsuccessful STOP (`recommended_action: none` and not beneficial), the
    same skill records `library/attempts/` (`--record-attempt`) so failed
    ports are not lost. Docs: `library/LIBRARY.md`, `library/ATTEMPTS.md`.
    The same call stores the run's overlay patch, manifest and wrapper in
    `library/ports/<model>__<gpu>__<method>/` (attempts:
    `library/ports/attempts/<model>__<gpu>__<method>/`; `library.py
    --export-port` backfills an older row), so a result can be reviewed and
    replayed.
    After the row exists, `library.py --record-vllm` attaches the serve
    result (passed with the vLLM repo, failed, unsupported with its reason,
    or pending) as the row's `vllm` block and the `LIBRARY.md` vLLM column.
13. `quant-sync` in the **parent** after catalog **or** after recording an
    unsuccessful attempt (or when asked to update GitHub and Hugging Face).
    Rebuilds `library/`, `git push`es allowlisted library/skill paths to this
    remote, and refreshes the one Hub collection (quality_ok repos only).
    Never commits `quantized/`, `jobs/`, `out/`, or `.env`. Third parties
    without push access still PR `library/contributions/`.
14. Tell the user where weights, metrics, the Hub URL (if uploaded), and
    `library/` rankings live. The stop report **must** include the WikiText-2
    metric table (quantized vs fp16 PPL / tok/s / VRAM) and whether a
    `kernel_triton` overlay ran. Users filter `model_id` / `method_name` /
    `gpu_instance`, then fetch `hub_repo_id` themselves. Rank: `quality_ok`,
    then tok/s, then lower VRAM. `is_best` / `best` is the row to pick. After
    sync, `library.py --rebuild` is the library table;
    `library.py --sync-hf-collection` lists those Hub repos in one collection.

## Request JSON

Handoff file: `out/requests/<slug>.json` (no secrets). Later stages read only that
file plus job ids it names.

Required keys:

| key | meaning |
| --- | --- |
| `method_name` | User method string |
| `model_id` | Model id gather snapshots (usually `org/name`) |
| `gpu_instance` | e.g. `g5.xlarge` |
| `arxiv_id` | arXiv id from gather |
| `paper_path` | `.cache/papers/<id>.txt` |
| `repo_url` | `https://github.com/owner/repo` |
| `repo_path` | `.venvs/<slug>/repo` |
| `repo_commit` | 40-char HEAD SHA |
| `hf_snapshot_path` | `.cache/hf-snapshots/...` |
| `slug` | Path-safe id used for venv, overlays, jobs |

Optional keys gather may add: `instance_type`, `gpu_arch`, `vram_gb`, `gpu`,
`venv_python`. Port may add `winner_overlay_dir`, `winner_script`, `ranked`.
The parent retry loop may add `tried_overlays`, `retry_gpu_jobs_used`,
`retry_gpu_jobs_max` (default 14), `last_job_id`, `last_diagnose_path`,
`best_job_id`, `best_overlay_dir`, `best_ppl_ratio` (lowest WikiText-2
ratio so the next `diagnose_fix` starts from the best overlay, not a
regression).
Gather/`request.py` seeds used=0, max=14, and an empty `tried_overlays` list
for every new slug.

## Layout

- Canonical clone: `.venvs/<slug>/repo` (never edited)
- Venv python: `.venvs/<slug>/bin/python`
- Overlays: `out/overlays/<slug>/<strategy>/<hash>/`
- Jobs: `jobs/<job_id>/`
- Weights: `./quantized/<slug>`
- vLLM export: `./quantized/<slug>-vllm` (`vllm_export.json` inside); records
  `jobs/<job_id>/vllm_export.json` and `jobs/<job_id>/vllm.json`
- vLLM venv: `.venvs/vllm-serve` (never a method venv)
- Library table: `library/catalog.json` (filter `model_id`, `method_name`,
  `gpu_instance`; fetch via `hub_repo_id`)
- Library boards: `library/index.json` and `library/groups/<model>__<gpu>.json`
- Third-party runs: `library/contributions/<model>__<gpu>__<method>.json` (PR)
- Unsuccessful stops: `library/attempts/` and `library/ATTEMPTS.md`
  (not ranked; no Hub weights)
- Ports: `library/ports/<model>__<gpu>__<method>/` (`overlay.patch`,
  `manifest.json`, `quantize.py`, `port.json`); for an attempt, the best
  job's port under `library/ports/attempts/`

## Invariants

- Secrets: follow `agents/AGENTS.md`. Never read, print, `cat`, or paste `.env`
  or token values. Existence only (`test -f .env`). Name keys, not values.
- Never edit the canonical method clone; overlays only, applied to a temp worktree.
- `quant-setup` never runs inside a subagent. Never paste tokens into prompts.
- Installers, git, dry-import, and measurements receive no cloud credentials.
- Quantization and verify receive HF auth only when the model is gated.
- Host-mutating scripts require `--allow-unsafe-host-execution`.
- One GPU job at a time. Port fans out **validate-only** workers.
- Verify must reload the **saved** quantized artifact, not the Hub snapshot.
- `quant-benchmark` always compares WikiText-2 perplexity (and throughput/VRAM)
  for that artifact against the original fp16 snapshot. `verify.py --baseline`
  is not a substitute. Benchmark does **not** write `library/`; only
  `quant-catalog` records a beneficial row. Unsuccessful STOPs use
  `library.py --record-attempt` (parent `quant-catalog`), not this stage.
- Benchmark measures; diagnose classifies; the **parent** retries. Do not fold
  GPU relaunch into run, benchmark, or diagnose. `quant-run` launches once.
- `compare.py` is a back-compat alias of `library.py`. Skills call `library.py`.
- A ranked overlay that **saves vanilla `nn.Linear` / copies `.linear.weight`**
  after `reparameterize_model` still drops method runtime (`T(x)`). Diagnose
  must skip it (`drops_runtime`) and `author_fix` a keep-runtime overlay.
- If the method repo already has a **packed / realquant** export (`pack_i4`,
  `Linear4bit`, `--quantized_save`, or that repo's equivalent), overlays must
  use that path for the saved artifact. Dense fp16 RTN is a calibration
  intermediate, not the efficiency result. `quant-kernel` prefers those
  official kernels over a from-scratch Triton rewrite. CUDA int4 bindings
  typically require **float16** scales; empty packed modules default to
  float32, so overlays must cast kernel-facing scales after pack **and**
  after load (`cuda_kernel_dtype_mismatch` if verify asserts `float16` on
  dequant). Eval-only flags (not in `state_dict`; examples `_eval_mode`,
  `use_diag`) must be restored after load (`eval_runtime_flags_missing`).
- WikiText-2 tokens/s is 2048-token **prefill**. Benchmark warms up one
  window so CUDA/Triton compile is not the score. After packed GEMM,
  kernel work is SDPA/flash attention and the repo's fused T+quant if it
  has one, not Triton-on-fakequant. The first kernel overlay should include
  those prefill kernels so the parent does not need a human to extend the
  GPU budget.

## Retry loop (parent)

Feedback bus is files, not a prose dump of WikiText-2:

- `jobs/<job_id>/benchmark.json` — numbers (`quality_ok`, `ppl_ratio`, VRAM, tok/s)
- `jobs/<job_id>/diagnose.json` — typed `issue_codes`, `error_excerpt` (last
  traceback / verify error), `notes`, `prior_issue_codes`,
  `recommended_action`, `next_overlay_dir` / `next_script`, retry budget

The parent switches **only** on `recommended_action`. It never pastes raw PPL
into a port worker to retune. If a `diagnose_fix` overlay is needed, pass
`issue:` codes plus the diagnose JSON path; the worker **must** read
`error_excerpt` and `notes` and patch that failure (not a generic overlay).

Budget (method/model/GPU-agnostic — same numbers for AWQ, FlatQuant, GPTQ, …):

- First winner GPU job does not count.
- At most **`retry_gpu_jobs_max` (default 14)** extra GPU jobs for the whole
  request: verify-fail next overlay, `retry_ranked_overlay`, and
  `author_fix` overlay followed by run. Diagnose still fills
  `next_overlay_dir` when budget is exhausted so the parent can resume if
  the user extends the budget.
- After quality is inside the 1.5× WikiText-2 gate, diagnose may still say
  `kernel` when the checkpoint is dense fakequant or tokens/s / VRAM did not
  beat fp16. The first kernel overlay must be complete (pack if the repo has
  it, dtype, eval flags, SDPA/flash). Diagnose returns `none` once the overlay
  already uses those, or after two `kernel_triton` overlays. Packed verify
  failures (`cuda_kernel_dtype_mismatch`, `packed_loader_failed`,
  `eval_runtime_flags_missing`, `packed_quality_gap`) and `prefill_kernel_missing` may still
  recommend a GPU job when `retry.remaining` is 0 (helper overage / kernel
  cap). The parent never inspects overlay text; it only switches on
  `recommended_action`. `kernel_triton` already listed does not stop a packed
  dtype/eval-flag fix.
- One GPU job at a time. Never three quantize jobs at once.
- Do not retry OOM-at-same-config, gated auth, or disk full.
- `classify` searches port `ranked` plus authored `diagnose_fix` and
  `kernel_triton` overlays. Packed verify failures only consider packed
  candidates. Skip by hashed path, wrapper `# QUANT_AGENT_OVERLAY_DIR=`, and
  `overlay.patch` sha256 (jobs snapshot dirs are not the hashed path).

| `recommended_action` | Parent does |
| --- | --- |
| `retry_ranked_overlay` | One `quant-run` on `next_overlay_dir`, then verify + benchmark (benchmark only if verify passed). Record the overlay on `tried_overlays` and increment `retry_gpu_jobs_used`. |
| `author_fix` | One validate-only port worker `strategy: diagnose_fix` (issue codes + `error_excerpt`, not raw PPL), then one `quant-run` if budget remains. Diagnose does not write the overlay. |
| `kernel` | `quant-kernel` when diagnose says so. Parent does not inspect the overlay for SDPA. Packed GEMM-only is `prefill_kernel_missing`, not a stop. A packed kernel that failed verify is not a stop. |
| `none` | Stop. Report the WikiText-2 metric table from `benchmark.json` plus `diagnose.json`. |

After each retry: if run or verify failed, diagnose again (no benchmark). After a
passed verify, benchmark, then **always** diagnose (including on success, so
`recommended_action` is `none`). Stop only on `none`. Packed-path
`author_fix` / `kernel` may run when `retry.remaining` is 0; the helper
caps overage and kernel attempts. A failed packed-kernel verify with
`cuda_kernel_dtype_mismatch` is still a fix, not a stop.

Typed issue codes (stable across methods):

| code | Meaning |
| --- | --- |
| `loader_arch` | Unpacked artifact will not load as this model (verify). Next ranked overlay. |
| `process_failed` | GPU quantize process crashed/timed out before verify. Retry untried keep-runtime overlay, or `author_fix` from `error_excerpt`. |
| `benchmark_failed` | WikiText-2 helper failed after generate-smoke passed. Not a stop. |
| `cuda_kernel_dtype_mismatch` | Packed CUDA GEMM/dequant asserts fp16 scales. Cast after pack and load; stay packed. |
| `packed_loader_failed` | Packed artifact failed verify. Stay on packed path; do not retry dense ranked overlays. |
| `eval_runtime_flags_missing` | Packed/fused load skipped `_eval_mode` / `use_diag=False`; transforms apply twice. Only if the current overlay does not already restore those flags. |
| `packed_quality_gap` | Packed, eval flags already restored, WikiText-2 still above 1.5×. Start from `best_overlay_dir`. Try the repo's official weight-only / fp16-activation path if it has one. |
| `prefill_kernel_missing` | Packed GEMM saved, but WikiText-2 tok/s still uses naive attention / unfused T+quant. |
| `transform_or_runtime_dropped_on_save` | Method runtime / activation transform stripped on save. |
| `fakequant_saved_as_dense` | Low-bit values stored as fp16; VRAM cannot beat baseline. |
| `ppl_exploded` | WikiText-2 PPL ≥ 3× fp16. Broken artifact, not “train longer”. |
| `oom_same_config` | Do not retry. |

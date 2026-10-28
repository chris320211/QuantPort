---
name: quant
description: >-
  Dispatcher for autonomous quantization porting. Use when the user names a
  quantization method, a model, and a GPU instance (for example AWQ,
  Qwen2.5 0.5B, g5.xlarge). Launches one subagent per stage.
---

# Quant

Human overview: [README.md](../../../README.md). Skill index: [agents/README.md](../../README.md).

Read [_shared/pipeline_contract.md](../_shared/pipeline_contract.md) and
[_shared/subagents.md](../_shared/subagents.md) first.

You are the **parent**. Collect `method_name`, `model_name`, `gpu_instance`.
This loop is **the same for every method × model × GPU**. Do not skip diagnose
or invent a method-specific shortcut. **First:** `quant-setup` in this session.
If `.env` is missing, ask the user to copy `.env.example` → `.env` (they type
values; you never create or read that file). If `.env` exists, ask them to
`source agents/skills/_shared/load_env.sh .env` when `HF_TOKEN` is unset.
Then launch **one subagent per stage**, in order. Do not do stage work yourself. Helpers:
`"$PY" agents/skills/_shared/scripts/<name>.py` with
`PY=$([ -x .venv/bin/python ] && echo .venv/bin/python || command -v python || command -v python3)`. Secrets: follow
`agents/AGENTS.md` — never read, print, or paste `.env` or token values.

## Order

1. Gather → `out/requests/<slug>.json` (seeds `retry_gpu_jobs_max=14`, `used=0`)
2. Port → ranked overlays under `out/overlays/<slug>/<strategy>/`
3. Run → `job_id` (winner overlay only)
4. Verify → saved artifact must generate (not Hub fp16)
5. Benchmark → **always** WikiText-2 vs the fp16 snapshot
6. **Retry loop (you)** until stop (below). Do not spawn a `quant-retry` skill.
7. Stop report **must** include the WikiText-2 **metric table** from
   `jobs/<job_id>/benchmark.json` (see Report below). Then paths: weights
   (`quantized/<slug>`), `diagnose.json`, `library/catalog.json`. Hugging Face
   Hub is pullable weights; the library table is `library/catalog.json`
   (filter model, method, instance; `is_best` is the pick; `--fetch` prints
   `huggingface-cli download`).
8. If verify passed and `quality_ok` **and** VRAM or tok/s beat fp16,
   launch **quant-serve** (subagent; vLLM export + serve check, never a
   gate), then **quant-publish**, **quant-catalog**, then **quant-sync** in this session
   (not subagents). Catalog writes the standard library row: model, GPU
   instance, method, paper, method GitHub, Hugging Face, plus the vLLM
   result when quant-serve ran (publish adds the `-vllm` repo; catalog
   runs `--record-vllm`). Sync pushes that
   row to this GitHub remote and refreshes the Hub collection. Do not commit
   weight files. Third parties without push access PR
   `library/contributions/<model>__<gpu>__<method>.json`. Docs:
   `library/LIBRARY.md`.
   On STOP when the run is **not** beneficial, still run **quant-catalog**
   with `--record-attempt` (then **quant-sync**) so SpinQuant/QuaRot-style
   failures land in `library/ATTEMPTS.md`. They stay off the public ranking
   and the Hub collection.

## Retry loop (any inputs)

This loop is **the same for every method × model × GPU**. It is not a FlatQuant,
Phi-3, or A10G special case. Do not skip diagnose. Do not invent a
method-specific shortcut. Do not write overlays, apply patches, or run GPU
smokes yourself — only launch stage subagents.

After a **failed** run launch `quant-diagnose` (no verify). After a **failed**
verify launch `quant-diagnose` (no benchmark). After **every**
passed benchmark launch `quant-diagnose` even when `quality_ok` and
`efficiency_improved` are already true, so stop is file-driven
(`recommended_action: none`). A crashed GPU process is not `none`. Do not
diagnose a passed verify before WikiText-2. Then execute **only**
`recommended_action` from `jobs/<job_id>/diagnose.json`. Never inspect overlay
text for SDPA/`kron_matmul`. Never paste WikiText-2 PPL tables or retune from
raw PPL. Always pass `diagnose.json` and `parent_job_id`. The port
`diagnose_fix` worker and `quant-kernel` **must** read `error_excerpt`,
`notes`, `issue_codes`, and the parent job stderr, and patch **that**
traceback. `quant-run` launches once and does not patch overlays. Typed codes
alone are not a fix if the excerpt names a concrete exception (missing
`flash_attn`, CPU vs CUDA, …).

```text
job = winner run
if run failed:
  diagnose(job)   # no verify/benchmark
  goto HANDLE
verify(job)
if verify failed:
  diagnose(job)   # no benchmark
  goto HANDLE
benchmark(job)
LOOP:
  diagnose(job)
HANDLE:
  action = diagnose.json recommended_action
  if action == none: STOP
  if action == retry_ranked_overlay:
      job = quant-run(next_overlay_dir, next_script, parent_job_id, issue)
  elif action == author_fix:
      if next_overlay_dir is set: treat as retry_ranked_overlay
      else: diagnose_fix worker (validate only); job = quant-run(that overlay)
  elif action == kernel:
      overlay = quant-kernel; job = quant-run(kernel overlay)
  record-retry(tried overlay, new job_id, diagnose path)
  if run failed:
    diagnose(job); goto HANDLE
  verify(job)
  if verify failed: diagnose(job); goto HANDLE
  benchmark(job)
  goto LOOP
```

Budget: `retry_gpu_jobs_max` default **14** extra GPU jobs after the first
winner for **quality** retries (ranked overlay, `author_fix` for dropped
runtime / exploded PPL). Diagnose may still recommend packed-path GPU work
when `remaining` is 0 (`cuda_kernel_dtype_mismatch`, `packed_loader_failed`,
`eval_runtime_flags_missing`, `packed_quality_gap`, `prefill_kernel_missing`,
first packed kernel):
at most **two** jobs past max for packed verify/eval-flag fixes, and at most
**two** `kernel_triton` overlays (complete pack overlay + one prefill
follow-up if the first missed SDPA/flash). You do **not** raise the budget
yourself and you do **not** STOP just because `remaining` is 0 when diagnose
says `kernel` / packed-path `author_fix`. STOP only on `recommended_action:
none`. Do not fall back to dense fakequant. One GPU job at a time.

Kernel overlay must be **complete** on the first `quant-kernel` launch (see
`quant-kernel`): if the repo has packed/realquant, pack in that overlay; cast
kernel dtypes; restore eval-only flags; use SDPA/flash for 2048-token prefill.
A second kernel is only for `prefill_kernel_missing`.

| action | You launch |
| --- | --- |
| `retry_ranked_overlay` | `quant-run` with `next_overlay_dir` / `next_script`, `parent_job_id`, `issue:` from `issue_codes`, and `diagnose_json` (worker reads `error_excerpt`). Then verify; benchmark only if verify passed. |
| `author_fix` | One port-style worker `strategy: diagnose_fix` with `issue:` codes, `diagnose.json` (`error_excerpt` + `notes` + `prior_issue_codes`), and parent job stderr (validate only). Then one `quant-run` if diagnose still wants a GPU job. |
| `kernel` | `quant-kernel` only when diagnose says so (quality OK, efficiency not, or `prefill_kernel_missing`). Not a quality retry. |
| `none` | Stop and report the metric table (Report below). Helper returns this for success, budget exhausted, or terminal OOM/auth/disk — not for a crashed GPU process or failed WikiText-2 while retries remain. If the run is not beneficial, `quant-catalog --record-attempt` then `quant-sync`. |

After a retry job starts:

```bash
$S/request.py out/requests/<slug>.json --record-retry \
  --tried-overlay <overlay_dir> --last-job-id <new_job_id> \
  --last-diagnose-path jobs/<prior_job_id>/diagnose.json
```

If `next_overlay_dir` points at an untried keep-runtime overlay, run it
(`retry_ranked_overlay`) instead of authoring another overlay. Only execute
`author_fix` when `next_overlay_dir` is empty. Do not retry OOM-at-same-config.

## Example

User: “port AWQ to Qwen/Qwen2.5-0.5B-Instruct on g5.xlarge”

Same steps for FlatQuant, GPTQ, or any other method name, any HF id, any
instance string. Launch gather (see [_shared/subagents.md](../_shared/subagents.md)).

Then launch **port** as a coordinator (it fans out up to three strategy
subagents). Use the **winner** overlay for run. If verify fails on an arch/loader
bug, try the next ranked overlay — do not launch three GPU jobs at once.

Then run:

```text
Read agents/skills/quant-run/SKILL.md and follow it.
Request JSON: out/requests/awq-qwen25-05b-g5xlarge.json
Overlay: <winner_overlay_dir>
Script: <winner_script>
Do only run. Return job_id.
```

Then verify with that `job_id` and the same request JSON. If **run** failed
(process exit non-zero, no artifact), launch `quant-diagnose` (no verify). If
verify fails, launch `quant-diagnose` (no benchmark). Packed CUDA scale asserts
stay on the packed path; only unpacked `loader_arch` may try the next ranked
overlay. Then **always** launch `quant-benchmark` after a **passed** verify (do
not skip, do not replace it with `verify --baseline`). Enter the retry loop
above. Kernel only if diagnose says `kernel`. When the loop stops, print the
Report table.

## Report

Every time a run stops (`recommended_action: none`, or verify/run failed with
nothing left to try), paste this table from the last `benchmark.json`. Do not
skip it. Say whether a `kernel_triton` overlay ran.

```markdown
### WikiText-2 (`<job_id>`)

| | PPL | tok/s | VRAM GB |
| --- | ---: | ---: | ---: |
| fp16 | <fp16_ppl> | <fp16_tok/s> | <fp16_vram> |
| quantized | <ppl> | <tok/s> | <vram> |

quality_ok: <bool> (ppl_ratio <ratio>)
efficiency: VRAM <delta_gb> / tok/s <delta>
kernel: <yes, kernel_triton | no>
```

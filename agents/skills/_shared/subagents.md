# Subagents

Human overview: root `README.md`. Skill list: `agents/README.md`.

Each stage is a **separate subagent**. The parent (`quant`) only collects the
three inputs, launches one subagent at a time, and passes the previous return
value. It does not search, clone, adapt, or run GPU jobs itself.

## Rules

- One stage per subagent. Fresh context.
- Pass only paths and the three user inputs. Never paste tokens, `.env`, or credential files.
  Secrets: `agents/AGENTS.md` — never read or print `.env`. Each stage shell
  sources `agents/skills/_shared/load_env.sh .env` when `HF_TOKEN` is unset
  and `.env` exists. Do not ask the parent for extra inputs mid-stage.
- `quant-setup`, `quant-publish`, `quant-catalog`, and `quant-sync` stay in
  the **parent**. Do not spawn a subagent for secrets, Hub upload, the library
  row, or `git push`. `quant-setup` asks for a local `.env` once (copy
  `.env.example`); load it every shell. Never read `.env`. Method comparison is
  `library/` (written by `quant-catalog` / `library.py`, shipped by
  `quant-sync`), not a Hub leaderboard and not a subagent. Unsuccessful
  STOPs use `--record-attempt` in that same parent skill.
- Wait until a subagent returns before starting the next **stage**.
- **Port is the exception:** the port coordinator launches up to three *named*
  strategy subagents at once (author + validate only). The parent still runs
  **one** GPU job, using the winner, then the next ranked overlay if verify fails.
- `quant-run` launches **once**. Process failures return `failed`; the parent
  diagnoses. Do not spawn a run subagent that patches overlays.
- If a subagent fails, stop or retry **that** stage. Do not silently skip ahead.
  **Serve is the exception:** `quant-serve` never gates. Publish proceeds
  whatever it returns; report its status alongside.
- Scripts: `"$PY" agents/skills/_shared/scripts/<name>.py` with
  `PY=$([ -x .venv/bin/python ] && echo .venv/bin/python || command -v python || command -v python3)`.

## How to launch

Use the host’s subagent tool (Cursor `Task`, Claude/Codex agent). Point it at the
stage skill and paste the prompt template from that skill.

## Example (gather)

User: `AWQ`, `Qwen/Qwen2.5-0.5B-Instruct`, `g5.xlarge`

Parent launches a subagent with:

```text
Read agents/skills/quant-gather/SKILL.md and follow it.

Inputs:
- method_name: AWQ
- model_name: Qwen/Qwen2.5-0.5B-Instruct
- gpu_instance: g5.xlarge

Do only gather. Return the path to out/requests/<slug>.json.
```

When that returns `out/requests/awq-qwen25-05b-g5xlarge.json`, the parent launches
**port**, then run, then verify, then **always** benchmark, then the **retry
loop** (diagnose → `recommended_action`) for **that** slug. Same loop if the
user named a different method, model, or GPU instance. Do not spawn a retry
skill. Do not paste WikiText-2 numbers into port workers to retune. Pass
`issue:` codes, `jobs/<job_id>/diagnose.json` (`error_excerpt` + `notes`),
and `parent_job_id`. Workers infer the patch from that traceback.

## Example (port — three strategies, one GPU winner)

Parent launches **one coordinator**:

```text
Read agents/skills/quant-port/SKILL.md and follow it.
Request JSON: out/requests/awq-qwen25-05b-g5xlarge.json
You are the port coordinator. Spawn workers for dispatch, llama_alias, and
adapter_only. Validate only. Return a ranked winner. Do not launch a GPU job.
```

Each worker:

```text
Read agents/skills/quant-port/SKILL.md and follow it.
Request JSON: out/requests/awq-qwen25-05b-g5xlarge.json
strategy: llama_alias
Do only this strategy. Write out/overlays/<slug>/llama_alias/. Return validate ok/fail.
```

Run **one** `quant-run` on the winner. If verify fails, diagnose first.
Next ranked overlay is only for unpacked `loader_arch`. Packed CUDA dtype
asserts stay on the packed path. Do not start three quantize jobs at once.

## Example (benchmark — always after verify)

```text
Read agents/skills/quant-benchmark/SKILL.md and follow it.
Request JSON: out/requests/awq-qwen25-05b-g5xlarge.json
job_id: <job_id>
Do only benchmark. Return quantized vs fp16 WikiText-2 metrics.
```

## Example (diagnose — after failed run / every verify / passed benchmark)

```text
Read agents/skills/quant-diagnose/SKILL.md and follow it.
Request JSON: out/requests/awq-qwen25-05b-g5xlarge.json
job_id: <job_id>
Do only diagnose. Return the helper JSON. Do not write an overlay.
```

## Example (parent retry — typed diagnose feedback)

```text
Read agents/skills/quant-run/SKILL.md and follow it.
Request JSON: out/requests/awq-qwen25-05b-g5xlarge.json
Overlay: <next_overlay_dir from diagnose.json>
Script: <next_script>
parent_job_id: <failed_or_unimproved_job_id>
issue: transform_or_runtime_dropped_on_save
diagnose_json: jobs/<job_id>/diagnose.json
# Worker must read error_excerpt + notes in that JSON and the parent stderr.
Do only run. Return job_id.
```

Then verify + (if passed) benchmark that new `job_id`. If the run or verify
failed, diagnose without benchmark. **Always** diagnose after a passed
benchmark (including success) and switch only on `recommended_action`. Stop on
`none`. A failed GPU process is `author_fix` / `retry_ranked_overlay`, not
`none`. Do not inspect overlays for SDPA or fused-kernel names. Packed-path
`kernel` / `author_fix` may run when `retry.remaining` is 0; the helper
owns the cap. `kernel` is efficiency after quality is OK, for any method.
A packed kernel that failed verify with `cuda_kernel_dtype_mismatch` is still
a packed fix, not a stop.

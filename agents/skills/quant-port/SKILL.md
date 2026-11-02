---
name: quant-port
description: >-
  Adapt a gathered quantization method to the chosen model. The coordinator
  plans from evidence: it probes the model's structure, reads the method repo,
  the paper and library recipes, writes a differences spec, picks 1-4 plans,
  has workers write overlays that pass a 2-layer real-weight smoke test on the
  GPU, and returns them ranked. Use only as a quant-port subagent launched by
  the parent quant skill after quant-gather. Also used for plan diagnose_fix
  when the parent executes author_fix.
disable-model-invocation: true
---

# Quant Port

If the prompt has **no** `plan:`, you are the **coordinator** (planner).
If it has `plan: <name>`, you are a **worker**; `plan: diagnose_fix` is the
bounded fix worker the parent launches for `author_fix`.
Request JSON path is required. Never edit the cloned repo. Do not ask the
parent mid-stage: drop a plan that cannot apply, rank what passed smoke,
write the request JSON, and return. If `HF_TOKEN` is unset and `.env` exists,
`source agents/skills/_shared/load_env.sh .env` (never print values).

`PY=$([ -x .venv/bin/python ] && echo .venv/bin/python || command -v python || command -v python3)`
`S="$PY agents/skills/_shared/scripts"`

Files this stage owns (all under the slug):

| Path | What |
| --- | --- |
| `out/analysis/<slug>/arch.json` | Meta-device structure probe (`arch_probe.py`) |
| `out/analysis/<slug>/differences.json` | The differences spec you write (`port_plan.py spec-check`) |
| `.cache/smoke-slices/<slug>-L2/` | First 2 decoder layers with real weights (`port_smoke.py slice`) |
| `out/overlays/<slug>/<plan>/<hash>/` | One overlay bundle per worker attempt |
| `out/overlays/<slug>/<plan>/quantize.py` | That plan's wrapper (diagnose looks here) |
| `out/smoke/<slug>/<plan>/<hash>/smoke.json` | Smoke verdict and metrics for that bundle |

## Coordinator

Do not write overlays yourself. Plan from evidence, not from a fixed menu of
port styles. Workers must **not** compile CUDA; that happened once in gather.

### 1. Gather context (parallel)

Run the probe and the recipe lookup yourself, and launch two read-only
context subagents **in parallel** with them:

```bash
$S/arch_probe.py --request out/requests/<slug>.json --allow-unsafe-host-execution
$S/port_plan.py recipes --request out/requests/<slug>.json
$S/port_smoke.py slice --request out/requests/<slug>.json   # once per slug; reused after
```

- **Architecture probe** (`arch.json`): layer container and groups, every
  linear in layer 0 with its bias, norm classes, tied embeddings, and `flags`
  (`attention_bias`, `gqa`, `qk_norm`, `moe`, ...): where this model departs
  from a plain Llama decoder.
- **Method-repo reader** (subagent): read `.venvs/<slug>/repo` at
  `repo_commit`. Return which model families the repo handles and where
  (file:line), how it walks layers, which linears it quantizes, what it does
  with biases, norms, embeddings, attention heads, how it saves and reloads
  (packed path such as `--quantized_save`, `pack_i4`, `Linear4bit`), and which
  CLI/function is the entry point. Give it `arch.json` so it checks each flag.
- **Paper reader** (subagent): read `paper_path`. Return the bit widths and
  group size the paper reports, which tensors it rotates/smooths/keeps in high
  precision, calibration set and size, and any per-architecture caveat.
- **Recipes**: library rows for this method on the same `model_id` or
  `model_type`. A positive recipe's `port` (rationale, `addresses`,
  `differences`) is a starting plan. A negative recipe (an attempt) names what
  already failed (`root_cause`, `issue_codes`, `packed_quality_gap`, notes):
  do not plan the same thing again without a reason the evidence supports.

### 2. Write the differences spec

Write `out/analysis/<slug>/differences.json`, then check it:

```json
{
  "schema_version": 1,
  "slug": "<slug>",
  "method_name": "<method>",
  "model_id": "<id>",
  "model_type": "<config model_type>",
  "differences": [
    {"id": "D1", "kind": "attention_bias", "severity": "must",
     "detail": "q/k/v_proj have bias; the repo's rotation fuses weights only, so rotated bias is wrong",
     "evidence": ["arch.json layer0.linears self_attn.q_proj bias=true", "quarot/rotation.py:212"],
     "repo_support": "missing"}
  ],
  "smoke": {"weight_bits": 4, "group_size": 128, "max_rel_error": 0.5, "min_coverage": 1.0},
  "recipes": ["library/ports/attempts/<stem>"]
}
```

```bash
$S/port_plan.py spec-check --request out/requests/<slug>.json
```

- One difference per thing the repo's own path does not already do right for
  this model. `must`: the port is wrong without it (a skipped tensor, a wrong
  layout, a reload that cannot rebuild the module). `should`: quality or speed.
  `info`: noted, no patch needed. Every `must` cites evidence (an `arch.json`
  key, a repo file:line, a paper section). Do not invent differences the
  evidence does not show.
- `smoke` sets the gate every candidate is judged by. `weight_bits` and
  `group_size` are what the wrapper quantizes to (the check uses them to tell
  fake-quant weights from float ones). Keep `max_rel_error` at 0.5 and
  `min_coverage` at 1.0 unless the paper keeps some linears in high precision
  (then set `target_regex` / `exclude_regex` to match) or the method is known
  to be lossier on two layers; say why in `notes`.

### 3. Pick plans

Choose **1 to 4** plans; you decide how many. A plan is a short snake_case
name (not `diagnose_fix` or `kernel_triton`) plus the D-ids it addresses and
how. Plans must differ in **how** they handle a `must` difference (e.g.
extend the repo's dispatch for this family vs. reuse its Llama path with a
bias rotation vs. leave the quantizer and add an inference adapter for
reload). One plan is enough when the spec has one obvious fix or a positive
recipe. Every plan must address every `must` it can; a plan that knowingly
skips a `must` needs a stated reason.

Launch the workers **in parallel**, one per plan, each with the request path,
spec path, plan name, D-ids, and the approach in two or three sentences.

### 4. Rank

When workers return:

```bash
$S/port_plan.py rank --request out/requests/<slug>.json --write \
  --overlay-dir out/overlays/<slug>/<plan_a>/<hash>/ --overlay-dir ...
```

Order: smoke passed (hard gate), then `must` coverage from the manifest
`addresses`, then target-linear coverage, then lower `rel_logit_error`, then
fewer patch lines, then fewer files. Only passing overlays are ranked; a
wrapper edited after its smoke, or pointing at another bundle, is not.
`--write` stores `winner_overlay_dir`, `winner_script`
(`<plan>/quantize.py`), `ranked`, and `port_spec_path` in the request JSON.

If nothing passes: **one** re-plan round. Read each failure's
`smoke.json` (`stage`, `reasons`, `uncovered`, `log_tail`), update the spec if
the failure shows a difference you missed, and launch at most 2 new workers.
If nothing passes after that, return `failed` with the smoke evidence. Do not
lower the smoke thresholds to get a pass.

The parent runs **only the winner** on GPU. If verify later fails with a
loader/arch bug, the parent runs the **next** ranked overlay; it does not
re-plan.

## Worker

Do only your `plan`. Read the request JSON, `differences.json`, `arch.json`,
paper, and `.venvs/<slug>/repo` (read-only). Author a unified diff and a
wrapper script, then:

```bash
$S/overlay.py write --slug <slug> --strategy <plan> --model-id <id> \
  --base-commit <40-char-sha> --patch-file /tmp/port.patch \
  --rationale "..." --evidence-file <path> --target-module <mod> \
  --addresses D1 --addresses D3 \
  [--inference-adapter-path inference_adapter.py]
$S/overlay.py apply-check --overlay-dir <bundle> --repo .venvs/<slug>/repo \
  --commit <sha>
$S/validate_script.py out/overlays/<slug>/<plan>/quantize.py --model-id <id> \
  --output-dir ./quantized/<slug> --overlay-dir <bundle> --smoke-hooks
$S/adapter.py <adapter.py> --check-hub-id   # only with an inference adapter
$S/port_smoke.py run --request out/requests/<slug>.json --overlay-dir <bundle> \
  --allow-unsafe-host-execution
```

`--addresses` lists only the D-ids this patch actually handles. The smoke run
is the only GPU use in this stage: a few minutes on two layers, serialized by
a lock. Do not launch a job. If smoke fails, read its `stage` and
`log_tail`, fix the patch or wrapper, write a new bundle, and smoke again (at
most 3 smokes per worker). Return the bundle path, smoke status, and metrics.

### Wrapper contract

Place it at `out/overlays/<slug>/<plan>/quantize.py`. It must contain, in the
first 20 lines:

```python
# QUANT_AGENT_OVERLAY_DIR=<absolute overlay bundle path>
```

and string literals `"QUANT_AGENT_OVERLAY_DIR"` and `"QUANT_AGENT_METHOD_REPO"`
(the launcher injects those env vars). `MODEL_ID` and `OUTPUT_DIR` must be the
exact request values. Do not `from_pretrained(model_id)` for the quantized load
path; load `OUTPUT_DIR` / `model_path`.

Smoke hooks (required; `validate_script.py --smoke-hooks` checks the literals):

| Env var | When set, the wrapper must |
| --- | --- |
| `QUANT_AGENT_SMOKE_MODEL_DIR` | Load the model and tokenizer from this dir instead of `MODEL_ID` |
| `QUANT_AGENT_SMOKE_OUTPUT_DIR` | Save the artifact here instead of `OUTPUT_DIR` |
| `QUANT_AGENT_SMOKE_CALIB_SAMPLES` | Use this many calibration samples |
| `QUANT_AGENT_SMOKE_SEQ_LEN` | Use this calibration sequence length |
| `QUANT_AGENT_SMOKE=1` | Cut optimizer iterations/epochs (rotation or transform training) to a few steps |

Everything else (method flags, bits, save format, adapter) stays identical, so
what passes smoke is what runs at full size.

If the cloned repo has packed/realquant (`pack_i4`, `Linear4bit`,
`--quantized_save`, `deploy/` int4 matmul), the wrapper should **save that
packed artifact** after calibration, not dense fp16 RTN. Fakequant is the
calibrator; packed weights are what later stages reload. If the repo's CUDA
bindings assert `float16` scales (`sym_quant` / `sym_dequant`), the overlay
must cast kernel-facing `weight_scales` / activation scales to that dtype
on pack **and** on adapter load (`cast_linear4bit_kernel_dtypes` or
equivalent). Empty packed modules default to float32 buffers. The adapter
must also restore eval-only flags that `state_dict` does not store
(`restore_packed_eval_runtime`: `_eval_mode`, `use_diag=False`) and must
not call `reparameterize_model` again on already packed weights.

Publish copies every weight and `.json` file in `OUTPUT_DIR`. If the wrapper
saves files the reload path never reads (calibration caches, runtime
snapshots), list them in `quantization_config.json` as
`"publish_exclude": ["<name or glob>", ...]` rather than leaving them for
publish to guess. Do not build custom kernels.

### `plan: diagnose_fix`

A **bounded fix worker**, not a new plan. Read `jobs/<job_id>/diagnose.json`
**and** the parent job stderr (`$S/jobs.py logs <job_id> -n 80`), plus
`port_spec_path` from the request. Author one overlay that fixes **that**
failure:

1. Start from `error_excerpt` and `notes` (the last traceback / verify error).
   Infer the concrete patch (missing `flash_attn` → SDPA/eager; CPU vs CUDA →
   `.to(weight.device)` on scales/RoPE; `sym_quant` fp16 assert → cast scales
   after pack **and** load). Typed `issue_codes` name the class of bug;
   the excerpt is what to change.
2. Also honor `prior_issue_codes` so you do not reintroduce a bug already
   classified on this slug (keep packed path, keep eval flags, keep fp16
   kernel scales if those codes already appeared).
3. If diagnose JSON has `best_overlay_dir`, start the patch from **that**
   overlay, not a last job that regressed. If `packed_quality_gap`, do not
   restore eval flags again; check the spec for a `must` difference the best
   overlay does not address (bias, tied embeddings, GQA, norms) and apply the
   same rotate/fuse the repo uses for the tensors it does handle. If the
   failure shows a difference the spec missed, add it (next free D-id) and
   re-run `spec-check`. If the repo has a weight-only or fp16-activation class
   versus an activation-quant class, try that official path on packed weights.
4. Do not retune from raw PPL. Do not write a generic packed overlay that
   ignores the traceback. Same worker for every method × model × GPU.

Write it with `--strategy diagnose_fix` and `--addresses`, then apply-check,
`validate_script.py --smoke-hooks`, and `port_smoke.py run`. **Smoke must pass
before the parent spends a GPU job on it.** Return the bundle, its smoke
status, and the script path `out/overlays/<slug>/diagnose_fix/quantize.py`.

## Return (coordinator)

```text
spec: out/analysis/<slug>/differences.json (must: D1, D3)
winner: out/overlays/<slug>/bias_rotate/<hash>/
script: out/overlays/<slug>/bias_rotate/quantize.py
ranked: out/overlays/<slug>/bias_rotate/<hash>/, out/overlays/<slug>/dispatch_qwen2/<hash>/
failed: adapter_reload (smoke reload: KeyError 'weight_scales')
```

`ranked` is overlay **directories** (hashed bundles), not plan names.
`port_plan.py rank --write` stores those paths so diagnose can pick the next
untried overlay. On `failed`, return each plan's smoke `stage` and `reasons`.

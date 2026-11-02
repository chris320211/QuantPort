---
name: quant-port
description: >-
  Adapt a gathered quantization method to the chosen model. The coordinator
  learns how the method quantizes (paper + repo, written up as a
  model-independent method model), maps its steps and tensor roles onto this
  model's modules, picks 1-4 plans to apply it, has workers write overlays
  that pass a 2-layer real-weight smoke test on the GPU, and returns them
  ranked. Use only as a quant-port subagent launched by
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
| `out/analysis/<slug>/method.md` | The algorithm, independent of any model, that you write |
| `out/analysis/<slug>/differences.json` | Its roles mapped onto this model, plus differences (`port_plan.py spec-check`) |
| `.cache/smoke-slices/<slug>-L2/` | First 2 decoder layers with real weights (`port_smoke.py slice`) |
| `out/overlays/<slug>/<plan>/<hash>/` | One overlay bundle per worker attempt |
| `out/overlays/<slug>/<plan>/quantize.py` | That plan's wrapper (diagnose looks here) |
| `out/smoke/<slug>/<plan>/<hash>/smoke.json` | Smoke verdict and metrics for that bundle |

## Coordinator

The job is to understand how the method quantizes a model, then apply that
algorithm to **this** model's weights the way the paper and repo describe it.
The repo's model-specific code (its Llama path, its family switch) is a
reference implementation, not the boundary of what you can port. Plan from
that understanding, not from a fixed menu of port styles. Do not write
overlays yourself. Workers must **not** compile CUDA; that happened once in
gather.

### 1. Gather context (parallel)

Run the probe, the slice and the recipe lookup yourself, and launch two
read-only context subagents **in parallel** with them:

```bash
$S/arch_probe.py --request out/requests/<slug>.json --allow-unsafe-host-execution
$S/port_plan.py recipes --request out/requests/<slug>.json
$S/port_smoke.py slice --request out/requests/<slug>.json   # once per slug; reused after
```

- **Architecture probe** (`arch.json`): layer container and groups, every
  linear in layer 0 with its bias, norm classes, tied embeddings, and `flags`
  (`attention_bias`, `gqa`, `qk_norm`, `moe`, ...). Flags are hints, not a
  checklist: read the module list itself for anything unusual.
- **Method reader** (subagent): read `paper_path` **and**
  `.venvs/<slug>/repo` at `repo_commit` together, and explain the algorithm
  without reference to any one model. It returns the content of `method.md`
  (below), citing a paper section and a repo `file:line` for each step. It
  must say which repo functions are generic building blocks (quantizers,
  Hadamard/rotation utilities, smoothing, GPTQ solver, packing, save/load
  classes, kernels) and which are glue for specific model families, and
  which entry point the repo's own pipeline uses.
- **Recipes**: library rows for this method on the same `model_id` or
  `model_type`. A positive recipe's `port` (its `method`, `mapping`,
  `differences`, rationale, `addresses`) is a head start; read its
  `method.md`. A negative recipe (an attempt) names what already failed
  (`root_cause`, `issue_codes`, `packed_quality_gap`, notes): do not plan the
  same thing again unless the evidence gives a new reason.

### 2. Write the method model

Write `out/analysis/<slug>/method.md` from the method reader's return. It
describes the algorithm, not this model, under these `##` headings
(`spec-check` rejects a missing or empty one):

- **Steps**: the algorithm in order (e.g. fuse norms, rotate, smooth,
  calibrate, solve, round, pack), each with what it computes, its inputs, and
  its paper/repo citation.
- **Tensor roles**: every kind of tensor the steps touch, named by role, not
  by module name: residual stream, attention input/output projections,
  value/output pair, MLP input/output, norm weights (and biases),
  embeddings, LM head, linear biases, KV cache, activations. Per role: which
  steps touch it and how (rotated on input side, output side, both; smoothed;
  quantized to which bits; kept in fp16).
- **Invariants**: what must stay equal for the transformed model to compute
  the same function before quantization (an orthogonal rotation is cancelled
  by its transpose on every reader and writer of the residual stream; a
  smoothing scale moved out of an activation is moved into the next weight;
  norms are fused before rotation). These decide what an unfamiliar tensor
  needs.
- **Calibration**: data, sample count, sequence length, what is optimized or
  measured, and for how many iterations.
- **Save and reload**: what the repo writes (packed int weights, scales,
  rotation matrices, fake-quant fp16), which classes rebuild it, and which
  runtime flags or kernels the reload needs.

### 3. Map the algorithm onto this model and write the spec

For every tensor role in `method.md`, find the modules in `arch.json` that
play it. That `mapping` is the core of `out/analysis/<slug>/differences.json`:

```json
{
  "schema_version": 1,
  "slug": "<slug>",
  "method_name": "<method>",
  "model_id": "<id>",
  "model_type": "<config model_type>",
  "mapping": [
    {"role": "attention input projections", "steps": ["rotate", "quantize"],
     "modules": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"], "status": "mapped"},
    {"role": "linear biases on residual readers", "steps": ["rotate"],
     "modules": ["self_attn.q_proj.bias", "self_attn.k_proj.bias", "self_attn.v_proj.bias"],
     "status": "extra", "note": "the method rotates weights only; these biases sit after the rotated input"},
    {"role": "embeddings", "steps": ["rotate"], "modules": ["model.embed_tokens", "lm_head"],
     "status": "partial", "note": "tied: one rotation must serve both sides"}
  ],
  "differences": [
    {"id": "D1", "kind": "attention_bias", "role": "linear biases on residual readers", "severity": "must",
     "detail": "biases are unaffected by an input-side rotation; check the invariant and keep them unrotated, or untie if the repo folds them",
     "evidence": ["arch.json layer0.linears self_attn.q_proj bias=true", "method.md Invariants", "quarot/rotation.py:212"],
     "repo_support": "missing"}
  ],
  "smoke": {"weight_bits": 4, "group_size": 128, "max_rel_error": 0.5, "min_coverage": 1.0},
  "recipes": ["library/ports/attempts/<stem>"]
}
```

```bash
$S/port_plan.py spec-check --request out/requests/<slug>.json
```

- `status`: `mapped` (the algorithm applies as written), `partial` (the role
  exists but differs: fused, tied, split, reshaped), `unmapped` (the method
  expects it, this model has none), `extra` (this model has tensors in that
  position the method never mentions: biases, extra norms, gates, experts).
  Every role that is not `mapped` needs a difference naming it in `role`;
  use an `info` difference if nothing has to change.
- Decide each difference from `method.md` **Invariants**, not from what the
  repo's glue happens to do. `must`: the port computes the wrong function or
  cannot reload without it. `should`: quality or speed. `info`: noted, no
  patch needed. Every `must` cites evidence (an `arch.json` key, a
  `method.md` section, a repo `file:line`, a paper section).
- `smoke` sets the gate every candidate is judged by. `weight_bits` and
  `group_size` are what the wrapper quantizes to (the check uses them to tell
  fake-quant weights from float ones). Keep `max_rel_error` at 0.5 and
  `min_coverage` at 1.0 unless the paper keeps some linears in high precision
  (then set `target_regex` / `exclude_regex` to match) or the method is known
  to be lossier on two layers; say why in `notes`.

### 3b. Pick plans

Choose **1 to 4** plans; you decide how many. A plan is a short snake_case
name (not `diagnose_fix` or `kernel_triton`), the D-ids it addresses, and how
it applies the algorithm to this model. Plans should differ in a real way,
for example:

- extend the repo's own pipeline for this family (its dispatch, its layer
  walker) and handle the `must` differences inside it;
- implement the algorithm for this model as a new module in the overlay that
  follows `method.md` step by step, calling the repo's generic building
  blocks (quantizer, rotation/Hadamard, solver, packing, save classes) on
  this model's modules, without the repo's model-specific glue;
- keep the repo's quantizer and change only how the artifact is saved and
  reloaded (an inference adapter), when save/load is the gap.

These are examples, not a menu: use whatever the mapping calls for. One plan
is enough when the spec has one obvious path or a positive recipe. Every plan
addresses every `must` it can; a plan that knowingly skips a `must` states
why.

Launch the workers **in parallel**, one per plan, each with the request path,
the `method.md` and spec paths, the plan name, its D-ids, and the approach in
two or three sentences.

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
`smoke.json` (`stage`, `reasons`, `uncovered`, `log_tail`) and the workers'
notes. If a failure shows a role or invariant you got wrong, fix `method.md`
or the mapping first, re-run `spec-check`, then launch at most 2 new workers.
If nothing passes after that, return `failed` with the smoke evidence. Do not
lower the smoke thresholds to get a pass.

The parent runs **only the winner** on GPU. If verify later fails with a
loader/arch bug, the parent runs the **next** ranked overlay; it does not
re-plan.

## Worker

Do only your `plan`. Read the request JSON, `method.md`, `differences.json`,
`arch.json`, the paper, and `.venvs/<slug>/repo` (read-only). Apply the
algorithm in `method.md` to this model's modules as the `mapping` lists them;
where the plan writes new code, follow the paper and the repo's generic
functions, and keep each step's invariant. A patch may add new files. Author
a unified diff and a wrapper script, then:

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
a lock. Do not launch a job. If smoke fails, read its `stage`, `reasons`,
`uncovered` and `log_tail`, decide which step or invariant broke, fix the
patch or wrapper, write a new bundle, and smoke again (at most 3 smokes per
worker). A missed role or invariant goes back to the coordinator in your
return, not into a silent workaround. Return the bundle path, smoke status, and metrics.

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
`port_spec_path` from the request and the `method.md` next to it. Author one overlay that fixes **that**
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
   failure shows a role the mapping got wrong or a difference the spec missed,
   fix the mapping, add the difference (next free D-id) and re-run
   `spec-check`. If the repo has a weight-only or fp16-activation class
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

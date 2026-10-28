---
name: quant-catalog
description: >-
  After a successful beneficial quantization run (quality_ok and better VRAM
  or tok/s than fp16) and quant-publish, record one library row: model, GPU
  instance, method, paper, method GitHub repo, and published weights. After an
  unsuccessful STOP, record the attempt instead (library/attempts) so failed
  ports stay off the public ranking but are not lost. Use in the parent after
  quant-publish, or on STOP when the run is not beneficial. Not a subagent.
  Does not refresh the Hub collection (quant-sync does).
disable-model-invocation: true
---

# Quant Catalog

Parent session only. Need `job_id` and the request JSON. Do not re-quantize.
Never read `.env`. Do not edit `catalog.json` or `attempts.json` by hand.
Do not commit weights. Then run **quant-sync** in this parent session.

`PY=$([ -x .venv/bin/python ] && echo .venv/bin/python || command -v python || command -v python3)`
`S="$PY agents/skills/_shared/scripts"`

## Beneficial run

Skip unless verify passed, `quality_ok`, and the run beat fp16 **VRAM or
tok/s**. Need the Hub URL from `quant-publish`.

Standard row: **model**, **GPU instance**, **method**, **paper**, **method
repo**, **weights**. Same shape every time. Writes
`library/contributions/` and rebuilds `library/catalog.json`.

```bash
$S/library.py --catalog --job-id <job_id> \
  --request out/requests/<slug>.json \
  --hub-url https://huggingface.co/<you>/<slug>
$S/library.py --rebuild
```

If quant-serve ran, attach its result to the row (always, whatever the
status: passed, failed, unsupported, or pending without a GPU):

```bash
$S/library.py --record-vllm --job-id <job_id> --request out/requests/<slug>.json \
  [--vllm-repo-id <org>/<name>-vllm]   # only when quant-publish uploaded it
```

This adds the row's `vllm` block and the `LIBRARY.md` vLLM column. It never
changes the ranking. The vLLM speedups name their own GPU, which may differ
from the board's.

Do not call `--sync-hf-collection` here. **quant-sync** refreshes the Hub
collection and pushes GitHub.

Third parties without push access PR
`library/contributions/<model>__<gpu>__<method>.json`.

## Unsuccessful stop

When diagnose is `recommended_action: none` and the run is **not**
beneficial (budget exhausted, PPL exploded, verify/run failed, or
quality/efficiency missed the gate), record the attempt. Do **not** upload
weights. Do **not** add a Hub collection item.

```bash
$S/library.py --record-attempt --job-id <job_id> \
  --request out/requests/<slug>.json
$S/library.py --rebuild
```

Writes `library/attempts/<model>__<gpu>__<method>.json` and
`library/ATTEMPTS.md`. Failed WikiText-2 jobs stay off
`library/catalog.json`. List later with `$S/library.py --attempts`.

Docs: `library/LIBRARY.md`, `library/ATTEMPTS.md`.

## Return

Beneficial:

```text
status: cataloged
model_id: <org/model>
gpu_instance: <instance>
method_name: <method>
paper_url: https://arxiv.org/abs/<id>
repo_url: https://github.com/<owner>/<repo>
hub_url: https://huggingface.co/<you>/<slug>
contribution: library/contributions/<file>.json
vllm: passed | failed | unsupported | pending | skipped
```

Unsuccessful:

```text
status: recorded
model_id: <org/model>
gpu_instance: <instance>
method_name: <method>
stop_reason: budget_exhausted | quality_failed | verify_failed | process_failed | ...
issue_codes: <codes>
job_id: <last job>
attempts: library/attempts.json
```

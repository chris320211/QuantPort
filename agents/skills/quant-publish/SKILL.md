---
name: quant-publish
description: >-
  After a passed verify and WikiText-2 benchmark that is beneficial
  (quality_ok and better VRAM or tok/s than fp16), stage the saved quantized
  artifact plus those metrics for Hugging Face Hub. Hub is the artifact store,
  not the method comparison board. Use in the parent session only (HF token).
disable-model-invocation: true
---

# Quant Publish

Run this in the **parent** session only, like `quant-setup`. Never spawn a
subagent (the Hub token must not enter a child prompt). Need `job_id` and
the request JSON. Skip unless verify passed, `comparison.quality_ok` is true,
**and** VRAM or tok/s beat fp16 (same gate as `quant-catalog`). Do not
re-quantize.

This stage is method-agnostic: upload whatever artifact that job saved, with
the WikiText-2 numbers from `benchmark.json`. Not a method catalog. One Hub
repo is one method's weights + card. Users rank methods in `library/`
(keyed by `model_id` × `gpu_instance`), not by browsing Hub model cards.

`PY=$([ -x .venv/bin/python ] && echo .venv/bin/python || command -v python || command -v python3)`
`S="$PY agents/skills/_shared/scripts"`

## Do

Stage without a token:

```bash
$S/publish.py --job-id <job_id> --request out/requests/<slug>.json
```

Writes `out/hub/<slug>/` (model card, `metrics.json`, tokenizer/config,
adapter/overlay when present). Large packed weights are referenced from
`quantized/<slug>` so they are not duplicated.

File selection knows no method names: weights and `.json` configs,
tokenizer/license files, `.py` that config.json's `auto_map` loads, and the
inference adapter. Files listed in `quantization_config.json`
`publish_exclude` are left out. The manifest's `skipped` says why each
dropped file was dropped; check it before `--upload`.

Upload (needs `HF_TOKEN` already in the parent environment from `.env`;
never paste the value into chat):

```bash
$S/publish.py --job-id <job_id> --request out/requests/<slug>.json \
  --repo-id <org>/<name> --upload --allow-unsafe-host-execution
```

`--repo-id` is `org/name` on the Hub. Default suggestion is
`<hf-username>/<slug>` (their account is fine). Create a **public model** repo.
Do not upload Hub fp16, job logs, or `.env`. This helper uploads weights
only. It does **not** write `library/` or refresh the Hub collection.
After a successful `--upload`, run **quant-catalog** then **quant-sync** in
this same parent session.

If the token is missing, leave the staged bundle and tell the user to set
`HF_TOKEN` via `quant-setup`, then re-run with `--upload`.

### vLLM build

Only when `jobs/<job_id>/vllm.json` has `status: passed` (quant-serve). Skip
otherwise; never publish an export without a passed check.

```bash
$S/publish_vllm.py --export-dir quantized/<slug>-vllm \
  --serve-report jobs/<job_id>/vllm.json --request out/requests/<slug>.json \
  --base-repo-id <org>/<name> --upload --allow-unsafe-host-execution
```

Publishes `<org>/<name>-vllm`: the export's safetensors unchanged, a card
with `vllm serve`, the vLLM PPL ratio, drift, prefill/decode speedups, and
the GPU and vLLM version they were measured with, plus `vllm.json` and
`vllm_export.json` with local paths removed. Then adds a "Serve with vLLM"
section to `<org>/<name>`'s card (replaced in place on re-runs). Drop
`--upload` to stage only (`out/hub/<name>-vllm/`).

## Return

```text
status: staged | uploaded
stage_dir: out/hub/<slug>/
hub_url: https://huggingface.co/<org>/<name>   # only if uploaded
metrics: out/hub/<slug>/metrics.json
vllm_hub_url: https://huggingface.co/<org>/<name>-vllm   # only if the serve check passed
```

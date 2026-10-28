# Contribute a library row

The ranking is [library/LIBRARY.md](../LIBRARY.md). Quantized weights stay on
**your** public Hub repo. The collection is
https://huggingface.co/collections/chris320211/quantport-library-6aaf22fcafd69b39eabc9230

Do not commit weight files. Do not edit `catalog.json`.

A run counts if WikiText-2 stays within 1.5× fp16 perplexity **and** VRAM or
tok/s beat fp16. Then:

1. `quant-publish` — upload `--repo-id <you>/<slug>`
2. `quant-catalog` — writes this JSON
3. `quant-sync` — GitHub + Hub collection. Without push access, open a PR.

Unsuccessful stops (budget exhausted, exploded PPL, verify/run failed) go to
[`../ATTEMPTS.md`](../ATTEMPTS.md) via `--record-attempt`. Do not PR those as
contributions.

`model_id` is the model you quantized (usually `org/name`). It is not limited
to Microsoft Phi or any one Hub family.

```bash
PY=$(command -v python || command -v python3)
S="$PY agents/skills/_shared/scripts"
$S/library.py --catalog --job-id <job_id> --request out/requests/<slug>.json \
  --hub-url https://huggingface.co/<you>/<slug>
$S/library.py --rebuild
```

Required fields: `model_id`, `gpu_instance`, `method_name`, `arxiv_id` or
`paper_url`, `repo_url` (`https://github.com/owner/repo`), `hub_repo_id`.

```json
{
  "schema_version": 1,
  "model_id": "org/your-model",
  "method_name": "AWQ",
  "gpu_instance": "g5.2xlarge",
  "arxiv_id": "2306.00978",
  "paper_url": "https://arxiv.org/abs/2306.00978",
  "repo_url": "https://github.com/mit-han-lab/llm-awq",
  "hub_repo_id": "your-org/your-quantized-model",
  "quality_ok": true,
  "ppl": 7.0,
  "ppl_ratio": 1.1,
  "tokens_per_s": 4000.0,
  "peak_vram_gb": 4.0,
  "fp16_ppl": 6.36,
  "fp16_tokens_per_s": 3300.0,
  "fp16_peak_vram_gb": 9.7,
  "packed": true,
  "dataset": "wikitext-2-raw-v1",
  "max_seq_len": 2048
}
```

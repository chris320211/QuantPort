---
name: quant-serve
description: >-
  After a beneficial STOP (quality_ok and a VRAM or tok/s win), convert the
  verified artifact into a checkpoint vLLM loads natively, then load it with
  vLLM and measure WikiText-2 PPL and prefill/decode throughput against vLLM
  fp16. Use only as a quant-serve subagent launched by the parent quant skill
  before quant-publish. Never re-quantizes. An unsupported format is a normal
  result, not a failure.
disable-model-invocation: true
---

# Quant Serve

You are a **subagent**. Do only serve. Need `job_id` and the request JSON.
Do not ask the parent mid-stage. Never read `.env`; this stage needs no tokens.

`PY=$([ -x .venv/bin/python ] && echo .venv/bin/python || command -v python || command -v python3)`
`S="$PY agents/skills/_shared/scripts"`

## Do

1. Export (CPU, no GPU, no tokens):

   ```bash
   $S/export_vllm.py --job-id <job_id> --request out/requests/<slug>.json
   ```

   Exit 0 `exported` → `quantized/<slug>-vllm/` (safetensors + `config.json`
   with a compressed-tensors `quantization_config`, tokenizer; no `.py`, no
   pickles). Exit 2 `unsupported` → stop and return `unsupported` with
   `reason_code`. Both write `jobs/<job_id>/vllm_export.json`.

   The format is detected from the saved tensors, not the method name. Only
   `w8a8_int8` (int8 per-channel weights + dynamic per-token int8
   activations) exports today. Typed refusals:

   | `reason_code` | Meaning |
   | --- | --- |
   | `activation_bits_unsupported` | A4 or lower (FlatQuant/QuaRot W4A4). No vLLM int4-activation GEMM outside NVFP4/MXFP4 on Blackwell. |
   | `pickle_weights` | `.pt`/`.bin` only. Re-run with `--allow-torch-load` (uses `torch.load(weights_only=True)`; run under `.venvs/<slug>/bin/python`). |
   | `activation_scheme_unknown` / `activation_scheme_unsupported` | `quantization_config.json` lacks `act_quant`, or it is not `per_token`. |
   | `weight_scheme_unsupported` / `weight_scale_layout_unsupported` | Not per-output-channel int8 scales. |
   | `unrecognized_quantized_tensors` | Integer tensors that match no profile (packed int4, GPTQ `qweight`, …), or leftover float side tensors (zero points, `input_scale`, smoothing scales, Hadamard/rotation matrices). |
   | `runtime_transform_unsupported` | `quantization_config.json` enables an online rotation or Hadamard. vLLM's int8 path has none. |
   | `asymmetric_unsupported` | Asymmetric weights or activations (`w_sym: false`, `*_asym`). |
   | `quantized_embedding_unsupported` | `lm_head` or the embedding is int8. vLLM has no int8 path for those layers. |
   | `mixed_fused_group` | Part of q/k/v or gate/up is int8 and part float. vLLM fuses each group with one scheme. |
   | `tied_lm_head_mismatch` | Config ties `lm_head` to the embedding but the saved head differs. vLLM would ignore the saved head. |
   | `no_quantized_linears`, `no_weights`, `config_missing`, `already_quantized_config` | Nothing to convert. |

   Do not hand-edit the export to get past a refusal. Do not author an
   overlay. A new profile is a change to `export_vllm.py` with tests.

   Check `scale_convention.verdict` in the output. `confirmed` means
   `int8 * weight_scale` reproduces the fp16 snapshot on modules the method
   left untransformed. `inconclusive` is normal for methods that transform
   every linear. `contradicted` means the export would be wrong: return
   `failed` with `vllm_ppl_mismatch` and skip step 2.

2. Serve check (GPU, own venv). Once per machine:

   ```bash
   $S/serve_check.py --install --python <python3.11+> --allow-unsafe-host-execution
   ```

   vLLM pins its own torch; `.venvs/vllm-serve` is never a method venv. The
   install ends by importing vLLM and checking CUDA, so it reports `failed` on
   a host without a GPU. If the host driver is older than the pinned wheel's
   CUDA runtime (import error such as `libcudart.so.13`, or `torch.cuda` false
   on a CUDA 12.x driver), reinstall with `--vllm-cuda cu129`: the same vLLM
   version from its CUDA 12.9 release wheel, with torch from the matching
   PyTorch index. The venv must be the real `.venvs/vllm-serve` directory (a
   symlink is refused). Keep `datasets<5`: 5.x pulls in `torchcodec`, which
   breaks the vLLM import on CUDA 12 builds.

   ```bash
   $S/serve_check.py --job-id <job_id> --request out/requests/<slug>.json \
     --allow-unsafe-host-execution
   ```

   Writes `jobs/<job_id>/vllm.json` and stamps `serve_check` on
   `quantized/<slug>-vllm/vllm_export.json`. Both sides run in float16. If
   vLLM refuses float16 for the architecture (Gemma 2), both rerun in
   bfloat16; the report's `dtype` and the published card say so. Gates:

   | issue code | Meaning |
   | --- | --- |
   | `vllm_runtime_unavailable` | The serve venv cannot import vLLM or sees no CUDA device. A host problem, not the export: fix the venv (`--vllm-spec` for a CUDA runtime mismatch) and rerun. |
   | `vllm_load_failed` | vLLM could not load or generate the export. Tail the log in `jobs/<job_id>/vllm/`. |
   | `vllm_baseline_failed` | vLLM fp16 failed (OOM, load error, non-finite PPL). The quantized numbers are kept in the report; no verdict on the export. |
   | `vllm_quality_failed` | vLLM PPL > 1.5× vLLM fp16. |
   | `vllm_ppl_mismatch` | vLLM PPL ratio / HF `benchmark.json` ratio > 1.02. Export bug, not quantization. |

   No NVIDIA GPU on this host: stop after step 1 and return `exported`
   (`serve_check: pending`). Never claim vLLM support without a passed check.

3. Same-GPU Transformers comparison (passed checks only, same GPU). The HF
   benchmark ran on the board's GPU, so its tok/s cannot be set against vLLM.
   Measure the native artifact and the fp16 snapshot in plain Transformers on
   this GPU with the **method venv** (never the serve venv), then attach both:

   ```bash
   M=.venvs/<slug>/bin/python; C=.cache/eval/wikitext-2-raw-v1-test.jsonl
   QUANT_AGENT_METHOD_REPO=<method repo> PYTHONPATH=<method repo> \
     $M $S/hf_runtime.py measure --model-dir quantized/<slug> --eval-path $C \
     --adapter quantized/<slug>/quant_agent_inference_adapter.py --out jobs/<job_id>/hf_runtime_int8.json
   $M $S/hf_runtime.py measure --model-dir <fp16 snapshot> --eval-path $C \
     --dtype <the report's dtype> --out jobs/<job_id>/hf_runtime_base.json
   $S/hf_runtime.py attach --report jobs/<job_id>/vllm.json \
     --quantized jobs/<job_id>/hf_runtime_int8.json --baseline jobs/<job_id>/hf_runtime_base.json
   ```

   Same windows and decode test as step 2. `attach` refuses sides from another
   GPU or a baseline dtype that differs from the serve check's. The result
   adds Transformers columns to the vLLM results on both Hub cards and
   Transformers rows to the vLLM table under the library board. Plain
   Transformers int8 decodes slower than fp16 (it quantizes activations at
   every step); say so, and point serving users to the `-vllm` build. Skip
   this step if the method venv is gone; the vLLM results stand alone.

## Backfill an already-published run

```bash
$S/export_vllm.py --model-dir <downloaded hub repo> --output-dir quantized/<slug>-vllm \
  --reference <fp16 snapshot>
$S/serve_check.py --model-dir quantized/<slug>-vllm --baseline-dir <fp16 snapshot> \
  --hf-benchmark <metrics.json from the hub repo> --out out/serve/<slug>.json \
  --allow-unsafe-host-execution
```

Then, if the check passed, publish and record it (parent session, needs
`HF_TOKEN` in the environment; never on a rented host you don't control):

```bash
$S/publish_vllm.py --export-dir quantized/<slug>-vllm --serve-report out/serve/<slug>.json \
  --base-repo-id <org>/<slug> --model-id <org/model> --method-name <method> \
  --upload --allow-unsafe-host-execution
$S/library.py --record-vllm --model-id <org/model> --gpu-instance <instance> --method <method> \
  --vllm-export quantized/<slug>-vllm/vllm_export.json --serve-report out/serve/<slug>.json \
  --vllm-repo-id <org>/<slug>-vllm
```

A failed or unsupported result is recorded the same way, without
`publish_vllm.py` and without `--vllm-repo-id`.

To add step 3's numbers to a build that is already published, attach them to
the serve report, then rerun the two commands above with `--refresh-card` in
place of `--upload`. It re-uploads only the card and records, never the
weights, and refreshes the base card's vLLM section.

`--force` replaces only a directory that already holds a `vllm_export.json`,
and the output may never contain the source. Export runs on any machine (CPU only). The serve check needs a GPU with
compute capability ≥ 7.5 (T4, A10G, L4, A100, …).

## Return

```text
status: passed | failed | exported | unsupported
reason_code: <export refusal>            # unsupported only
issue_codes: <vllm_* codes>              # failed only
export_dir: quantized/<slug>-vllm/
report: jobs/<job_id>/vllm.json          # after a serve check
ppl_ratio: <vLLM quant / vLLM fp16>  drift: <vs HF benchmark>
prefill_speedup: <x>  decode_speedup: {1: <x>, 16: <x>}
hf_runtime: attached | skipped           # step 3
```

The parent publishes the native artifact whatever this returns. On
`passed`, quant-publish also publishes `<repo>-vllm`; quant-catalog records
every status with `--record-vllm`. Serving status never changes `quality_ok`
or the library ranking.

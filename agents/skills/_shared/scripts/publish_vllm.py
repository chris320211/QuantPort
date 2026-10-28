#!/usr/bin/env python3
"""Stage (and optionally upload) a vLLM export as its own Hub repo.

Only a passed serve check publishes: the repo's card claims vLLM support.
The weights are the export's safetensors, unchanged. The card carries the
serve-check numbers and names the GPU they were measured on; `vllm.json` and
`vllm_export.json` ship as the raw records with local paths removed. On upload
the base repo's card gains a short "Serve with vLLM" section linking here.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths
from env import child_env, host_execution_policy, require_host_execution
from export_vllm import EXPORT_RECORD
from io_utils import atomic_write_text
from publish import _REPO_ID_RE, _fmt, _redact, base_model_license, upload_hub_bundle
from request import load_request
from serve_check import EXPORT_PPL_DRIFT_MAX

SERVE_REPORT = "vllm.json"
VLLM_REPO_SUFFIX = "-vllm"
CARD_SECTION_BEGIN = "<!-- quant-agent:vllm -->"
CARD_SECTION_END = "<!-- /quant-agent:vllm -->"
_LARGE_FILE_BYTES = 64 * 1024 * 1024
# Keys whose values are paths on the machine that ran the check.
_LOCAL_PATH_KEYS = {"model_dir", "baseline_dir", "model_path", "source_dir", "output_dir", "stage_dir", "source"}


def scrub_local_paths(value):
    if isinstance(value, dict):
        return {k: scrub_local_paths(v) for k, v in value.items() if k not in _LOCAL_PATH_KEYS}
    if isinstance(value, list):
        return [scrub_local_paths(v) for v in value]
    return value


def default_repo_id(base_repo_id: str) -> str:
    return base_repo_id + VLLM_REPO_SUFFIX


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" + ("" if n == 1 else "s")


def _decode_rows(side: dict) -> dict[int, dict]:
    return {int(row["batch"]): row for row in side.get("decode") or [] if "batch" in row}


def render_vllm_card(
    *,
    model_id: str,
    method_name: str,
    base_repo_id: str,
    repo_id: str,
    report: dict,
    export_record: dict,
    license_info: dict | None = None,
) -> str:
    quant = report.get("quantized") or {}
    base = report.get("fp16_baseline") or {}
    cmp_ = report.get("comparison") or {}
    gpu = quant.get("gpu_name") or "unknown GPU"
    version = quant.get("vllm_version") or "unknown"
    lic = license_info or {"license": "other"}
    license_lines = "".join(
        f"{key}: {lic[key]}\n" for key in ("license", "license_name", "license_link") if lic.get(key)
    )
    q_decode, b_decode = _decode_rows(quant), _decode_rows(base)
    decode_rows = "".join(
        f"| Decode tok/s, batch {batch} | {_fmt(q_decode[batch].get('tokens_per_s'), 1)} "
        f"| {_fmt(b_decode.get(batch, {}).get('tokens_per_s'), 1)} |\n"
        for batch in sorted(q_decode)
    )
    decode_speedups = ", ".join(
        f"{_fmt(value, 2)}× at batch {batch}"
        for batch, value in sorted((cmp_.get("decode_speedup") or {}).items(), key=lambda kv: int(kv[0]))
    )
    drift = cmp_.get("ppl_ratio_drift")
    drift_line = (
        f"- Drift against the HF benchmark ratio ({_fmt(cmp_.get('hf_ppl_ratio'), 4)}): "
        f"{_fmt(drift, 4)}. The check requires ≤ 1.02; it tests the export, not the quantization.\n"
        if drift is not None
        else "- No HF benchmark ratio was available, so export drift was not checked.\n"
    )
    if drift is not None and drift < 1 / EXPORT_PPL_DRIFT_MAX:
        drift_line += (
            "- vLLM scores better than the HF benchmark. vLLM's int8 path quantizes only "
            "the inputs of linear layers and keeps attention in fp16, so this build rounds "
            "less than the benchmarked model (for example, one that also quantized attention "
            "inputs). Same weights, not bit-identical outputs.\n"
        )
    method_tag = method_name.lower().replace(" ", "-")
    return f"""---
{license_lines}base_model: {model_id}
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- quantized
- quant-agent
- {method_tag}
- vllm
- compressed-tensors
- int8
- w8a8
---

# {method_name} {model_id} for vLLM

A native vLLM checkpoint of [`{base_repo_id}`](https://huggingface.co/{base_repo_id}),
the {method_name} quantization of [`{model_id}`](https://huggingface.co/{model_id}).
The int8 weights are the same bytes, renamed into the `compressed-tensors`
format (`int-quantized`: per-channel int8 weights, dynamic per-token int8
activations). No custom code, no `trust_remote_code`.

## Serve

```bash
pip install vllm
vllm serve {repo_id}
```

```python
from vllm import LLM, SamplingParams

llm = LLM("{repo_id}")
print(llm.generate("Hello", SamplingParams(max_tokens=32))[0].outputs[0].text)
```

## vLLM results

Measured with vLLM {version} on **{gpu}**, against vLLM fp16 of `{model_id}`.
WikiText-2 test, 2048-token windows ({quant.get("tokens_evaluated", "?")} tokens),
the same windows as the HF benchmark in `{base_repo_id}`.

| | vLLM int8 | vLLM fp16 |
| --- | ---: | ---: |
| Perplexity | {_fmt(quant.get("perplexity"), 4)} | {_fmt(base.get("perplexity"), 4)} |
| Prefill tok/s | {_fmt((quant.get("prefill") or {}).get("tokens_per_s"), 1)} | {_fmt((base.get("prefill") or {}).get("tokens_per_s"), 1)} |
{decode_rows}
- PPL ratio (int8 / fp16): {_fmt(cmp_.get("ppl_ratio"), 4)}
{drift_line}- Prefill speedup: {_fmt(cmp_.get("prefill_speedup"), 2)}×. Decode speedup: {decode_speedups or "n/a"}.

Speed numbers are from {gpu}. The HF benchmark in `{base_repo_id}` used a
different GPU and runtime, so compare speed within this table only.

## Files

- `{SERVE_REPORT}`: the full serve-check report.
- `{EXPORT_RECORD}`: how the checkpoint was converted (profile `{export_record.get("profile")}`,
  {_count(len(export_record.get("ignore") or []), "module")} kept in fp16).

## License

Inherits the license of [`{model_id}`](https://huggingface.co/{model_id})
(`{lic["license"]}`), plus the method repository license.
"""


def render_card_section(*, repo_id: str, report: dict) -> str:
    """The short block added to the base repo's card."""
    quant = report.get("quantized") or {}
    cmp_ = report.get("comparison") or {}
    decode = cmp_.get("decode_speedup") or {}
    decode_text = ", ".join(
        f"{_fmt(v, 2)}× (batch {b})" for b, v in sorted(decode.items(), key=lambda kv: int(kv[0]))
    )
    return f"""{CARD_SECTION_BEGIN}
## Serve with vLLM

A native vLLM build of these weights: [`{repo_id}`](https://huggingface.co/{repo_id}).

```bash
vllm serve {repo_id}
```

vLLM PPL ratio {_fmt(cmp_.get("ppl_ratio"), 4)} against vLLM fp16; prefill
{_fmt(cmp_.get("prefill_speedup"), 2)}×, decode {decode_text or "n/a"}. Measured on
{quant.get("gpu_name")} with vLLM {quant.get("vllm_version")}.
{CARD_SECTION_END}
"""


def merge_card_section(card: str, section: str) -> str:
    """Replace the marked section, or insert it before `## License` (else append)."""
    begin, end = card.find(CARD_SECTION_BEGIN), card.find(CARD_SECTION_END)
    if begin != -1 and end != -1 and end > begin:
        tail = card[end + len(CARD_SECTION_END):].lstrip("\n")
        return card[:begin] + section + ("\n" + tail if tail else "")
    marker = card.find("\n## License")
    if marker != -1:
        return card[: marker + 1] + section + "\n" + card[marker + 1:]
    return card.rstrip("\n") + "\n\n" + section


def stage_vllm_bundle(
    *,
    export_dir: Path,
    serve_report: dict,
    model_id: str,
    method_name: str,
    base_repo_id: str,
    repo_id: str | None = None,
    license_info: dict | None = None,
) -> dict:
    export_dir = Path(export_dir).resolve()
    record_path = export_dir / EXPORT_RECORD
    if not record_path.is_file():
        raise RuntimeError(f"{export_dir} has no {EXPORT_RECORD}; run export_vllm.py first")
    export_record = json.loads(record_path.read_text())
    if export_record.get("status") != "exported":
        raise RuntimeError(f"export status is {export_record.get('status')!r}, not exported")
    if serve_report.get("status") != "passed":
        raise RuntimeError(
            f"serve check status is {serve_report.get('status')!r} "
            f"({', '.join(serve_report.get('issue_codes') or [])}); only a passed check publishes"
        )
    if not _REPO_ID_RE.fullmatch(base_repo_id):
        raise ValueError(f"base repo id is not org/name: {base_repo_id!r}")
    repo_id = repo_id or default_repo_id(base_repo_id)
    if not _REPO_ID_RE.fullmatch(repo_id):
        raise ValueError(f"repo_id is not org/name: {repo_id!r}")
    dest = paths.OUT_ROOT / "hub" / repo_id.split("/", 1)[1]
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    copied: list[str] = []
    large: list[str] = []
    for path in sorted(export_dir.iterdir()):
        if not path.is_file() or path.name == EXPORT_RECORD or path.name.startswith("."):
            continue
        if path.stat().st_size > _LARGE_FILE_BYTES:
            # Upload from the export directly instead of duplicating it.
            large.append(str(path))
        else:
            shutil.copy2(path, dest / path.name)
        copied.append(path.name)
    report = scrub_local_paths(serve_report)
    atomic_write_text(dest / EXPORT_RECORD, json.dumps(scrub_local_paths(export_record), indent=2) + "\n", mode=0o644)
    atomic_write_text(dest / SERVE_REPORT, json.dumps(report, indent=2) + "\n", mode=0o644)
    lic = license_info or base_model_license(model_id)
    card = render_vllm_card(
        model_id=model_id,
        method_name=method_name,
        base_repo_id=base_repo_id,
        repo_id=repo_id,
        report=report,
        export_record=export_record,
        license_info=lic,
    )
    atomic_write_text(dest / "README.md", card, mode=0o644)
    manifest = {
        "status": "staged",
        "stage_dir": str(dest.resolve()),
        "weight_src": None,
        "weight_srcs": large,
        "repo_id": repo_id,
        "base_repo_id": base_repo_id,
        "copied": copied + [EXPORT_RECORD, SERVE_REPORT, "README.md"],
        "uploaded": False,
    }
    atomic_write_text(dest / "hub_manifest.json", json.dumps(manifest, indent=2) + "\n", mode=0o644)
    return manifest


def update_base_card(base_repo_id: str, section: str) -> None:
    """Add or refresh the 'Serve with vLLM' section on the base repo's README."""
    require_host_execution("huggingface hub card update")
    hf_env = child_env(include_hf=True)
    token = hf_env.get("HUGGINGFACE_HUB_TOKEN") or hf_env.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN / HUGGINGFACE_HUB_TOKEN missing; set it via quant-setup")
    from huggingface_hub import HfApi, hf_hub_download

    try:
        card = Path(hf_hub_download(base_repo_id, "README.md", repo_type="model", token=token)).read_text()
        merged = merge_card_section(card, section)
        if merged != card:
            HfApi(token=token).upload_file(
                path_or_fileobj=merged.encode(),
                path_in_repo="README.md",
                repo_id=base_repo_id,
                repo_type="model",
                commit_message="Link the vLLM build",
            )
    except Exception as exc:
        raise RuntimeError(_redact(str(exc), token)) from None


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage/upload a vLLM export as its own Hub repo")
    parser.add_argument("--export-dir", type=Path, required=True, help="quantized/<slug>-vllm")
    parser.add_argument("--serve-report", type=Path, required=True, help="jobs/<id>/vllm.json or out/serve/<slug>.json")
    parser.add_argument("--base-repo-id", required=True, help="The published native repo (org/name)")
    parser.add_argument("--request", type=Path, help="Request JSON (model_id, method_name)")
    parser.add_argument("--model-id", help="Base model id (backfill without a request)")
    parser.add_argument("--method-name", help="Method name (backfill without a request)")
    parser.add_argument("--repo-id", help="Default: <base-repo-id>-vllm")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--allow-unsafe-host-execution", action="store_true", help="Required for --upload")
    args = parser.parse_args()
    try:
        request = load_request(args.request) if args.request else {}
        model_id = args.model_id or request.get("model_id")
        method_name = args.method_name or request.get("method_name")
        if not model_id or not method_name:
            raise ValueError("need --request, or --model-id and --method-name")
        manifest = stage_vllm_bundle(
            export_dir=args.export_dir,
            serve_report=json.loads(args.serve_report.read_text()),
            model_id=str(model_id),
            method_name=str(method_name),
            base_repo_id=args.base_repo_id,
            repo_id=args.repo_id,
        )
        if args.upload:
            if not args.allow_unsafe_host_execution:
                raise RuntimeError("upload requires --allow-unsafe-host-execution")
            report = json.loads((Path(manifest["stage_dir"]) / SERVE_REPORT).read_text())
            with host_execution_policy(True):
                manifest = upload_hub_bundle(manifest, manifest["repo_id"])
                update_base_card(args.base_repo_id, render_card_section(repo_id=manifest["repo_id"], report=report))
            manifest["base_card_updated"] = True
        print(json.dumps(manifest, indent=2))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

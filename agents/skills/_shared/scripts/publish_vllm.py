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
from serve_check import EXPORT_PPL_DRIFT_MAX, dtype_label

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


def _speedup_text(decode: dict | None, fmt: str = "{v}× at batch {b}") -> str:
    items = sorted((decode or {}).items(), key=lambda kv: int(kv[0]))
    return ", ".join(fmt.format(v=_fmt(v, 2), b=b) for b, v in items) or "n/a"


def render_results_table(report: dict) -> str:
    """vLLM int8 vs the vLLM baseline, plus plain-Transformers columns when `hf_runtime` is attached."""
    ref = dtype_label(report)
    sides = [("vLLM int8", report.get("quantized") or {}), (f"vLLM {ref}", report.get("fp16_baseline") or {})]
    hf = report.get("hf_runtime")
    if isinstance(hf, dict):
        sides += [("Transformers int8", hf.get("quantized") or {}), (f"Transformers {ref}", hf.get("baseline") or {})]
    batches = sorted(_decode_rows(sides[0][1]))
    rows = [
        ("Perplexity", [_fmt(s.get("perplexity"), 4) for _, s in sides]),
        ("Prefill tok/s", [_fmt((s.get("prefill") or {}).get("tokens_per_s"), 1) for _, s in sides]),
        *(
            (f"Decode tok/s, batch {b}", [_fmt(_decode_rows(s).get(b, {}).get("tokens_per_s"), 1) for _, s in sides])
            for b in batches
        ),
    ]
    if isinstance(hf, dict):
        # vLLM preallocates its KV cache, so its peak VRAM says nothing about the weights.
        rows.append(("Peak VRAM GB", ["–", "–"] + [_fmt(s.get("peak_vram_gb"), 2) for _, s in sides[2:]]))
    head = "| | " + " | ".join(name for name, _ in sides) + " |\n"
    rule = "| --- |" + " ---: |" * len(sides) + "\n"
    return head + rule + "".join(f"| {label} | " + " | ".join(cells) + " |\n" for label, cells in rows)


def _hf_runtime_notes(report: dict, base_repo_id: str) -> str:
    hf = report.get("hf_runtime")
    if not isinstance(hf, dict):
        return ""
    ref = dtype_label(report)
    cmp_ = hf.get("comparison") or {}
    q_dtype = (hf.get("quantized") or {}).get("dtype")
    q_note = f" (in {dtype_label({'dtype': q_dtype})}, as benchmarked)" if q_dtype and q_dtype != hf.get("dtype") else ""
    return (
        f"- Transformers columns: the native artifact `{base_repo_id}` through its inference adapter{q_note}, and "
        f"`{report.get('model_id') or 'the original model'}` in {ref}, in Transformers "
        f"{hf.get('transformers_version')} (torch {hf.get('torch_version')}) on the same GPU. Transformers "
        "prefills 8 windows per forward; vLLM schedules all 32.\n"
        f"- Against Transformers {ref}: vLLM int8 prefill {_fmt(cmp_.get('vllm_prefill_speedup'), 2)}×, decode "
        f"{_speedup_text(cmp_.get('vllm_decode_speedup'))}. The native artifact in Transformers: prefill "
        f"{_fmt(cmp_.get('int8_prefill_speedup'), 2)}×, decode {_speedup_text(cmp_.get('int8_decode_speedup'))}.\n"
    )


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
    cmp_ = report.get("comparison") or {}
    ref = dtype_label(report)
    gpu = quant.get("gpu_name") or "unknown GPU"
    version = quant.get("vllm_version") or "unknown"
    lic = license_info or {"license": "other"}
    license_lines = "".join(
        f"{key}: {lic[key]}\n" for key in ("license", "license_name", "license_link") if lic.get(key)
    )
    decode_speedups = _speedup_text(cmp_.get("decode_speedup"))
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
            f"the inputs of linear layers and keeps attention in {ref}, so this build rounds "
            "less than the benchmarked model (for example, one that also quantized attention "
            "inputs). Same weights, not bit-identical outputs.\n"
        )
    method_tag = method_name.lower().replace(" ", "-")
    report = {**report, "model_id": model_id}
    if isinstance(report.get("hf_runtime"), dict):
        speed_note = (
            f"All columns are from {gpu}. The HF benchmark in `{base_repo_id}` used a different GPU, "
            "so compare its speed numbers with this table's Transformers columns, not across GPUs."
        )
    else:
        speed_note = (
            f"Speed numbers are from {gpu}. The HF benchmark in `{base_repo_id}` used a\n"
            "different GPU and runtime, so compare speed within this table only."
        )
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

Measured with vLLM {version} on **{gpu}**, against vLLM {ref} of `{model_id}`.
WikiText-2 test, 2048-token windows ({quant.get("tokens_evaluated", "?")} tokens),
the same windows as the HF benchmark in `{base_repo_id}`.

{render_results_table(report)}
- PPL ratio (int8 / {ref}): {_fmt(cmp_.get("ppl_ratio"), 4)}
{drift_line}- Prefill speedup: {_fmt(cmp_.get("prefill_speedup"), 2)}×. Decode speedup: {decode_speedups}.
{_hf_runtime_notes(report, base_repo_id)}
{speed_note}

## Files

- `{SERVE_REPORT}`: the full serve-check report.
- `{EXPORT_RECORD}`: how the checkpoint was converted (profile `{export_record.get("profile")}`,
  {_count(len(export_record.get("ignore") or []), "module")} kept in {ref}).

## License

Inherits the license of [`{model_id}`](https://huggingface.co/{model_id})
(`{lic["license"]}`), plus the method repository license.
"""


def render_card_section(
    *, repo_id: str, report: dict, base_repo_id: str | None = None, model_id: str | None = None
) -> str:
    """The block added to the base repo's card: where the vLLM build is, and its results."""
    quant = report.get("quantized") or {}
    cmp_ = report.get("comparison") or {}
    ref = dtype_label(report)
    decode_text = _speedup_text(cmp_.get("decode_speedup"), "{v}× (batch {b})")
    results = ""
    if isinstance(report.get("hf_runtime"), dict):
        int8_decode = (report["hf_runtime"].get("comparison") or {}).get("int8_decode_speedup") or {}
        slower = any(float(v) < 1 for v in int8_decode.values())
        results = f"""
## vLLM results

Measured on {quant.get("gpu_name")}: the vLLM build against vLLM {ref}, and this
repo's weights in plain Transformers against the original model.

{render_results_table({**report, "model_id": model_id})}
{_hf_runtime_notes({**report, "model_id": model_id}, base_repo_id or "this repo")}"""
        if slower:
            results += (
                f"\nFor serving, use [`{repo_id}`](https://huggingface.co/{repo_id}). In plain Transformers this "
                f"artifact decodes slower than {ref}; its gain there is memory.\n"
            )
    return f"""{CARD_SECTION_BEGIN}
## Serve with vLLM

A native vLLM build of these weights: [`{repo_id}`](https://huggingface.co/{repo_id}).

```bash
vllm serve {repo_id}
```

vLLM PPL ratio {_fmt(cmp_.get("ppl_ratio"), 4)} against vLLM {ref}; prefill
{_fmt(cmp_.get("prefill_speedup"), 2)}×, decode {decode_text}. Measured on
{quant.get("gpu_name")} with vLLM {quant.get("vllm_version")}.
{results}{CARD_SECTION_END}
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


def refresh_hub_card(manifest: dict) -> dict:
    """Re-upload the staged card and records to a repo that already holds the weights."""
    require_host_execution("huggingface hub card refresh")
    hf_env = child_env(include_hf=True)
    token = hf_env.get("HUGGINGFACE_HUB_TOKEN") or hf_env.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN / HUGGINGFACE_HUB_TOKEN missing; set it via quant-setup")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    repo_id = manifest["repo_id"]
    try:
        remote = set(api.list_repo_files(repo_id, repo_type="model"))
        missing = [Path(src).name for src in manifest.get("weight_srcs") or [] if Path(src).name not in remote]
        if missing:
            raise RuntimeError(f"{repo_id} lacks {', '.join(missing)}; publish with --upload instead")
        api.upload_folder(
            folder_path=manifest["stage_dir"],
            repo_id=repo_id,
            repo_type="model",
            ignore_patterns=["hub_manifest.json"],
            commit_message="Refresh vLLM results",
        )
    except Exception as exc:
        raise RuntimeError(_redact(str(exc), token)) from None
    return {**manifest, "uploaded": True, "refreshed": True}


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage/upload a vLLM export as its own Hub repo")
    parser.add_argument("--export-dir", type=Path, required=True, help="quantized/<slug>-vllm")
    parser.add_argument("--serve-report", type=Path, required=True, help="jobs/<id>/vllm.json or out/serve/<slug>.json")
    parser.add_argument("--base-repo-id", required=True, help="The published native repo (org/name)")
    parser.add_argument("--request", type=Path, help="Request JSON (model_id, method_name)")
    parser.add_argument("--model-id", help="Base model id (backfill without a request)")
    parser.add_argument("--method-name", help="Method name (backfill without a request)")
    parser.add_argument("--repo-id", help="Default: <base-repo-id>-vllm")
    publish = parser.add_mutually_exclusive_group()
    publish.add_argument("--upload", action="store_true")
    publish.add_argument(
        "--refresh-card", action="store_true",
        help="Already published: re-upload only the card and records, and refresh the base card",
    )
    parser.add_argument("--allow-unsafe-host-execution", action="store_true", help="Required for --upload/--refresh-card")
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
        if args.upload or args.refresh_card:
            if not args.allow_unsafe_host_execution:
                raise RuntimeError("upload requires --allow-unsafe-host-execution")
            report = json.loads((Path(manifest["stage_dir"]) / SERVE_REPORT).read_text())
            section = render_card_section(
                repo_id=manifest["repo_id"], report=report,
                base_repo_id=args.base_repo_id, model_id=str(model_id),
            )
            with host_execution_policy(True):
                if args.refresh_card:
                    manifest = refresh_hub_card(manifest)
                else:
                    manifest = upload_hub_bundle(manifest, manifest["repo_id"])
                update_base_card(args.base_repo_id, section)
            manifest["base_card_updated"] = True
        print(json.dumps(manifest, indent=2))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

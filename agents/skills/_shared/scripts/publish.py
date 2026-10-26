#!/usr/bin/env python3
"""Stage (and optionally upload) a quantized artifact + WikiText-2 metrics to the Hub.

Does not execute the model. Upload uses HF_TOKEN from the environment and never
prints it. Staging works without a token.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jobs as jobs_mod
import overlay as overlay_mod
import paths
import library as library_mod
from env import child_env, require_host_execution
from io_utils import atomic_write_text
from request import load_request
from verify import resolve_workspace_path

_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9._-]+$")
_SKIP_DIRS = {".cache", ".git"}
_SKIP_FILES = {
    "README.md",
    "model.safetensors.index.json",
    "sample_finetune.py",
    "SECURITY.md",
    "CODE_OF_CONDUCT.md",
    "data_summary_card.md",
    "flatquant_runtime.pt",
}
_SKIP_SUFFIXES = {".metadata"}
_WEIGHT_SUFFIXES = {".pt", ".pth", ".bin", ".safetensors"}
_STAGE_SUFFIXES = _WEIGHT_SUFFIXES | {".json", ".py"}
_COPY_NAMES = {
    "config.json",
    "configuration_phi3.py",
    "modeling_phi3.py",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "merges.txt",
    "vocab.json",
    "quantization_config.json",
    "flatquant_args.json",
    "LICENSE",
    "NOTICE.md",
    "packed_w4a4.pt",
    "quant_agent_inference_adapter.py",
}


def _redact(message: str, token: str | None) -> str:
    if token:
        return message.replace(token, "***")
    return message


def _read_json(path: Path) -> dict:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"expected object JSON: {path}")
    return data


def _weight_files(stage_dir: Path, weight_srcs: list[str]) -> list[str]:
    """Weight files a Hub bundle will carry: staged copies plus large originals."""
    staged = []
    if stage_dir.is_dir():
        staged = [
            path.name
            for path in stage_dir.iterdir()
            if path.is_file() and path.suffix.lower() in _WEIGHT_SUFFIXES
        ]
    external = [Path(src).name for src in weight_srcs if Path(src).is_file()]
    return sorted(set(staged + external))


def _fmt(value, digits: int = 4) -> str:
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def base_model_license(model_id: str) -> dict:
    """License fields from the base model's Hub card, so derivatives inherit them.

    `request["license"]` should be preferred by callers when set. Falls back to
    `other` when the Hub cannot be reached rather than guessing a license.
    """
    try:
        from huggingface_hub import HfApi

        hf_env = child_env(include_hf=True)
        token = hf_env.get("HUGGINGFACE_HUB_TOKEN") or hf_env.get("HF_TOKEN")
        info = HfApi(token=token or False).model_info(model_id, timeout=30)
        card = info.card_data.to_dict() if info.card_data else {}
    except Exception:
        card = {}
    out = {"license": str(card.get("license") or "other")}
    for key in ("license_name", "license_link"):
        if card.get(key):
            out[key] = str(card[key])
    if out["license"] == "other" and "license_link" not in out and model_id:
        out["license_link"] = f"https://huggingface.co/{model_id}"
    return out


def render_model_card(
    *,
    request: dict,
    job_id: str,
    benchmark: dict,
    repo_id: str | None,
    license_info: dict | None = None,
) -> str:
    method = str(request.get("method_name") or "quantization")
    model_id = str(request.get("model_id") or "")
    gpu = str(request.get("gpu_name") or request.get("gpu") or request.get("gpu_instance") or "")
    instance = str(request.get("gpu_instance") or "")
    quant = benchmark.get("quantized") if isinstance(benchmark.get("quantized"), dict) else {}
    base = benchmark.get("fp16_baseline") if isinstance(benchmark.get("fp16_baseline"), dict) else {}
    cmp_ = benchmark.get("comparison") if isinstance(benchmark.get("comparison"), dict) else {}
    hub = repo_id or f"<org>/{request.get('slug')}"
    lic = license_info or {"license": "other"}
    license_lines = "".join(
        f"{key}: {lic[key]}\n" for key in ("license", "license_name", "license_link") if lic.get(key)
    )
    return f"""---
{license_lines}base_model: {model_id}
base_model_relation: quantized
library_name: transformers
pipeline_tag: text-generation
tags:
- quantized
- quant-agent
- {method.lower().replace(" ", "-")}
- wikitext-2
---

# {method} {model_id}

Quantized derivative of [`{model_id}`](https://huggingface.co/{model_id}).
Not Hub fp16. Weights are the packed/runtime artifact from job `{job_id}`.

## WikiText-2 (test, 2048-token windows, 65504 tokens)

Measured on **{gpu}** (`{instance}`) against the original fp16 snapshot.
Numbers copied from `jobs/{job_id}/benchmark.json`.

| | Quantized | fp16 snapshot |
| --- | ---: | ---: |
| Perplexity | {_fmt(quant.get("perplexity"), 6)} | {_fmt(base.get("perplexity"), 6)} |
| NLL loss | {_fmt(quant.get("loss"), 6)} | {_fmt(base.get("loss"), 6)} |
| Tokens/s (2048 prefill, after warmup) | {_fmt(quant.get("tokens_per_s"), 1)} | {_fmt(base.get("tokens_per_s"), 1)} |
| Peak VRAM (GB) | {_fmt(quant.get("peak_vram_gb"), 3)} | {_fmt(base.get("peak_vram_gb"), 3)} |

- `ppl_ratio`: {_fmt(cmp_.get("ppl_ratio"), 6)} (`quality_ok={cmp_.get("quality_ok")}`)
- `improved_throughput`: {cmp_.get("improved_throughput")}
- `improved_vram`: {cmp_.get("improved_vram")}

## Pull

```python
from huggingface_hub import snapshot_download
path = snapshot_download("{hub}")
```

This is **not** a drop-in `AutoModelForCausalLM.from_pretrained` checkpoint.
Reload with the included `quant_agent_inference_adapter.py` after the method
repo (and overlay, if any) is on `QUANT_AGENT_METHOD_REPO`. See
`quantization_config.json`.

## License

Inherits the license of [`{model_id}`](https://huggingface.co/{model_id})
(`{lic["license"]}`), plus the method repository license. Keep `LICENSE` /
`NOTICE.md` from the snapshot when present.
"""


def stage_hub_bundle(
    *,
    job_id: str,
    request: dict,
    repo_id: str | None = None,
) -> dict:
    meta = jobs_mod.refresh_status(job_id)
    if meta.status != "completed":
        raise RuntimeError(f"publish requires a completed job, got {meta.status}")
    if str(meta.verification_status or "").lower() != "passed":
        raise RuntimeError("publish requires quant-verify passed")
    job_directory = jobs_mod.job_dir(job_id)
    benchmark_path = job_directory / "benchmark.json"
    if not benchmark_path.is_file():
        raise RuntimeError("publish requires jobs/<id>/benchmark.json")
    benchmark = _read_json(benchmark_path)
    comparison = benchmark.get("comparison") if isinstance(benchmark.get("comparison"), dict) else {}
    if not library_mod.is_beneficial(comparison):
        raise RuntimeError(
            "publish requires quality_ok and a VRAM or tok/s win versus fp16"
        )

    output_dir = resolve_workspace_path(meta.output_dir)
    if not output_dir.is_dir():
        raise RuntimeError(f"quantized output missing: {output_dir}")
    slug = paths.require_slug(str(request.get("slug") or ""))
    dest = paths.OUT_ROOT / "hub" / slug
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    large_weights: list[Path] = []
    for path in output_dir.iterdir():
        if path.name in _SKIP_DIRS or path.name in _SKIP_FILES:
            continue
        if path.suffix.lower() in _SKIP_SUFFIXES:
            continue
        if path.is_dir():
            continue
        if path.name not in _COPY_NAMES and path.suffix.lower() not in _STAGE_SUFFIXES:
            continue
        target = dest / path.name
        if path.stat().st_size > 64 * 1024 * 1024:
            # Point at the original large weight file instead of duplicating it.
            large_weights.append(path)
            copied.append(path.name)
            continue
        shutil.copy2(path, target)
        copied.append(path.name)

    overlay_dir = Path(meta.overlay_path) if meta.overlay_path else None
    adapter_text = None
    if overlay_dir and (overlay_dir / "overlay.patch").is_file():
        shutil.copy2(overlay_dir / "overlay.patch", dest / "overlay.patch")
        copied.append("overlay.patch")
        patch = (overlay_dir / "overlay.patch").read_text(errors="replace")
        adapter_text = overlay_mod.extract_new_file(patch, "quant_agent_inference_adapter.py")
    if adapter_text:
        atomic_write_text(dest / "quant_agent_inference_adapter.py", adapter_text, mode=0o644)
        copied.append("quant_agent_inference_adapter.py")

    model_id = str(request.get("model_id") or "")
    license_info = (
        {"license": str(request["license"])}
        if request.get("license")
        else base_model_license(model_id)
    )
    card = render_model_card(
        request=request,
        job_id=job_id,
        benchmark=benchmark,
        repo_id=repo_id,
        license_info=license_info,
    )
    atomic_write_text(dest / "README.md", card, mode=0o644)
    metrics = {
        "job_id": job_id,
        "method_name": request.get("method_name"),
        "model_id": request.get("model_id"),
        "gpu_instance": request.get("gpu_instance"),
        "gpu_name": request.get("gpu_name") or request.get("gpu"),
        "dataset": benchmark.get("dataset"),
        "split": benchmark.get("split"),
        "max_seq_len": benchmark.get("max_seq_len"),
        "tokens_evaluated": (benchmark.get("quantized") or {}).get("tokens_evaluated"),
        "quantized": benchmark.get("quantized"),
        "fp16_baseline": benchmark.get("fp16_baseline"),
        "comparison": comparison,
        "source": str(benchmark_path),
    }
    atomic_write_text(dest / "metrics.json", json.dumps(metrics, indent=2) + "\n", mode=0o644)
    packed = output_dir / "packed_w4a4.pt"
    if packed.is_file() and packed.resolve() not in {path.resolve() for path in large_weights}:
        large_weights.append(packed)
    weight_srcs = [str(path.resolve()) for path in large_weights]
    if not _weight_files(dest, weight_srcs):
        # A bundle without weights uploads configs only and cannot be reloaded.
        shutil.rmtree(dest)
        raise RuntimeError(
            f"no weight file ({', '.join(sorted(_WEIGHT_SUFFIXES))}) found in {output_dir}; "
            "refusing to stage a Hub bundle without weights"
        )
    weight_src = weight_srcs[0] if weight_srcs else None
    manifest = {
        "status": "staged",
        "job_id": job_id,
        "stage_dir": str(dest.resolve()),
        "output_dir": str(output_dir.resolve()),
        "weight_src": weight_src,
        "weight_srcs": weight_srcs,
        "repo_id": repo_id,
        "copied": sorted(set(copied)),
        "uploaded": False,
    }
    atomic_write_text(dest / "hub_manifest.json", json.dumps(manifest, indent=2) + "\n", mode=0o644)
    return manifest


def upload_hub_bundle(manifest: dict, repo_id: str) -> dict:
    require_host_execution("huggingface hub upload")
    if not _REPO_ID_RE.fullmatch(repo_id):
        raise ValueError(f"repo_id is not org/name: {repo_id!r}")
    weight_srcs = list(manifest.get("weight_srcs") or [])
    if manifest.get("weight_src"):
        weight_srcs.append(str(manifest["weight_src"]))
    stage_dir = manifest.get("stage_dir")
    if not stage_dir or not _weight_files(Path(stage_dir), weight_srcs):
        raise RuntimeError(
            "staged bundle has no weight file; refusing to upload a Hub repo without weights"
        )
    hf_env = child_env(include_hf=True)
    token = hf_env.get("HUGGINGFACE_HUB_TOKEN") or hf_env.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN / HUGGINGFACE_HUB_TOKEN missing. Set it in the parent "
            "environment (quant-setup); do not paste the value into chat."
        )
    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True, private=False, token=token)
        stage = Path(manifest["stage_dir"])
        api.upload_folder(
            folder_path=str(stage),
            repo_id=repo_id,
            repo_type="model",
            ignore_patterns=["hub_manifest.json"],
        )
        seen: set[str] = set()
        extra = list(manifest.get("weight_srcs") or [])
        if manifest.get("weight_src"):
            extra.insert(0, str(manifest["weight_src"]))
        for weight_src in extra:
            resolved = str(Path(weight_src).resolve())
            if resolved in seen or not Path(weight_src).is_file():
                continue
            seen.add(resolved)
            api.upload_file(
                path_or_fileobj=weight_src,
                path_in_repo=Path(weight_src).name,
                repo_id=repo_id,
                repo_type="model",
            )
    except Exception as exc:
        raise RuntimeError(_redact(str(exc), token)) from None
    payload = {
        **manifest,
        "status": "uploaded",
        "repo_id": repo_id,
        "uploaded": True,
        "hub_url": f"https://huggingface.co/{repo_id}",
    }
    atomic_write_text(
        Path(manifest["stage_dir"]) / "hub_manifest.json",
        json.dumps(payload, indent=2) + "\n",
        mode=0o644,
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage/upload quantized weights + WikiText-2 metrics")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--repo-id", help="Hugging Face org/name")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument(
        "--allow-unsafe-host-execution",
        action="store_true",
        help="Required for Hub upload (not for staging)",
    )
    args = parser.parse_args()
    try:
        request = load_request(args.request)
        manifest = stage_hub_bundle(job_id=args.job_id, request=request, repo_id=args.repo_id)
        if args.upload:
            if not args.repo_id:
                raise ValueError("--repo-id is required for --upload")
            if not args.allow_unsafe_host_execution:
                raise RuntimeError("upload requires --allow-unsafe-host-execution")
            from env import host_execution_policy

            with host_execution_policy(True):
                manifest = upload_hub_bundle(manifest, args.repo_id)
            hub_url = manifest.get("hub_url")
            if hub_url:
                manifest["hub_url"] = hub_url
        print(json.dumps(manifest, indent=2))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

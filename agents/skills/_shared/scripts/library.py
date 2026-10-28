#!/usr/bin/env python3
"""Filterable catalog of quantized methods (model × method × GPU instance).

Library users filter `library/catalog.json`, compare WikiText-2 metrics, then
fetch weights themselves from Hugging Face (`hub_repo_id`). This file is the
table. Hub is only the weight store.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jobs as jobs_mod
import paths
from io_utils import atomic_write_text
from request import load_request
from verify import resolve_workspace_path

# 2: optional per-run `vllm` block (serve-check results for the vLLM export).
SCHEMA_VERSION = 2
QUALITY_PPL_RATIO_MAX = 1.5
VLLM_STATUSES = ("passed", "failed", "unsupported", "pending")
VLLM_PPL_DRIFT_MAX = 1.02
COLLECTION_TITLE = "quant-agent library"
COLLECTION_DESCRIPTION = (
    "One library of quantized LLMs measured by quant-agent. "
    "Rank WikiText-2 in the GitHub library/ catalog, then download these repos."
)
ATTEMPT_STOP_REASONS = (
    "budget_exhausted",
    "quality_failed",
    "not_beneficial",
    "verify_failed",
    "process_failed",
    "oom_same_config",
    "auth_or_disk",
)
_EXCERPT_MAX = 700
_NOTE_MAX = 240
_NOTES_MAX = 8
_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_HUB_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9._-]+$")
_GITHUB_REPO_RE = re.compile(r"^https://github.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_REQUIRED_NUMS = (
    "ppl",
    "ppl_ratio",
    "tokens_per_s",
    "peak_vram_gb",
    "fp16_ppl",
    "fp16_tokens_per_s",
    "fp16_peak_vram_gb",
)


def group_key(model_id: str, gpu_instance: str) -> str:
    return f"{model_id.strip()}|{gpu_instance.strip()}"


def group_filename(model_id: str, gpu_instance: str) -> str:
    model = _SAFE_RE.sub("_", model_id.strip()).strip("._-") or "model"
    gpu = _SAFE_RE.sub("_", gpu_instance.strip()).strip("._-") or "gpu"
    return f"{model}__{gpu}.json"


def contribution_filename(model_id: str, gpu_instance: str, method_name: str) -> str:
    model = _SAFE_RE.sub("_", model_id.strip()).strip("._-") or "model"
    gpu = _SAFE_RE.sub("_", gpu_instance.strip()).strip("._-") or "gpu"
    method = _SAFE_RE.sub("_", method_name.strip()).strip("._-") or "method"
    return f"{model}__{gpu}__{method}.json"


def contributions_dir() -> Path:
    path = paths.LIBRARY_ROOT / "contributions"
    path.mkdir(parents=True, exist_ok=True)
    return path


def attempts_dir() -> Path:
    path = paths.LIBRARY_ROOT / "attempts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def attempt_path(model_id: str, gpu_instance: str, method_name: str) -> Path:
    return attempts_dir() / contribution_filename(model_id, gpu_instance, method_name)


def clear_attempt(model_id: str, gpu_instance: str, method_name: str) -> None:
    dest = attempt_path(model_id, gpu_instance, method_name)
    if dest.is_file():
        dest.unlink()


def _read_json(path: Path) -> dict:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"expected object JSON: {path}")
    return data


def _packed_artifact(output_dir: Path) -> bool:
    cfg = output_dir / "quantization_config.json"
    if cfg.is_file():
        try:
            payload = json.loads(cfg.read_text())
            fmt = str(payload.get("format") or "").lower()
            if "int4" in fmt or "packed" in fmt:
                return True
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    return any(
        path.is_file() and ("packed" in path.name.lower() or "int4" in path.name.lower())
        for path in output_dir.glob("*")
    )


def rank_tuple(row: dict) -> tuple:
    """Higher is better. quality_ok first, then tok/s, then lower VRAM, packed, lower PPL ratio."""
    quality = 1 if row.get("quality_ok") is True else 0
    packed = 1 if row.get("packed") else 0
    tps = row.get("tokens_per_s")
    tps_v = float(tps) if isinstance(tps, (int, float)) else -1.0
    vram = row.get("peak_vram_gb")
    vram_v = -float(vram) if isinstance(vram, (int, float)) else -1e9
    ratio = row.get("ppl_ratio")
    ratio_v = -float(ratio) if isinstance(ratio, (int, float)) else -1e9
    return (quality, tps_v, vram_v, packed, ratio_v)


def rank_methods(methods: list[dict]) -> list[dict]:
    ordered = sorted(methods, key=rank_tuple, reverse=True)
    ranked: list[dict] = []
    for index, row in enumerate(ordered, start=1):
        item = dict(row)
        item["rank"] = index
        ranked.append(item)
    return ranked


def pick_best(methods: list[dict]) -> dict | None:
    usable = [row for row in methods if row.get("quality_ok") is True]
    pool = usable or []
    if not pool:
        return None
    return max(pool, key=rank_tuple)


def empty_group(model_id: str, gpu_instance: str, gpu_name: str | None = None) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_id": model_id,
        "gpu_instance": gpu_instance,
        "gpu_name": gpu_name,
        "fp16_baseline": None,
        "methods": [],
        "best": None,
    }


def load_group(model_id: str, gpu_instance: str) -> dict:
    path = paths.LIBRARY_ROOT / "groups" / group_filename(model_id, gpu_instance)
    if not path.is_file():
        return empty_group(model_id, gpu_instance)
    data = _read_json(path)
    data.setdefault("methods", [])
    return data


def _write_group(group: dict) -> Path:
    paths.LIBRARY_ROOT.mkdir(parents=True, exist_ok=True)
    group_dir = paths.LIBRARY_ROOT / "groups"
    group_dir.mkdir(parents=True, exist_ok=True)
    methods = rank_methods(list(group.get("methods") or []))
    group["methods"] = methods
    best = pick_best(methods)
    if best is None:
        group["best"] = None
    else:
        group["best"] = {
            "method_name": best.get("method_name"),
            "slug": best.get("slug"),
            "job_id": best.get("job_id"),
            "rank": best.get("rank"),
            "why": (
                "quality_ok, then higher WikiText-2 tok/s, then lower peak VRAM, "
                "prefer packed, then lower ppl_ratio"
            ),
            "artifact_dir": best.get("artifact_dir"),
            "hub_url": best.get("hub_url"),
        }
    dest = group_dir / group_filename(str(group["model_id"]), str(group["gpu_instance"]))
    atomic_write_text(dest, json.dumps(group, indent=2) + "\n", mode=0o644)
    rebuild_derived()
    return dest


def _index_path() -> Path:
    return paths.LIBRARY_ROOT / "index.json"


def _write_index() -> Path:
    entries: list[dict] = []
    for group in load_boards():
        methods = rank_methods(list(group.get("methods") or []))
        best = pick_best(methods)
        best = best if isinstance(best, dict) else {}
        entries.append(
            {
                "model_id": group.get("model_id"),
                "gpu_instance": group.get("gpu_instance"),
                "gpu_name": group.get("gpu_name"),
                "n_methods": len(methods),
                "best_method": best.get("method_name"),
            }
        )
    index = {"schema_version": SCHEMA_VERSION, "groups": entries}
    dest = _index_path()
    atomic_write_text(dest, json.dumps(index, indent=2) + "\n", mode=0o644)
    return dest


def _hub_repo_id(hub_url: str | None) -> str | None:
    if not hub_url:
        return None
    text = str(hub_url).strip().rstrip("/")
    marker = "huggingface.co/"
    if marker in text:
        text = text.split(marker, 1)[1]
    text = text.removeprefix("models/")
    if text.count("/") != 1:
        return None
    return text


def paper_url(arxiv_id: str | None, explicit: str | None = None) -> str | None:
    url = str(explicit or "").strip()
    if url.startswith("http"):
        return url
    aid = str(arxiv_id or "").strip().removeprefix("arXiv:").removeprefix("arxiv:")
    if not aid:
        return None
    return f"https://arxiv.org/abs/{aid}"


def source_links(request: dict, extra: dict | None = None) -> dict:
    extra = extra or {}
    arxiv = str(extra.get("arxiv_id") or request.get("arxiv_id") or "").strip()
    repo = str(extra.get("repo_url") or request.get("repo_url") or "").strip().rstrip("/")
    commit = str(extra.get("repo_commit") or request.get("repo_commit") or "").strip()
    return {
        "arxiv_id": arxiv or None,
        "paper_url": paper_url(arxiv, extra.get("paper_url") or request.get("paper_url")),
        "repo_url": repo or None,
        "repo_commit": commit or None,
    }


def is_beneficial(comparison: dict) -> bool:
    return comparison.get("quality_ok") is True and (
        comparison.get("improved_vram") is True or comparison.get("improved_throughput") is True
    )


def _eq(value, want: str | None) -> bool:
    if want is None or want == "":
        return True
    return str(value or "").strip().lower() == want.strip().lower()


def _finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate_contribution(data: dict, *, require_hub: bool = True) -> dict:
    if not isinstance(data, dict):
        raise ValueError("contribution must be a JSON object")
    for key in ("model_id", "method_name", "gpu_instance"):
        if not str(data.get(key) or "").strip():
            raise ValueError(f"contribution missing {key}")
    if data.get("quality_ok") is not True:
        raise ValueError("only quality_ok runs can be contributed")
    for key in _REQUIRED_NUMS:
        if not _finite_number(data.get(key)):
            raise ValueError(f"contribution missing numeric {key}")
    ratio = float(data["ppl_ratio"])
    expected = float(data["ppl"]) / float(data["fp16_ppl"])
    if expected <= 0 or abs(ratio - expected) / expected > 0.02:
        raise ValueError("ppl_ratio must match ppl / fp16_ppl")
    if ratio > QUALITY_PPL_RATIO_MAX:
        raise ValueError(f"ppl_ratio must be <= {QUALITY_PPL_RATIO_MAX}")
    fp16_tps = float(data["fp16_tokens_per_s"])
    fp16_vram = float(data["fp16_peak_vram_gb"])
    tps = float(data["tokens_per_s"])
    vram = float(data["peak_vram_gb"])
    if tps <= fp16_tps and vram >= fp16_vram:
        raise ValueError("contribution must beat fp16 VRAM or tok/s")
    dataset = str(data.get("dataset") or "wikitext-2-raw-v1")
    if "wikitext-2" not in dataset.lower():
        raise ValueError("contributions must use WikiText-2")
    if data.get("max_seq_len") not in (None, 2048):
        raise ValueError("contributions must use 2048-token windows")
    repo = str(data.get("hub_repo_id") or "").strip() or _hub_repo_id(
        data.get("hub_url") if isinstance(data.get("hub_url"), str) else None
    )
    if require_hub:
        if not repo or not _HUB_REPO_RE.fullmatch(repo):
            raise ValueError("contribution needs public hub_repo_id org/name so others can fetch")
        data = dict(data)
        data["hub_repo_id"] = repo
        data["hub_url"] = data.get("hub_url") or f"https://huggingface.co/{repo}"
    links = source_links({}, data)
    if not links["paper_url"]:
        raise ValueError("contribution needs paper_url or arxiv_id")
    repo_url = links["repo_url"]
    if not repo_url or not _GITHUB_REPO_RE.fullmatch(repo_url):
        raise ValueError("contribution needs repo_url https://github.com/owner/repo")
    data = dict(data)
    data["arxiv_id"] = links["arxiv_id"]
    data["paper_url"] = links["paper_url"]
    data["repo_url"] = repo_url
    if links["repo_commit"]:
        data["repo_commit"] = links["repo_commit"]
    if data.get("vllm") is not None:
        data["vllm"] = validate_vllm_block(data["vllm"])
    return data


def validate_vllm_block(block: dict) -> dict:
    """The optional serve-check summary on a run. Never affects ranking."""
    if not isinstance(block, dict):
        raise ValueError("vllm must be an object")
    status = block.get("status")
    if status not in VLLM_STATUSES:
        raise ValueError(f"vllm.status must be one of {', '.join(VLLM_STATUSES)}")
    if status == "unsupported" and not str(block.get("reason_code") or "").strip():
        raise ValueError("vllm.status unsupported needs reason_code")
    if status == "failed" and not block.get("issue_codes"):
        raise ValueError("vllm.status failed needs issue_codes")
    repo = block.get("hub_repo_id")
    if repo is not None and not _HUB_REPO_RE.fullmatch(str(repo)):
        raise ValueError("vllm.hub_repo_id must be org/name")
    if status == "passed":
        for key in ("vllm_version", "gpu_name"):
            if not str(block.get(key) or "").strip():
                raise ValueError(f"vllm.status passed needs {key}")
        ratio = block.get("ppl_ratio")
        if not _finite_number(ratio) or float(ratio) > QUALITY_PPL_RATIO_MAX:
            raise ValueError(f"vllm.status passed needs ppl_ratio <= {QUALITY_PPL_RATIO_MAX}")
        drift = block.get("ppl_ratio_drift")
        if drift is not None and (not _finite_number(drift) or float(drift) > VLLM_PPL_DRIFT_MAX):
            raise ValueError(f"vllm.status passed needs ppl_ratio_drift <= {VLLM_PPL_DRIFT_MAX}")
    elif repo is not None:
        raise ValueError("only a passed vLLM check may link a vLLM Hub repo")
    return dict(block)


def vllm_block(
    export_record: dict,
    serve_report: dict | None = None,
    *,
    hub_repo_id: str | None = None,
    checked_at: str | None = None,
) -> dict:
    """Summarize `vllm_export.json` + the serve-check report into the library block."""
    if export_record.get("status") == "unsupported":
        block = {"status": "unsupported", "reason_code": export_record.get("reason_code")}
    elif export_record.get("status") != "exported":
        raise ValueError(f"export record status is {export_record.get('status')!r}")
    elif not serve_report:
        block = {"status": "pending", "profile": export_record.get("profile")}
    else:
        comparison = serve_report.get("comparison") or {}
        quantized = serve_report.get("quantized") or {}
        block = {
            "status": serve_report.get("status"),
            "issue_codes": list(serve_report.get("issue_codes") or []) or None,
            "profile": export_record.get("profile"),
            "vllm_version": quantized.get("vllm_version"),
            "gpu_name": quantized.get("gpu_name"),
            "ppl_ratio": comparison.get("ppl_ratio"),
            "hf_ppl_ratio": comparison.get("hf_ppl_ratio"),
            "ppl_ratio_drift": comparison.get("ppl_ratio_drift"),
            "prefill_speedup": comparison.get("prefill_speedup"),
            "decode_speedup": comparison.get("decode_speedup"),
            "checked_at": checked_at,
        }
        if hub_repo_id:
            block["hub_repo_id"] = hub_repo_id
            block["hub_url"] = f"https://huggingface.co/{hub_repo_id}"
    return validate_vllm_block({k: v for k, v in block.items() if v is not None})


def record_vllm(
    *,
    model_id: str,
    gpu_instance: str,
    method_name: str,
    export_record: dict,
    serve_report: dict | None = None,
    hub_repo_id: str | None = None,
) -> dict:
    """Attach the vLLM result to an existing library contribution and rebuild."""
    path = contributions_dir() / contribution_filename(model_id, gpu_instance, method_name)
    if not path.is_file():
        raise RuntimeError(f"no library row for {model_id} × {gpu_instance} × {method_name} ({path})")
    data = _read_json(path)
    data["vllm"] = vllm_block(
        export_record,
        serve_report,
        hub_repo_id=hub_repo_id,
        checked_at=datetime.now(timezone.utc).date().isoformat(),
    )
    data["schema_version"] = SCHEMA_VERSION
    payload = validate_contribution(data, require_hub=True)
    atomic_write_text(path, json.dumps(payload, indent=2) + "\n", mode=0o644)
    rebuild_derived()
    return {"status": "recorded", "contribution": str(path), "vllm": payload["vllm"]}


def _method_from_contribution(data: dict) -> dict:
    hub_url = data.get("hub_url")
    repo = data.get("hub_repo_id") or _hub_repo_id(hub_url if isinstance(hub_url, str) else None)
    if repo and not hub_url:
        hub_url = f"https://huggingface.co/{repo}"
    return {
        "method_name": data.get("method_name"),
        "slug": data.get("slug"),
        "job_id": data.get("job_id"),
        "quality_ok": data.get("quality_ok") is True,
        "ppl": data.get("ppl"),
        "ppl_ratio": data.get("ppl_ratio"),
        "tokens_per_s": data.get("tokens_per_s"),
        "peak_vram_gb": data.get("peak_vram_gb"),
        "improved_throughput": data.get("improved_throughput"),
        "improved_vram": data.get("improved_vram"),
        "packed": bool(data.get("packed")),
        "artifact_dir": data.get("artifact_dir"),
        "hub_url": hub_url,
        "arxiv_id": data.get("arxiv_id"),
        "paper_url": data.get("paper_url"),
        "repo_url": data.get("repo_url"),
        "repo_commit": data.get("repo_commit"),
        "verification_status": data.get("verification_status"),
        "vllm": data.get("vllm"),
    }


def _apply_contribution(boards: dict[str, dict], data: dict) -> None:
    model_id = str(data["model_id"])
    gpu_instance = str(data["gpu_instance"])
    key = group_key(model_id, gpu_instance)
    group = boards.get(key) or empty_group(model_id, gpu_instance, data.get("gpu_name"))
    if data.get("gpu_name"):
        group["gpu_name"] = data.get("gpu_name")
    if not group.get("fp16_baseline"):
        group["fp16_baseline"] = {
            "perplexity": data.get("fp16_ppl"),
            "tokens_per_s": data.get("fp16_tokens_per_s"),
            "peak_vram_gb": data.get("fp16_peak_vram_gb"),
        }
    method_row = _method_from_contribution(data)
    by_method = {str(old.get("method_name")): dict(old) for old in group.get("methods") or []}
    name = str(method_row.get("method_name"))
    prev = by_method.get(name, {})
    merged = {**prev, **{k: v for k, v in method_row.items() if v is not None}}
    if not merged.get("hub_url"):
        merged["hub_url"] = prev.get("hub_url")
    by_method[name] = merged
    group["methods"] = list(by_method.values())
    boards[key] = group


def load_boards() -> list[dict]:
    boards: dict[str, dict] = {}
    groups_dir = paths.LIBRARY_ROOT / "groups"
    if groups_dir.is_dir():
        for path in sorted(groups_dir.glob("*.json")):
            group = _read_json(path)
            key = group_key(str(group.get("model_id") or ""), str(group.get("gpu_instance") or ""))
            if key == "|":
                continue
            boards[key] = group
    contrib_dir = paths.LIBRARY_ROOT / "contributions"
    if contrib_dir.is_dir():
        for path in sorted(contrib_dir.glob("*.json")):
            data = validate_contribution(_read_json(path), require_hub=True)
            _apply_contribution(boards, data)
    return list(boards.values())


def catalog_rows() -> list[dict]:
    rows: list[dict] = []
    for group in load_boards():
        methods = rank_methods(list(group.get("methods") or []))
        best = pick_best(methods)
        best_name = best.get("method_name") if best else None
        fp16 = group.get("fp16_baseline") if isinstance(group.get("fp16_baseline"), dict) else {}
        for method in methods:
            hub_url = method.get("hub_url")
            repo = _hub_repo_id(hub_url if isinstance(hub_url, str) else None)
            rows.append(
                {
                    "model_id": group.get("model_id"),
                    "method_name": method.get("method_name"),
                    "gpu_instance": group.get("gpu_instance"),
                    "gpu_name": group.get("gpu_name"),
                    "rank": method.get("rank"),
                    "is_best": (
                        method.get("method_name") == best_name
                        and method.get("quality_ok") is True
                    ),
                    "quality_ok": method.get("quality_ok"),
                    "ppl": method.get("ppl"),
                    "ppl_ratio": method.get("ppl_ratio"),
                    "tokens_per_s": method.get("tokens_per_s"),
                    "peak_vram_gb": method.get("peak_vram_gb"),
                    "fp16_ppl": fp16.get("perplexity"),
                    "fp16_tokens_per_s": fp16.get("tokens_per_s"),
                    "fp16_peak_vram_gb": fp16.get("peak_vram_gb"),
                    "packed": method.get("packed"),
                    "hub_url": hub_url,
                    "hub_repo_id": repo,
                    "arxiv_id": method.get("arxiv_id"),
                    "paper_url": method.get("paper_url"),
                    "repo_url": method.get("repo_url"),
                    "repo_commit": method.get("repo_commit"),
                    "artifact_dir": method.get("artifact_dir"),
                    "job_id": method.get("job_id"),
                    "slug": method.get("slug"),
                    "vllm": method.get("vllm"),
                }
            )
    return rows


def query(
    *,
    model_id: str | None = None,
    method_name: str | None = None,
    gpu_instance: str | None = None,
    best_only: bool = False,
) -> list[dict]:
    rows = [
        row
        for row in catalog_rows()
        if _eq(row.get("model_id"), model_id)
        and _eq(row.get("method_name"), method_name)
        and _eq(row.get("gpu_instance"), gpu_instance)
    ]
    if best_only:
        rows = [row for row in rows if row.get("is_best")]
    return rows


def fetch_howto(row: dict) -> dict:
    repo = row.get("hub_repo_id") or _hub_repo_id(
        row.get("hub_url") if isinstance(row.get("hub_url"), str) else None
    )
    if repo:
        local = repo.replace("/", "__")
        return {
            "status": "ok",
            "hub_repo_id": repo,
            "command": f"huggingface-cli download {repo} --local-dir ./{local}",
            "python": f'from huggingface_hub import snapshot_download\nsnapshot_download("{repo}")',
        }
    return {
        "status": "local_only",
        "artifact_dir": row.get("artifact_dir"),
        "message": (
            "No Hub upload for this row yet. Copy artifact_dir, or run "
            "quant-publish --upload, then fetch with huggingface-cli."
        ),
    }


def library_meta_path() -> Path:
    return paths.LIBRARY_ROOT / "library.json"


def load_library_meta() -> dict:
    path = library_meta_path()
    if not path.is_file():
        return {}
    data = _read_json(path)
    return data if isinstance(data, dict) else {}


def collection_url(slug: str) -> str:
    return f"https://huggingface.co/collections/{slug}"


def collection_entries(rows: list[dict] | None = None) -> list[dict]:
    """One Hub collection item per public quality_ok repo. Weights may live on any account."""
    entries: list[dict] = []
    seen: set[str] = set()
    for row in rows if rows is not None else catalog_rows():
        repo = row.get("hub_repo_id") or _hub_repo_id(
            row.get("hub_url") if isinstance(row.get("hub_url"), str) else None
        )
        if not repo or repo in seen or row.get("quality_ok") is not True:
            continue
        seen.add(repo)
        note = (
            f"{row.get('method_name')} · {row.get('model_id')} · "
            f"{row.get('gpu_instance')} · WikiText-2 ppl_ratio={_num(row.get('ppl_ratio'), 4)}"
        )[:500]
        entries.append({"item_id": repo, "item_type": "model", "note": note})
        vllm = row.get("vllm") if isinstance(row.get("vllm"), dict) else {}
        vllm_repo = vllm.get("hub_repo_id")
        if vllm.get("status") == "passed" and vllm_repo and vllm_repo not in seen:
            seen.add(vllm_repo)
            vllm_note = (
                f"vLLM build of {repo} · {row.get('method_name')} · {row.get('model_id')} · "
                f"vLLM ppl_ratio={_num(vllm.get('ppl_ratio'), 4)} on {vllm.get('gpu_name')}"
            )[:500]
            entries.append({"item_id": vllm_repo, "item_type": "model", "note": vllm_note})
    return entries


def _write_library_meta(meta: dict) -> Path:
    dest = library_meta_path()
    paths.LIBRARY_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_write_text(dest, json.dumps(meta, indent=2) + "\n", mode=0o644)
    return dest


def sync_hf_collection(*, collection_slug: str | None = None) -> dict:
    """Add every catalog Hub repo to one public collection. Does not move weights."""
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN missing. Load .env in the parent shell; do not paste the value."
        )
    from huggingface_hub import (
        HfApi,
        add_collection_item,
        create_collection,
        get_collection,
        list_collections,
        update_collection_metadata,
    )

    def _redact(message: str) -> str:
        return message.replace(token, "***")

    meta = load_library_meta()
    slug = (
        (collection_slug or os.environ.get("QUANT_AGENT_HF_COLLECTION") or meta.get("hf_collection_slug") or "")
        .strip()
        or None
    )
    entries = collection_entries()
    try:
        if not slug:
            owner = str((HfApi(token=token).whoami() or {}).get("name") or "").strip()
            if not owner:
                raise RuntimeError("Hugging Face whoami returned no username")
            for col in list_collections(owner=owner, token=token):
                if getattr(col, "title", None) == COLLECTION_TITLE and getattr(col, "slug", None):
                    slug = col.slug
                    break
            if not slug:
                created = create_collection(
                    title=COLLECTION_TITLE,
                    namespace=owner,
                    description=COLLECTION_DESCRIPTION,
                    private=False,
                    exists_ok=True,
                    token=token,
                )
                slug = created.slug
        update_collection_metadata(
            slug,
            title=COLLECTION_TITLE,
            description=COLLECTION_DESCRIPTION,
            token=token,
        )
        collection = get_collection(slug, token=token)
        existing = {
            item.item_id
            for item in (collection.items or [])
            if getattr(item, "item_type", None) == "model"
        }
        added: list[str] = []
        for entry in entries:
            if entry["item_id"] in existing:
                continue
            add_collection_item(
                slug,
                entry["item_id"],
                "model",
                note=entry["note"],
                exists_ok=True,
                token=token,
            )
            added.append(entry["item_id"])
    except Exception as exc:
        raise RuntimeError(_redact(str(exc))) from None
    payload = {
        "schema_version": SCHEMA_VERSION,
        "title": COLLECTION_TITLE,
        "hf_collection_slug": slug,
        "hf_collection_url": collection_url(str(slug)),
        "status": "synced",
        "added": added,
        "n_items": len(entries),
    }
    _write_library_meta(
        {
            "schema_version": SCHEMA_VERSION,
            "title": COLLECTION_TITLE,
            "hf_collection_slug": slug,
            "hf_collection_url": payload["hf_collection_url"],
        }
    )
    rebuild_derived()
    return payload


def rebuild_derived() -> dict:
    index = _write_index()
    catalog = _write_catalog()
    readme = _write_markdown()
    attempts = _write_attempts()
    return {
        "index": str(index),
        "catalog": str(catalog),
        "library_md": str(readme),
        "attempts": str(attempts["index"]),
        "attempts_md": str(attempts["markdown"]),
    }


def _write_catalog() -> Path:
    dest = paths.LIBRARY_ROOT / "catalog.json"
    payload = {"schema_version": SCHEMA_VERSION, "rows": catalog_rows()}
    atomic_write_text(dest, json.dumps(payload, indent=2) + "\n", mode=0o644)
    return dest


def _write_markdown() -> Path:
    meta = load_library_meta()
    collection = meta.get("hf_collection_url")
    lines = [
        "# Library",
        "",
        "Measured quantized runs. One row per **method × model × GPU**, ranked",
        "on WikiText-2 against the original fp16 checkpoint (PPL, tok/s, VRAM).",
        "",
        "Each heading is one published board. New runs can use any model;",
        "Hugging Face is where **weights** live, not which models are allowed.",
        "",
        "This table is the ranking. Weights live on Hugging Face.",
        "",
    ]
    if collection:
        lines.extend(
            [
                f"Hugging Face collection: {collection}",
                "",
            ]
        )
    lines.extend(
        [
            "Rank: `quality_ok`, then tokens/s, then lower VRAM.",
            "vLLM: a native vLLM build that passed the serve check (linked), or why",
            "there is none. It never changes the ranking; its speedups are measured",
            "on the GPU named under the table, not the board's GPU.",
            "Unsuccessful stops stay off this ranking. See [ATTEMPTS.md](ATTEMPTS.md).",
            "Do not edit this file by hand. See the repo README to add a row.",
            "",
        ]
    )
    for group in load_boards():
        methods = rank_methods(list(group.get("methods") or []))
        best = pick_best(methods) or {}
        lines.append(f"## {group.get('model_id')} on {group.get('gpu_instance')}")
        if group.get("gpu_name"):
            lines.append(f"GPU: {group['gpu_name']}")
        lines.append("")
        lines.append(
            "| Model | GPU | Method | Paper | Repo | Hugging Face | vLLM | PPL ratio | tok/s | VRAM GB |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |")
        for row in methods:
            lines.append(
                "| {model} | {gpu} | {method} | {paper} | {repo} | {hub} | {vllm} | {ratio} | {tps} | {vram} |".format(
                    model=group.get("model_id") or "",
                    gpu=group.get("gpu_instance") or "",
                    method=row.get("method_name") or "",
                    paper=_md_link("paper", row.get("paper_url")),
                    repo=_md_link("repo", row.get("repo_url")),
                    hub=_md_link("hub", row.get("hub_url")),
                    vllm=_vllm_cell(row.get("vllm")),
                    ratio=_num(row.get("ppl_ratio"), 4),
                    tps=_num(row.get("tokens_per_s"), 1),
                    vram=_num(row.get("peak_vram_gb"), 3),
                )
            )
        if best.get("method_name"):
            lines.append("")
            lines.append(f"**Best:** {best['method_name']} (`{best.get('slug') or best.get('method_name')}`).")
        vllm_lines = [_vllm_detail(row) for row in methods]
        vllm_lines = [line for line in vllm_lines if line]
        if vllm_lines:
            lines.append("")
            lines.extend(vllm_lines)
        lines.append("")
    dest = paths.LIBRARY_ROOT / "LIBRARY.md"
    atomic_write_text(dest, "\n".join(lines) + "\n", mode=0o644)
    return dest


def _vllm_cell(block) -> str:
    if not isinstance(block, dict):
        return ""
    status = block.get("status")
    if status == "passed":
        return _md_link("vLLM", block.get("hub_url")) or "passed"
    if status == "unsupported":
        return f"no (`{block.get('reason_code')}`)"
    if status == "failed":
        return f"failed (`{', '.join(block.get('issue_codes') or [])}`)"
    return str(status or "")


def _vllm_detail(row: dict) -> str | None:
    block = row.get("vllm")
    if not isinstance(block, dict) or block.get("status") != "passed":
        return None
    decode = block.get("decode_speedup") or {}
    decode_text = ", ".join(
        f"{_num(value, 2)}× batch {batch}" for batch, value in sorted(decode.items(), key=lambda kv: int(kv[0]))
    )
    return (
        f"- vLLM {row.get('method_name')}: PPL ratio {_num(block.get('ppl_ratio'), 4)} vs vLLM fp16"
        f" (drift {_num(block.get('ppl_ratio_drift'), 4)} vs this board), prefill "
        f"{_num(block.get('prefill_speedup'), 2)}×, decode {decode_text or 'n/a'}. "
        f"Measured on {block.get('gpu_name')} with vLLM {block.get('vllm_version')}"
        f"{', ' + block['checked_at'] if block.get('checked_at') else ''}."
    )


def _md_link(label: str, url) -> str:
    text = str(url or "").strip()
    if not text:
        return ""
    return f"[{label}]({text})"


def _num(value, digits: int) -> str:
    if not isinstance(value, (int, float)):
        return ""
    return f"{float(value):.{digits}f}"


def _public_artifact_dir(output_dir: Path) -> str:
    resolved = output_dir.expanduser().resolve()
    try:
        return str(resolved.relative_to(paths.REPO_ROOT))
    except ValueError:
        return str(resolved)


def _clip(text, limit: int) -> str | None:
    value = str(text or "").strip()
    if not value:
        return None
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)] + "…"


def _clip_tail(text, limit: int) -> str | None:
    value = str(text or "").strip()
    if not value:
        return None
    if len(value) <= limit:
        return value
    return "…" + value[-(limit - 1) :]


def _clean_excerpt(text) -> str | None:
    raw = str(text or "")
    kept: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if "%|" in stripped and ("it/s" in stripped or "layer]" in stripped or "?it/s" in stripped):
            continue
        kept.append(line)
    return _clip_tail("\n".join(kept), _EXCERPT_MAX)


def _clip_notes(notes) -> list[str]:
    if not isinstance(notes, list):
        return []
    clipped: list[str] = []
    for item in notes:
        text = _clip(item, _NOTE_MAX)
        if text:
            clipped.append(text)
        if len(clipped) >= _NOTES_MAX:
            break
    return clipped


def _load_json_if_present(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {}
    try:
        data = _read_json(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _job_benchmark(job_id: str | None) -> dict:
    if not job_id or not jobs_mod.valid_job_id(str(job_id)):
        return {}
    try:
        return _load_json_if_present(jobs_mod.job_dir(str(job_id)) / "benchmark.json")
    except Exception:
        return {}


def _job_diagnose(job_id: str | None, request: dict | None = None) -> dict:
    request = request or {}
    if job_id and jobs_mod.valid_job_id(str(job_id)):
        try:
            payload = _load_json_if_present(jobs_mod.job_dir(str(job_id)) / "diagnose.json")
            if payload:
                return payload
        except Exception:
            pass
    raw = str(request.get("last_diagnose_path") or "").strip()
    if not raw:
        return {}
    path = Path(raw)
    if not path.is_absolute():
        path = paths.REPO_ROOT / path
    return _load_json_if_present(path)


def classify_attempt_stop(
    *,
    request: dict,
    diagnose: dict,
    comparison: dict,
    meta,
) -> str:
    issues = [
        str(item)
        for item in (diagnose.get("issue_codes") or diagnose.get("issues") or [])
        if str(item).strip()
    ]
    notes = " ".join(str(item) for item in (diagnose.get("notes") or []))
    retry = diagnose.get("retry") if isinstance(diagnose.get("retry"), dict) else {}
    remaining = retry.get("remaining")
    if remaining is None:
        used = int(request.get("retry_gpu_jobs_used") or 0)
        maximum = int(request.get("retry_gpu_jobs_max") or 14)
        remaining = max(0, maximum - used)
    if "oom_same_config" in issues:
        return "oom_same_config"
    lowered = notes.lower()
    if "disk full" in lowered or ("gated" in lowered and "auth" in lowered):
        return "auth_or_disk"
    if remaining == 0 or "budget exhausted" in lowered:
        return "budget_exhausted"
    if comparison:
        if comparison.get("quality_ok") is False:
            return "quality_failed"
        if comparison.get("quality_ok") is True and not is_beneficial(comparison):
            return "not_beneficial"
    verification = getattr(meta, "verification_status", None) if meta is not None else None
    status = getattr(meta, "status", None) if meta is not None else None
    if verification == "failed" or "loader_arch" in issues:
        return "verify_failed"
    if status in {"failed", "killed", "timeout"} or "process_failed" in issues:
        return "process_failed"
    if "benchmark_failed" in issues:
        return "quality_failed"
    return "quality_failed"


def validate_attempt(data: dict) -> dict:
    if not isinstance(data, dict):
        raise ValueError("attempt must be a JSON object")
    for key in ("model_id", "method_name", "gpu_instance"):
        if not str(data.get(key) or "").strip():
            raise ValueError(f"attempt missing {key}")
    reason = str(data.get("stop_reason") or "").strip()
    if reason not in ATTEMPT_STOP_REASONS:
        raise ValueError(f"attempt stop_reason must be one of {', '.join(ATTEMPT_STOP_REASONS)}")
    comparison = {
        "quality_ok": data.get("quality_ok") is True,
        "improved_vram": data.get("improved_vram") is True,
        "improved_throughput": data.get("improved_throughput") is True,
    }
    if is_beneficial(comparison):
        raise ValueError("beneficial runs belong in the catalog, not attempts")
    payload = dict(data)
    payload["stop_reason"] = reason
    payload["quality_ok"] = data.get("quality_ok") is True
    payload["schema_version"] = SCHEMA_VERSION
    return payload


def attempt_rows() -> list[dict]:
    rows: list[dict] = []
    directory = paths.LIBRARY_ROOT / "attempts"
    if not directory.is_dir():
        return rows
    for path in sorted(directory.glob("*.json")):
        try:
            rows.append(validate_attempt(_read_json(path)))
        except (OSError, json.JSONDecodeError, ValueError):
            continue
    rows.sort(
        key=lambda row: (
            str(row.get("model_id") or ""),
            str(row.get("gpu_instance") or ""),
            str(row.get("method_name") or ""),
        )
    )
    return rows


def query_attempts(
    *,
    model_id: str | None = None,
    method_name: str | None = None,
    gpu_instance: str | None = None,
) -> list[dict]:
    return [
        row
        for row in attempt_rows()
        if _eq(row.get("model_id"), model_id)
        and _eq(row.get("method_name"), method_name)
        and _eq(row.get("gpu_instance"), gpu_instance)
    ]


def _write_attempts() -> dict:
    rows = attempt_rows()
    index_path = paths.LIBRARY_ROOT / "attempts.json"
    paths.LIBRARY_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        index_path,
        json.dumps({"schema_version": SCHEMA_VERSION, "rows": rows}, indent=2) + "\n",
        mode=0o644,
    )
    lines = [
        "# Unsuccessful attempts",
        "",
        "Stops that did **not** publish: GPU budget exhausted, WikiText-2 PPL",
        "exploded, verify/run failed, or quality/efficiency missed the gate.",
        "These rows are **not** ranked and have no Hub weights. The published",
        "board is [LIBRARY.md](LIBRARY.md).",
        "",
        "Do not edit this file by hand. Parent `quant` records a row on STOP",
        "when the run is not beneficial (`library.py --record-attempt`).",
        "",
    ]
    if not rows:
        lines.append("No unsuccessful attempts recorded yet.")
        lines.append("")
    else:
        lines.append(
            "| Model | GPU | Method | Paper | Repo | Stop | Issues | PPL ratio | Retries | Job |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | --- |")
        for row in rows:
            issues = row.get("issue_codes") if isinstance(row.get("issue_codes"), list) else []
            used = row.get("retry_gpu_jobs_used")
            maximum = row.get("retry_gpu_jobs_max")
            retries = ""
            if isinstance(used, int) and isinstance(maximum, int):
                retries = f"{used}/{maximum}"
            lines.append(
                "| {model} | {gpu} | {method} | {paper} | {repo} | {stop} | {issues} | {ratio} | {retries} | {job} |".format(
                    model=row.get("model_id") or "",
                    gpu=row.get("gpu_instance") or "",
                    method=row.get("method_name") or "",
                    paper=_md_link("paper", row.get("paper_url")),
                    repo=_md_link("repo", row.get("repo_url")),
                    stop=row.get("stop_reason") or "",
                    issues=", ".join(str(item) for item in issues[:4]),
                    ratio=_num(row.get("ppl_ratio") or row.get("best_ppl_ratio"), 4),
                    retries=retries,
                    job=row.get("best_job_id") or row.get("job_id") or "",
                )
            )
        lines.append("")
    markdown_path = paths.LIBRARY_ROOT / "ATTEMPTS.md"
    atomic_write_text(markdown_path, "\n".join(lines) + "\n", mode=0o644)
    return {"index": index_path, "markdown": markdown_path}


def record_attempt(*, job_id: str, request: dict) -> dict:
    """Record an unsuccessful STOP for method × model × GPU. Not the public ranking."""
    meta = jobs_mod.refresh_status(job_id)
    diagnose = _job_diagnose(job_id, request)
    best_job_id = str(
        request.get("best_job_id") or diagnose.get("best_job_id") or job_id
    ).strip()
    benchmark = _job_benchmark(best_job_id) or _job_benchmark(job_id)
    comparison = benchmark.get("comparison") if isinstance(benchmark.get("comparison"), dict) else {}
    if is_beneficial(comparison):
        raise RuntimeError(
            "this run is beneficial (quality_ok and VRAM or tok/s beat fp16); use --catalog"
        )
    quantized = benchmark.get("quantized") if isinstance(benchmark.get("quantized"), dict) else {}
    baseline = benchmark.get("fp16_baseline") if isinstance(benchmark.get("fp16_baseline"), dict) else {}
    model_id = str(request.get("model_id") or "").strip()
    gpu_instance = str(request.get("gpu_instance") or "").strip()
    method_name = str(request.get("method_name") or "").strip()
    if not model_id or not gpu_instance or not method_name:
        raise ValueError("request must include model_id, gpu_instance, and method_name")
    output_dir = resolve_workspace_path(meta.output_dir)
    links = source_links(request)
    retry = diagnose.get("retry") if isinstance(diagnose.get("retry"), dict) else {}
    used = retry.get("used")
    if used is None:
        used = request.get("retry_gpu_jobs_used")
    maximum = retry.get("max")
    if maximum is None:
        maximum = request.get("retry_gpu_jobs_max")
    issue_codes = [
        str(item)
        for item in (diagnose.get("issue_codes") or diagnose.get("issues") or [])
        if str(item).strip()
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_id": model_id,
        "method_name": method_name,
        "gpu_instance": gpu_instance,
        "gpu_name": request.get("gpu_name") or request.get("gpu"),
        "slug": request.get("slug"),
        "stop_reason": classify_attempt_stop(
            request=request, diagnose=diagnose, comparison=comparison, meta=meta
        ),
        "quality_ok": comparison.get("quality_ok") is True,
        "issue_codes": issue_codes,
        "root_cause": diagnose.get("root_cause") or diagnose.get("root_cause_generic"),
        "notes": _clip_notes(diagnose.get("notes")),
        "error_excerpt": _clean_excerpt(diagnose.get("error_excerpt")),
        "ppl": quantized.get("perplexity"),
        "ppl_ratio": comparison.get("ppl_ratio"),
        "tokens_per_s": quantized.get("tokens_per_s"),
        "peak_vram_gb": quantized.get("peak_vram_gb"),
        "fp16_ppl": baseline.get("perplexity"),
        "fp16_tokens_per_s": baseline.get("tokens_per_s"),
        "fp16_peak_vram_gb": baseline.get("peak_vram_gb"),
        "improved_throughput": comparison.get("improved_throughput"),
        "improved_vram": comparison.get("improved_vram"),
        "packed": _packed_artifact(output_dir) if output_dir.is_dir() else False,
        "arxiv_id": links["arxiv_id"],
        "paper_url": links["paper_url"],
        "repo_url": links["repo_url"],
        "repo_commit": links["repo_commit"],
        "job_id": job_id,
        "best_job_id": best_job_id or job_id,
        "best_ppl_ratio": request.get("best_ppl_ratio")
        or diagnose.get("best_ppl_ratio")
        or comparison.get("ppl_ratio"),
        "best_overlay_dir": request.get("best_overlay_dir") or diagnose.get("best_overlay_dir"),
        "retry_gpu_jobs_used": int(used) if used is not None else 0,
        "retry_gpu_jobs_max": int(maximum) if maximum is not None else 14,
        "artifact_dir": _public_artifact_dir(output_dir) if output_dir.exists() else str(output_dir),
        "dataset": benchmark.get("dataset") or "wikitext-2-raw-v1",
        "max_seq_len": benchmark.get("max_seq_len") or 2048,
        "verification_status": meta.verification_status,
        "benchmark_status": meta.benchmark_status or ("passed" if comparison else None),
    }
    payload = validate_attempt(payload)
    dest = attempt_path(model_id, gpu_instance, method_name)
    atomic_write_text(dest, json.dumps(payload, indent=2) + "\n", mode=0o644)
    rebuild_derived()
    return {
        "status": "recorded",
        "path": str(dest),
        "attempts": str(paths.LIBRARY_ROOT / "attempts.json"),
        "attempts_md": str(paths.LIBRARY_ROOT / "ATTEMPTS.md"),
        "model_id": model_id,
        "gpu_instance": gpu_instance,
        "method_name": method_name,
        "stop_reason": payload["stop_reason"],
        "issue_codes": issue_codes,
        "job_id": job_id,
        "best_job_id": payload["best_job_id"],
        "quality_ok": payload["quality_ok"],
        "ppl_ratio": payload.get("ppl_ratio") or payload.get("best_ppl_ratio"),
        "retry_gpu_jobs_used": payload["retry_gpu_jobs_used"],
        "retry_gpu_jobs_max": payload["retry_gpu_jobs_max"],
    }


def record_job(
    *,
    job_id: str,
    request: dict,
    hub_url: str | None = None,
) -> dict:
    meta = jobs_mod.refresh_status(job_id)
    job_directory = jobs_mod.job_dir(job_id)
    benchmark_path = job_directory / "benchmark.json"
    if not benchmark_path.is_file():
        raise RuntimeError("library record requires jobs/<id>/benchmark.json")
    benchmark = _read_json(benchmark_path)
    comparison = benchmark.get("comparison") if isinstance(benchmark.get("comparison"), dict) else {}
    if not is_beneficial(comparison):
        raise RuntimeError(
            "library record is only for quality_ok runs that also beat fp16 VRAM or tok/s"
        )
    quantized = benchmark.get("quantized") if isinstance(benchmark.get("quantized"), dict) else {}
    baseline = benchmark.get("fp16_baseline") if isinstance(benchmark.get("fp16_baseline"), dict) else {}
    output_dir = resolve_workspace_path(meta.output_dir)
    model_id = str(request.get("model_id") or "")
    gpu_instance = str(request.get("gpu_instance") or "")
    if not model_id or not gpu_instance:
        raise ValueError("request must include model_id and gpu_instance")
    group = load_group(model_id, gpu_instance)
    group["gpu_name"] = request.get("gpu_name") or request.get("gpu") or group.get("gpu_name")
    if baseline:
        group["fp16_baseline"] = {
            "perplexity": baseline.get("perplexity"),
            "tokens_per_s": baseline.get("tokens_per_s"),
            "peak_vram_gb": baseline.get("peak_vram_gb"),
        }
    method_name = str(request.get("method_name") or "unknown")
    slug = str(request.get("slug") or "")
    links = source_links(request)
    row = {
        "method_name": method_name,
        "slug": slug,
        "job_id": job_id,
        "quality_ok": comparison.get("quality_ok") is True,
        "ppl": quantized.get("perplexity"),
        "ppl_ratio": comparison.get("ppl_ratio"),
        "tokens_per_s": quantized.get("tokens_per_s"),
        "peak_vram_gb": quantized.get("peak_vram_gb"),
        "improved_throughput": comparison.get("improved_throughput"),
        "improved_vram": comparison.get("improved_vram"),
        "packed": _packed_artifact(output_dir) if output_dir.is_dir() else False,
        "artifact_dir": _public_artifact_dir(output_dir) if output_dir.exists() else str(output_dir),
        "hub_url": hub_url,
        "arxiv_id": links["arxiv_id"],
        "paper_url": links["paper_url"],
        "repo_url": links["repo_url"],
        "repo_commit": links["repo_commit"],
        "verification_status": meta.verification_status,
    }
    by_method: dict[str, dict] = {}
    for old in group.get("methods") or []:
        by_method[str(old.get("method_name"))] = dict(old)
    prev = by_method.get(method_name, {})
    merged = {**prev, **row}
    if not hub_url:
        merged["hub_url"] = prev.get("hub_url")
    by_method[method_name] = merged
    group["methods"] = list(by_method.values())
    clear_attempt(model_id, gpu_instance, method_name)
    dest = _write_group(group)
    group["path"] = str(dest)
    effective_hub = merged.get("hub_url")
    if row["quality_ok"] and effective_hub:
        try:
            group["contribution"] = str(
                export_contribution(job_id=job_id, request=request, hub_url=str(effective_hub))
            )
        except Exception:
            pass
    return group


def export_contribution(*, job_id: str, request: dict, hub_url: str) -> Path:
    meta = jobs_mod.refresh_status(job_id)
    benchmark = _read_json(jobs_mod.job_dir(job_id) / "benchmark.json")
    comparison = benchmark.get("comparison") if isinstance(benchmark.get("comparison"), dict) else {}
    quantized = benchmark.get("quantized") if isinstance(benchmark.get("quantized"), dict) else {}
    baseline = benchmark.get("fp16_baseline") if isinstance(benchmark.get("fp16_baseline"), dict) else {}
    output_dir = resolve_workspace_path(meta.output_dir)
    links = source_links(request)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_id": request.get("model_id"),
        "method_name": request.get("method_name"),
        "gpu_instance": request.get("gpu_instance"),
        "gpu_name": request.get("gpu_name") or request.get("gpu"),
        "quality_ok": comparison.get("quality_ok") is True,
        "ppl": quantized.get("perplexity"),
        "ppl_ratio": comparison.get("ppl_ratio"),
        "tokens_per_s": quantized.get("tokens_per_s"),
        "peak_vram_gb": quantized.get("peak_vram_gb"),
        "fp16_ppl": baseline.get("perplexity"),
        "fp16_tokens_per_s": baseline.get("tokens_per_s"),
        "fp16_peak_vram_gb": baseline.get("peak_vram_gb"),
        "packed": _packed_artifact(output_dir) if output_dir.is_dir() else False,
        "hub_url": hub_url,
        "hub_repo_id": _hub_repo_id(hub_url),
        "arxiv_id": links["arxiv_id"],
        "paper_url": links["paper_url"],
        "repo_url": links["repo_url"],
        "repo_commit": links["repo_commit"],
        "dataset": benchmark.get("dataset") or "wikitext-2-raw-v1",
        "max_seq_len": benchmark.get("max_seq_len") or 2048,
        "job_id": job_id,
        "slug": request.get("slug"),
        "artifact_dir": _public_artifact_dir(output_dir) if output_dir.exists() else None,
        "verification_status": meta.verification_status,
    }
    payload = validate_contribution(payload, require_hub=True)
    dest = contributions_dir() / contribution_filename(
        str(payload["model_id"]),
        str(payload["gpu_instance"]),
        str(payload["method_name"]),
    )
    atomic_write_text(dest, json.dumps(payload, indent=2) + "\n", mode=0o644)
    rebuild_derived()
    return dest


def catalog_run(*, job_id: str, request: dict, hub_url: str) -> dict:
    """Library row for a beneficial quality_ok run: model, GPU, method, paper, repo, Hub."""
    hub_url = str(hub_url or "").strip()
    if not hub_url:
        raise ValueError("quant-catalog needs --hub-url (run quant-publish first)")
    benchmark = _read_json(jobs_mod.job_dir(job_id) / "benchmark.json")
    comparison = benchmark.get("comparison") if isinstance(benchmark.get("comparison"), dict) else {}
    if not is_beneficial(comparison):
        raise RuntimeError(
            "catalog is only for quality_ok runs that also beat fp16 VRAM or tok/s"
        )
    links = source_links(request)
    if not links["paper_url"] or not links["repo_url"]:
        raise ValueError("request JSON must include arxiv_id and repo_url")
    group = record_job(job_id=job_id, request=request, hub_url=hub_url)
    return {
        "status": "cataloged",
        "model_id": request.get("model_id"),
        "gpu_instance": request.get("gpu_instance"),
        "method_name": request.get("method_name"),
        "paper_url": links["paper_url"],
        "repo_url": links["repo_url"],
        "hub_url": hub_url,
        "hub_repo_id": _hub_repo_id(hub_url),
        "contribution": group.get("contribution"),
        "catalog": str(paths.LIBRARY_ROOT / "catalog.json"),
        "quality_ok": True,
        "improved_vram": comparison.get("improved_vram"),
        "improved_throughput": comparison.get("improved_throughput"),
    }


def accept_contribution(path: Path) -> Path:
    payload = validate_contribution(_read_json(path), require_hub=True)
    dest = contributions_dir() / contribution_filename(
        str(payload["model_id"]),
        str(payload["gpu_instance"]),
        str(payload["method_name"]),
    )
    atomic_write_text(dest, json.dumps(payload, indent=2) + "\n", mode=0o644)
    rebuild_derived()
    return dest


def _record_vllm_cli(args) -> dict:
    if args.job_id:
        if args.request is None:
            raise SystemExit("--request is required with --job-id")
        request = load_request(args.request)
        key = (request.get("model_id"), request.get("gpu_instance"), request.get("method_name"))
        export_path = jobs_mod.job_dir(args.job_id) / "vllm_export.json"
        report_path = jobs_mod.job_dir(args.job_id) / "vllm.json"
    else:
        key = (args.model_id, args.gpu_instance, args.method_name)
        if not all(key) or args.vllm_export is None:
            raise SystemExit("--record-vllm needs --job-id/--request, or --model-id --gpu-instance --method --vllm-export")
        export_path = args.vllm_export
        report_path = args.serve_report
    if not export_path.is_file():
        raise SystemExit(f"{export_path} missing; run export_vllm.py first")
    serve_report = _read_json(report_path) if report_path is not None and report_path.is_file() else None
    return record_vllm(
        model_id=str(key[0]),
        gpu_instance=str(key[1]),
        method_name=str(key[2]),
        export_record=_read_json(export_path),
        serve_report=serve_report,
        hub_repo_id=args.vllm_repo_id,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Filter the quantized catalog, record a beneficial row, or record an unsuccessful attempt"
    )
    parser.add_argument("--job-id")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--hub-url")
    parser.add_argument("--model-id", help="Filter: Hugging Face model id")
    parser.add_argument("--method", dest="method_name", help="Filter: quantization method")
    parser.add_argument("--gpu-instance", help="Filter: instance type, e.g. g5.2xlarge")
    parser.add_argument("--best", action="store_true", help="Only the best row per matching board")
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Print how to download matching Hub repos (does not download)",
    )
    parser.add_argument(
        "--export-contribution",
        action="store_true",
        help="Write library/contributions/<model>__<gpu>__<method>.json for a PR",
    )
    parser.add_argument(
        "--accept-contribution",
        type=Path,
        help="Validate a contribution JSON and merge it into the catalog",
    )
    parser.add_argument(
        "--catalog",
        action="store_true",
        help="Record a beneficial run into the library (model, GPU, method, paper, repo, Hub)",
    )
    parser.add_argument(
        "--record-attempt",
        action="store_true",
        help="Record an unsuccessful STOP into library/attempts (not the public ranking)",
    )
    parser.add_argument(
        "--attempts",
        action="store_true",
        help="List unsuccessful attempts instead of the published catalog",
    )
    parser.add_argument(
        "--record-vllm",
        action="store_true",
        help="Attach the vLLM export + serve-check result to an existing library row",
    )
    parser.add_argument("--vllm-export", type=Path, help="vllm_export.json (backfill; job mode reads jobs/<id>/)")
    parser.add_argument("--serve-report", type=Path, help="serve_check report (backfill; job mode reads jobs/<id>/vllm.json)")
    parser.add_argument("--vllm-repo-id", help="Hub org/name of the published vLLM build (passed checks only)")
    parser.add_argument("--rebuild", action="store_true", help="Rewrite catalog.json from groups + contributions")
    parser.add_argument(
        "--sync-hf-collection",
        action="store_true",
        help="Create/update the one public Hugging Face collection from catalog hub_repo_id rows",
    )
    parser.add_argument(
        "--collection-slug",
        help="Existing collection slug (owner/name-hash). Default: library/library.json or QUANT_AGENT_HF_COLLECTION",
    )
    args = parser.parse_args()
    try:
        if args.sync_hf_collection:
            print(json.dumps(sync_hf_collection(collection_slug=args.collection_slug), indent=2))
            return 0
        if args.accept_contribution:
            dest = accept_contribution(args.accept_contribution)
            print(json.dumps({"status": "accepted", "path": str(dest)}, indent=2))
            return 0
        if args.rebuild:
            print(json.dumps(rebuild_derived(), indent=2))
            return 0
        if args.record_vllm:
            print(json.dumps(_record_vllm_cli(args), indent=2))
            return 0
        if args.job_id:
            if args.request is None:
                raise SystemExit("--request is required with --job-id")
            request = load_request(args.request)
            if args.catalog and args.record_attempt:
                raise SystemExit("use either --catalog or --record-attempt, not both")
            if args.catalog:
                payload = catalog_run(
                    job_id=args.job_id, request=request, hub_url=str(args.hub_url or "")
                )
                print(json.dumps(payload, indent=2))
                return 0
            if args.record_attempt:
                payload = record_attempt(job_id=args.job_id, request=request)
                print(json.dumps(payload, indent=2))
                return 0
            if args.export_contribution:
                if not args.hub_url:
                    raise SystemExit("--export-contribution needs --hub-url (public Hub repo)")
                dest = export_contribution(
                    job_id=args.job_id, request=request, hub_url=args.hub_url
                )
                print(json.dumps({"status": "exported", "path": str(dest)}, indent=2))
                return 0
            raise SystemExit("library write requires --catalog, --record-attempt, or --export-contribution")
        if args.attempts:
            rows = query_attempts(
                model_id=args.model_id,
                method_name=args.method_name,
                gpu_instance=args.gpu_instance,
            )
            print(json.dumps({"rows": rows}, indent=2))
            return 0
        rows = query(
            model_id=args.model_id,
            method_name=args.method_name,
            gpu_instance=args.gpu_instance,
            best_only=args.best,
        )
        if args.fetch:
            print(json.dumps({"rows": rows, "fetch": [fetch_howto(row) for row in rows]}, indent=2))
        else:
            print(json.dumps({"rows": rows}, indent=2))
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

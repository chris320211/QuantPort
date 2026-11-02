"""Request JSON handoff between skills. No secrets."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from io_utils import atomic_write_text
from paths import REQUESTS_ROOT, require_slug

# Extra GPU jobs after the first winner. Novel-method ports on a new arch
# typically need overlay switches plus packed dtype/eval-flag fixes.
RETRY_GPU_JOBS_MAX = 14

_SLUGIFY_RE = re.compile(r"[^a-z0-9]+")
_SECRET_KEY_RE = re.compile(
    r"token|secret|password|api[_-]?key|hf_token|openai|github",
    re.I,
)

REQUIRED = (
    "method_name",
    "model_id",
    "gpu_instance",
    "arxiv_id",
    "paper_path",
    "repo_url",
    "repo_path",
    "repo_commit",
    "hf_snapshot_path",
    "slug",
)

OPTIONAL = (
    "instance_type",
    "gpu_arch",
    "vram_gb",
    "gpu",
    "gpu_count",
    "compute_capability",
    "gpu_name",
    "nvidia_smi",
    "gpus",
    "vram_gb_free",
    "vram_gb_total_live",
    "venv_python",
    "winner_overlay_dir",
    "winner_script",
    "ranked",
    "tried_overlays",
    "retry_gpu_jobs_used",
    "retry_gpu_jobs_max",
    "last_job_id",
    "last_diagnose_path",
    "best_job_id",
    "best_overlay_dir",
    "best_ppl_ratio",
    # out/analysis/<slug>/differences.json, written by quant-port's planner.
    "port_spec_path",
)

ALLOWED_KEYS = frozenset(REQUIRED + OPTIONAL)


def slugify(*parts: str) -> str:
    raw = "-".join(parts)
    return _SLUGIFY_RE.sub("-", raw.lower()).strip("-")[:80] or "request"


def request_path(slug: str) -> Path:
    require_slug(slug)
    return REQUESTS_ROOT / f"{slug}.json"


def _reject_secret_keys(data: dict) -> None:
    for key in data:
        if _SECRET_KEY_RE.search(str(key)):
            raise ValueError(f"request JSON must not contain secret-like key {key!r}")


def _validate(data: dict) -> dict:
    _reject_secret_keys(data)
    unknown = sorted(set(data) - ALLOWED_KEYS)
    if unknown:
        raise ValueError(f"request JSON has unsupported keys: {unknown}")
    missing = [key for key in REQUIRED if not data.get(key)]
    if missing:
        raise ValueError(f"request JSON missing {missing}")
    require_slug(str(data["slug"]))
    return data


def load_request(path: Path) -> dict:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("request JSON must be an object")
    return _validate(data)


def record_retry(
    path: Path,
    *,
    tried_overlay: str | None = None,
    last_job_id: str | None = None,
    last_diagnose_path: str | None = None,
    increment_used: bool = True,
) -> dict:
    """Append a tried overlay and count one parent retry GPU job."""
    data = load_request(path)
    tried = [str(item) for item in (data.get("tried_overlays") or [])]
    if tried_overlay:
        marker = str(tried_overlay).rstrip("/")
        if marker not in tried and f"{marker}/" not in tried:
            tried.append(marker)
    data["tried_overlays"] = tried
    if increment_used:
        data["retry_gpu_jobs_used"] = int(data.get("retry_gpu_jobs_used") or 0) + 1
    if data.get("retry_gpu_jobs_max") in (None, ""):
        data["retry_gpu_jobs_max"] = RETRY_GPU_JOBS_MAX
    if last_job_id:
        data["last_job_id"] = str(last_job_id)
    if last_diagnose_path:
        data["last_diagnose_path"] = str(last_diagnose_path)
    write_request(data)
    return load_request(path)


def write_request(payload: dict) -> Path:
    if not isinstance(payload, dict):
        raise ValueError("request JSON must be an object")
    _reject_secret_keys(payload)
    slug = payload.get("slug") or slugify(
        str(payload.get("method_name", "method")),
        str(payload.get("model_id", "model")),
        str(payload.get("gpu_instance", "gpu")),
    )
    require_slug(str(slug))
    filtered = {key: payload[key] for key in ALLOWED_KEYS if key in payload}
    filtered["slug"] = slug
    filtered.setdefault("retry_gpu_jobs_used", 0)
    filtered.setdefault("retry_gpu_jobs_max", RETRY_GPU_JOBS_MAX)
    filtered.setdefault("tried_overlays", [])
    _validate(filtered)
    path = request_path(slug)
    atomic_write_text(path, json.dumps(filtered, indent=2, sort_keys=True) + "\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Read or write a gather request JSON")
    parser.add_argument("path", nargs="?", type=Path)
    parser.add_argument("--write-json", help="JSON object to write")
    parser.add_argument("--record-retry", action="store_true")
    parser.add_argument("--tried-overlay")
    parser.add_argument("--last-job-id")
    parser.add_argument("--last-diagnose-path")
    args = parser.parse_args()
    if args.write_json:
        path = write_request(json.loads(args.write_json))
        print(path)
        return 0
    if args.record_retry:
        if args.path is None:
            raise SystemExit("path required")
        payload = record_retry(
            args.path,
            tried_overlay=args.tried_overlay,
            last_job_id=args.last_job_id,
            last_diagnose_path=args.last_diagnose_path,
        )
        print(json.dumps(payload, indent=2))
        return 0
    if args.path is None:
        raise SystemExit("path required")
    print(json.dumps(load_request(args.path), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

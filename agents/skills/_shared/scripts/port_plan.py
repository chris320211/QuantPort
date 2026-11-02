#!/usr/bin/env python3
"""Port planning helpers: the differences spec, candidate ranking, and recipes.

``spec-check`` validates ``out/analysis/<slug>/differences.json``: what this
model does that the method repo's supported path does not, each item with an
id (``D1``...), a severity, and the evidence it came from, plus the ``smoke``
thresholds every candidate is judged by.

``rank`` orders smoked overlays by evidence: smoke passed first, then how many
``must`` differences the overlay claims to address, then target-linear
coverage, then lower relative logit error, then smaller patch, then fewer
files. Only passing overlays are ranked; ``--write`` stores the winner in the
request JSON.

``recipes`` lists what the library already knows about this method on this
model or model family: published rows are positive recipes, attempts that
stopped short are negative ones.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

sys.path.insert(0, str(Path(__file__).resolve().parent))

import overlay as overlay_mod
import paths
import request as request_mod
from port_smoke import SmokeSettings, read_result

SEVERITIES = ("must", "should", "info")
# Common kinds; free text is allowed so a new architecture is not forced into a box.
KNOWN_KINDS = (
    "attention_bias",
    "mlp_bias",
    "qkv_layout",
    "gqa",
    "qk_norm",
    "norm_type",
    "norm_placement",
    "tied_embeddings",
    "rope",
    "activation",
    "moe",
    "layer_pattern",
    "module_names",
    "head_dim",
    "save_load",
    "kernel_dtype",
    "remote_code",
    "other",
)


class Difference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: str = Field(..., min_length=1, max_length=64)
    detail: str = Field(..., min_length=1, max_length=2000)
    severity: Literal["must", "should", "info"]
    # arch.json keys, repo file:line, paper section: where this came from.
    evidence: list[str] = Field(default_factory=list, max_length=20)
    # How the method repo's own path handles it, as read from the checkout.
    repo_support: Literal["supported", "partial", "missing", "unknown"] = "unknown"

    @field_validator("id")
    @classmethod
    def _id(cls, value: str) -> str:
        overlay_mod.require_difference_ids([value])
        return value

    @model_validator(mode="after")
    def _must_has_evidence(self) -> "Difference":
        if self.severity == "must" and not self.evidence:
            raise ValueError(f"{self.id}: a must difference needs at least one evidence entry")
        return self


class PortSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    slug: str
    method_name: str = Field(..., min_length=1)
    model_id: str = Field(..., min_length=1)
    model_type: str | None = None
    differences: list[Difference] = Field(default_factory=list, max_length=100)
    smoke: SmokeSettings = Field(default_factory=SmokeSettings)
    # library/ports/... refs the planner read before writing this spec.
    recipes: list[str] = Field(default_factory=list, max_length=20)
    notes: str | None = Field(None, max_length=4000)

    @field_validator("slug")
    @classmethod
    def _slug(cls, value: str) -> str:
        return paths.require_slug(value)

    @model_validator(mode="after")
    def _unique_ids(self) -> "PortSpec":
        seen: set[str] = set()
        dupes = sorted({d.id for d in self.differences if d.id in seen or seen.add(d.id)})
        if dupes:
            raise ValueError(f"duplicate difference ids: {dupes}")
        return self

    def must_ids(self) -> list[str]:
        return [d.id for d in self.differences if d.severity == "must"]


def spec_path_for(slug: str) -> Path:
    return paths.ANALYSIS_ROOT / paths.require_slug(slug) / "differences.json"


def load_spec(path: Path) -> PortSpec:
    return PortSpec.model_validate_json(path.read_text())


def check_spec(path: Path, request: dict | None = None) -> dict:
    spec = load_spec(path)
    if request is not None:
        for key in ("slug", "model_id"):
            if str(request.get(key)) != getattr(spec, key):
                raise ValueError(f"spec {key} {getattr(spec, key)!r} does not match request {request.get(key)!r}")
    counts = {severity: 0 for severity in SEVERITIES}
    for diff in spec.differences:
        counts[diff.severity] += 1
    unknown_kinds = sorted({d.kind for d in spec.differences if d.kind not in KNOWN_KINDS})
    return {
        "status": "ok",
        "path": str(path.resolve()),
        "differences": len(spec.differences),
        "severity": counts,
        "must": spec.must_ids(),
        "missing_in_repo": [d.id for d in spec.differences if d.repo_support in ("missing", "partial")],
        "unknown_kinds": unknown_kinds,
        "smoke": spec.smoke.model_dump(),
    }


# --- rank ---------------------------------------------------------------------


def patch_lines(patch: str) -> int:
    """Added plus removed lines, not counting file headers."""
    return sum(
        1
        for line in patch.splitlines()
        if (line.startswith("+") and not line.startswith("+++"))
        or (line.startswith("-") and not line.startswith("---"))
    )


def discover_overlays(slug: str) -> list[Path]:
    """Every bundle under out/overlays/<slug>/<plan>/<hash>/ that has a smoke result."""
    root = paths.OVERLAYS_ROOT / paths.require_slug(slug)
    if not root.is_dir():
        return []
    found = []
    for bundle in sorted(root.glob("*/*")):
        if bundle.is_dir() and not bundle.name.startswith(".") and (bundle / "manifest.json").is_file():
            if read_result(slug, bundle) is not None:
                found.append(bundle.resolve())
    return found


def candidate(slug: str, overlay_dir: Path, spec: PortSpec) -> dict:
    """Everything rank needs about one overlay, plus why it is not eligible."""
    overlay_dir = overlay_dir.expanduser().resolve()
    entry: dict = {"overlay_dir": str(overlay_dir), "plan": overlay_dir.parent.name, "reasons": []}
    try:
        manifest = overlay_mod.validate_overlay_bundle(overlay_dir)
    except Exception as exc:  # noqa: BLE001 - any invalid bundle is just ineligible
        entry.update(eligible=False, reasons=[f"invalid_bundle: {exc}"])
        return entry
    patch = (overlay_dir / "overlay.patch").read_text()
    must = spec.must_ids()
    known = {d.id for d in spec.differences}
    addressed = [item for item in manifest.addresses if item in known]
    entry.update(
        addresses=list(manifest.addresses),
        unknown_addresses=[item for item in manifest.addresses if item not in known],
        must_addressed=[item for item in must if item in addressed],
        must_missing=[item for item in must if item not in addressed],
        must_coverage=(sum(1 for item in must if item in addressed) / len(must)) if must else 1.0,
        patch_lines=patch_lines(patch),
        files_touched=len(overlay_mod.validate_unified_patch(patch)),
    )
    if manifest.method_id != spec.slug or manifest.model_id != spec.model_id:
        entry["reasons"].append("overlay_identity_mismatch")
    smoke = read_result(slug, overlay_dir)
    if smoke is None:
        entry["reasons"].append("not_smoked")
        entry.update(eligible=False, smoke_status=None)
        return entry
    metrics = smoke.get("metrics") or {}
    entry.update(
        smoke_status=smoke.get("status"),
        smoke_stage=smoke.get("stage"),
        smoke_reasons=smoke.get("reasons") or [],
        coverage=metrics.get("coverage"),
        rel_logit_error=metrics.get("rel_logit_error"),
        top1_agreement=metrics.get("top1_agreement"),
        script=smoke.get("script"),
    )
    if smoke.get("status") != "passed":
        entry["reasons"].append(f"smoke_{smoke.get('status') or 'unknown'}")
    if smoke.get("settings") != spec.smoke.model_dump():
        entry["reasons"].append("smoke_settings_changed")
    script = Path(smoke["script"]) if smoke.get("script") else None
    if script is None or not script.is_file():
        entry["reasons"].append("script_missing")
    else:
        code = script.read_text()
        if smoke.get("script_sha256") != hashlib.sha256(code.encode("utf-8")).hexdigest():
            entry["reasons"].append("script_changed_since_smoke")
        if overlay_mod.overlay_path_from_script(code) != overlay_dir:
            entry["reasons"].append("script_points_at_another_overlay")
    entry["eligible"] = not entry["reasons"]
    return entry


def _sort_key(entry: dict) -> tuple:
    rel = entry.get("rel_logit_error")
    return (
        not entry.get("eligible"),
        -float(entry.get("must_coverage") or 0.0),
        -float(entry.get("coverage") or 0.0),
        rel if isinstance(rel, (int, float)) and math.isfinite(rel) else math.inf,
        int(entry.get("patch_lines") or 0),
        int(entry.get("files_touched") or 0),
        entry["overlay_dir"],
    )


def rank(slug: str, spec: PortSpec, overlay_dirs: list[Path]) -> dict:
    entries = sorted((candidate(slug, path, spec) for path in overlay_dirs), key=_sort_key)
    ranked = [entry["overlay_dir"] for entry in entries if entry["eligible"]]
    winner = entries[0] if ranked else None
    return {
        "status": "ok" if ranked else "no_passing_overlay",
        "winner_overlay_dir": winner["overlay_dir"] if winner else None,
        "winner_script": winner["script"] if winner else None,
        "ranked": ranked,
        "candidates": entries,
    }


def write_winner(request_path: Path, result: dict, spec_path: Path) -> Path:
    data = request_mod.load_request(request_path)
    data["winner_overlay_dir"] = result["winner_overlay_dir"]
    data["winner_script"] = result["winner_script"]
    data["ranked"] = result["ranked"]
    data["port_spec_path"] = str(spec_path.resolve())
    return request_mod.write_request(data)


# --- recipes ------------------------------------------------------------------


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _norm(value: object) -> str:
    return str(value or "").strip().lower()


def _port_summary(ref: str | None) -> dict | None:
    if not ref:
        return None
    folder = paths.REPO_ROOT / ref
    record = _read_json(folder / "port.json")
    if record is None:
        return None
    manifest = _read_json(folder / "manifest.json") or {}
    summary = {
        "path": ref,
        "model_type": record.get("model_type"),
        "repo_commit": record.get("repo_commit"),
        "rationale": manifest.get("rationale"),
        "addresses": manifest.get("addresses") or [],
        "target_modules": (manifest.get("target_modules") or [])[:20],
    }
    differences = _read_json(folder / "differences.json")
    if differences is not None:
        summary["differences"] = [
            {key: item.get(key) for key in ("id", "kind", "severity", "detail")}
            for item in differences.get("differences") or []
            if isinstance(item, dict)
        ]
    return summary


def recipes(*, method: str, model_type: str | None = None, model_id: str | None = None) -> dict:
    """Library rows for ``method`` on the same model or model_type, best evidence first."""
    root = paths.LIBRARY_ROOT
    rows: list[tuple[str, dict]] = []
    for kind, folder in (("positive", root / "contributions"), ("negative", root / "attempts")):
        for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
            row = _read_json(path)
            if row is not None and _norm(row.get("method_name")) == _norm(method):
                rows.append((kind, {**row, "_file": str(path.relative_to(paths.REPO_ROOT))}))
    out = []
    for kind, row in rows:
        port = _port_summary(row.get("port"))
        row_type = (port or {}).get("model_type")
        if model_id and _norm(row.get("model_id")) == _norm(model_id):
            match = "model_id"
        elif model_type and row_type and _norm(row_type) == _norm(model_type):
            match = "model_type"
        else:
            continue
        if kind == "positive" and row.get("quality_ok") is False:
            kind = "negative"
        item = {
            "outcome": kind,
            "match": match,
            "row": row["_file"],
            "model_id": row.get("model_id"),
            "gpu_instance": row.get("gpu_instance"),
            "packed": row.get("packed"),
            "ppl_ratio": row.get("ppl_ratio"),
            "repo_commit": row.get("repo_commit"),
            "port": port,
        }
        if kind == "negative":
            item.update(
                stop_reason=row.get("stop_reason"),
                root_cause=row.get("root_cause"),
                issue_codes=row.get("issue_codes") or [],
                best_ppl_ratio=row.get("best_ppl_ratio"),
                notes=(row.get("notes") or [])[:4],
                retry_gpu_jobs_used=row.get("retry_gpu_jobs_used"),
            )
        out.append(item)
    out.sort(key=lambda item: (item["match"] != "model_id", item["outcome"] != "positive", item["row"]))
    return {"status": "ok", "method": method, "model_type": model_type, "model_id": model_id, "recipes": out}


def snapshot_model_type(request: dict) -> str | None:
    snapshot = request.get("hf_snapshot_path")
    if not snapshot:
        return None
    config = _read_json(Path(snapshot).expanduser() / "config.json")
    value = (config or {}).get("model_type")
    return str(value) if value else None


# --- CLI ----------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Port spec check, overlay ranking, library recipes")
    sub = parser.add_subparsers(dest="command", required=True)

    spec_p = sub.add_parser("spec-check", help="Validate out/analysis/<slug>/differences.json")
    spec_p.add_argument("--request", type=Path)
    spec_p.add_argument("--spec", type=Path, help="default out/analysis/<slug>/differences.json")

    rank_p = sub.add_parser("rank", help="Order smoked overlays; only passing ones are ranked")
    rank_p.add_argument("--request", type=Path, required=True)
    rank_p.add_argument("--spec", type=Path)
    rank_p.add_argument("--overlay-dir", type=Path, action="append", default=[],
                        help="candidate bundle (repeat); default: every smoked bundle for the slug")
    rank_p.add_argument("--write", action="store_true", help="store winner/ranked in the request JSON")

    rec_p = sub.add_parser("recipes", help="Library rows for this method on this model or family")
    rec_p.add_argument("--request", type=Path, help="fills method, model id and model_type")
    rec_p.add_argument("--method")
    rec_p.add_argument("--model-type")
    rec_p.add_argument("--model-id")

    args = parser.parse_args()
    try:
        if args.command == "spec-check":
            request = request_mod.load_request(args.request) if args.request else None
            if args.spec is None and request is None:
                raise ValueError("pass --spec or --request")
            spec_path = args.spec or spec_path_for(str(request["slug"]))
            payload = check_spec(spec_path, request)
        elif args.command == "rank":
            request = request_mod.load_request(args.request)
            slug = str(request["slug"])
            spec_path = args.spec or spec_path_for(slug)
            spec = load_spec(spec_path)
            check_spec(spec_path, request)
            tried = {str(item).rstrip("/") for item in request.get("tried_overlays") or []}
            dirs = args.overlay_dir or [d for d in discover_overlays(slug) if str(d) not in tried]
            payload = rank(slug, spec, dirs)
            if args.write and payload["status"] == "ok":
                payload["request"] = str(write_winner(args.request, payload, spec_path))
        else:
            request = request_mod.load_request(args.request) if args.request else {}
            method = args.method or request.get("method_name")
            if not method:
                raise ValueError("pass --method or --request")
            payload = recipes(
                method=str(method),
                model_type=args.model_type or snapshot_model_type(request),
                model_id=args.model_id or request.get("model_id"),
            )
    except (ValueError, ValidationError, RuntimeError, OSError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, indent=2))
        return 1
    print(json.dumps(payload, indent=2))
    return 0 if payload.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Shared kernel library: list entries, match one to a job's saved format, check the tree.

Entries live in `agents/skills/_shared/kernels/<name>/kernel.json` and are keyed
by the saved format an artifact declares in its `quantization_config.json`,
never by model or GPU name. `quant-kernel` matches before writing a new kernel.
Reads JSON only; never imports or runs a kernel.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

import jobs as jobs_mod
import paths

KERNELS_ROOT = Path(__file__).resolve().parents[1] / "kernels"
QUANT_CONFIG_NAME = "quantization_config.json"


class TunedOn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gpu: str = Field(..., min_length=1)
    compute_capability: str = Field(..., pattern=r"^\d+\.\d+$")
    torch: str | None = None
    triton: str | None = None


class KernelEntry(BaseModel):
    """One `kernel.json`."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    name: str = Field(..., pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    kind: Literal["module", "port"]
    module: str | None = None
    entry_points: list[str] = Field(default_factory=list)
    port: str | None = None
    # quantization_config.json key -> allowed values; every key must match.
    match: dict[str, list[Any]] = Field(..., min_length=1)
    requires: list[str] = Field(default_factory=list)
    min_compute_capability: str | None = Field(None, pattern=r"^\d+\.\d+$")
    tuned_on: list[TunedOn] = Field(default_factory=list)
    results: list[dict[str, Any]] = Field(default_factory=list)
    notes: str | None = None

    @model_validator(mode="after")
    def _kind_fields(self) -> "KernelEntry":
        if self.kind == "module" and not (self.module and self.entry_points):
            raise ValueError(f"{self.name}: a module entry needs module and entry_points")
        if self.kind == "port" and not self.port:
            raise ValueError(f"{self.name}: a port entry needs port")
        if any(not values for values in self.match.values()):
            raise ValueError(f"{self.name}: every match key needs at least one allowed value")
        return self


def _capability(value: str) -> tuple[int, int]:
    major, minor = value.split(".")
    return int(major), int(minor)


def load_entries(root: Path = KERNELS_ROOT) -> list[tuple[Path, KernelEntry]]:
    entries = []
    for path in sorted(root.glob("*/kernel.json")):
        entries.append((path.parent, KernelEntry.model_validate_json(path.read_text())))
    return entries


def check(root: Path = KERNELS_ROOT, repo_root: Path = paths.REPO_ROOT) -> dict:
    """Every entry parses, names are unique, and the files it points at exist."""
    entries = load_entries(root)
    problems = []
    names = [entry.name for _, entry in entries]
    for name in sorted({n for n in names if names.count(n) > 1}):
        problems.append(f"duplicate name {name}")
    for folder, entry in entries:
        if entry.kind == "module" and not (folder / entry.module).is_file():
            problems.append(f"{entry.name}: missing {folder.name}/{entry.module}")
        if entry.kind == "port" and not (repo_root / entry.port).is_dir():
            problems.append(f"{entry.name}: missing port {entry.port}")
    return {
        "status": "ok" if not problems else "error",
        "entries": names,
        "problems": problems,
    }


def matches(entry: KernelEntry, config: dict) -> bool:
    return all(key in config and config[key] in allowed for key, allowed in entry.match.items())


def match(config: dict, *, compute_capability: str | None = None, root: Path = KERNELS_ROOT) -> dict:
    """Entries whose `match` block the saved config satisfies, best first by tuned GPU."""
    found = []
    for folder, entry in load_entries(root):
        if not matches(entry, config):
            continue
        item = {
            "name": entry.name,
            "kind": entry.kind,
            "path": str(folder),
            "module": str(folder / entry.module) if entry.module else None,
            "entry_points": entry.entry_points,
            "port": entry.port,
            "requires": entry.requires,
            "tuned_on": [t.model_dump(exclude_none=True) for t in entry.tuned_on],
            "notes": entry.notes,
        }
        if compute_capability:
            tuned = {t.compute_capability for t in entry.tuned_on}
            item["retune"] = compute_capability not in tuned
            minimum = entry.min_compute_capability
            item["supported"] = not minimum or _capability(compute_capability) >= _capability(minimum)
        found.append(item)
    found.sort(key=lambda item: (not item.get("supported", True), item.get("retune", False)))
    return {"status": "ok", "format": config.get("format"), "matches": found}


def config_for_job(job_id: str) -> dict:
    meta = jobs_mod.read_meta(job_id)
    output_dir = Path(meta.output_dir)
    if not output_dir.is_absolute():
        output_dir = paths.REPO_ROOT / output_dir
    path = output_dir / QUANT_CONFIG_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; the artifact declares no saved format")
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description="Shared kernel library")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="Every entry")
    sub.add_parser("check", help="Validate every kernel.json and the files it points at")
    match_p = sub.add_parser("match", help="Entries that serve a job's saved format")
    source = match_p.add_mutually_exclusive_group(required=True)
    source.add_argument("--job-id")
    source.add_argument("--quant-config", type=Path, help="a quantization_config.json")
    match_p.add_argument("--compute-capability", help="e.g. 8.6; flags entries that need retuning")

    args = parser.parse_args()
    try:
        if args.command == "list":
            payload = {
                "status": "ok",
                "entries": [entry.model_dump(exclude_none=True) for _, entry in load_entries()],
            }
        elif args.command == "check":
            payload = check()
        else:
            config = (
                config_for_job(args.job_id) if args.job_id
                else json.loads(args.quant_config.read_text())
            )
            payload = match(config, compute_capability=args.compute_capability)
    except (ValueError, ValidationError, OSError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, indent=2))
        return 1
    print(json.dumps(payload, indent=2))
    return 0 if payload.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())

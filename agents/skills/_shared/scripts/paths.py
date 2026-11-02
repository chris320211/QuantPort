"""Workspace paths for skill scripts. No quant_agent package."""
from __future__ import annotations

import os
import re
from pathlib import Path

_SKILLS_ROOT = Path(__file__).resolve().parents[1]  # agents/skills/_shared
_REPO_CANDIDATE = _SKILLS_ROOT.parents[2]  # repo root from agents/skills/_shared/scripts

REPO_ROOT = Path(os.environ.get("QUANT_AGENT_WORKSPACE", _REPO_CANDIDATE)).expanduser().resolve()
OUT_ROOT = REPO_ROOT / "out"
REQUESTS_ROOT = OUT_ROOT / "requests"
OVERLAYS_ROOT = OUT_ROOT / "overlays"
ANALYSIS_ROOT = OUT_ROOT / "analysis"
SMOKE_ROOT = OUT_ROOT / "smoke"
VENV_ROOT = REPO_ROOT / ".venvs"
JOBS_ROOT = REPO_ROOT / "jobs"
PAPER_CACHE = REPO_ROOT / ".cache" / "papers"
HF_SNAP_ROOT = REPO_ROOT / ".cache" / "hf-snapshots"
EVAL_CACHE = REPO_ROOT / ".cache" / "eval"
SMOKE_SLICES = REPO_ROOT / ".cache" / "smoke-slices"
LIBRARY_ROOT = REPO_ROOT / "library"

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,79}$")


def require_slug(slug: str) -> str:
    if not isinstance(slug, str) or not SLUG_RE.fullmatch(slug):
        raise ValueError(f"invalid slug: {slug!r}")
    return slug


def contained_in(path: Path, root: Path) -> bool:
    resolved = path.expanduser().resolve()
    base = root.expanduser().resolve()
    return resolved == base or base in resolved.parents


def venv_python(slug: str) -> Path:
    """Return ``.venvs/<slug>/bin/python`` without following the interpreter symlink.

    A normal venv's ``bin/python`` points at the system interpreter (e.g.
    ``/usr/bin/python3.10``). Resolving that symlink and then requiring the
    target to live under ``.venvs`` treats every real venv as a path escape.
    Containment is on the venv directory; the returned path is the in-venv
    ``bin/python`` entry (still a symlink).
    """
    require_slug(slug)
    venv_dir = (VENV_ROOT / slug).expanduser().resolve()
    if not contained_in(venv_dir, VENV_ROOT):
        raise ValueError(f"venv python escaped .venvs: {venv_dir}")
    return venv_dir / "bin" / "python"

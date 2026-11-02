#!/usr/bin/env python3
"""Reviewable, content-addressed overlays for model-family ports.

The canonical method checkout is never edited. A port worker writes a unified
diff bundle under ``out/overlays/<slug>/<strategy>/<hash>/``. Launch applies
that diff to a temporary detached Git worktree, never the ``.venvs`` clone.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapter import InvalidInferenceAdapter, adapter_loads_hub_id, validate_adapter_file, validate_adapter_source
from env import child_env
from io_utils import atomic_write_text
import paths

_MAX_PATCH_BYTES = 200_000
_DIFF_HEADER_RE = re.compile(r"^diff --git a/(\S+) b/(\S+)$", re.MULTILINE)
_OVERLAY_HEADER_RE = re.compile(r"^# QUANT_AGENT_OVERLAY_DIR=(.+)$", re.MULTILINE)
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")
DIFFERENCE_ID_RE = re.compile(r"^D[0-9]{1,3}$")


class PortOverlayManifest(BaseModel):
    schema_version: int = 1
    method_id: str
    model_id: str
    base_commit: str
    patch_sha256: str
    rationale: str = Field(..., min_length=1, max_length=4000)
    evidence_files: list[str] = Field(default_factory=list, max_length=20)
    target_modules: list[str] = Field(default_factory=list, max_length=200)
    inference_adapter_path: str | None = None
    # Difference ids from out/analysis/<slug>/differences.json this patch handles.
    addresses: list[str] = Field(default_factory=list, max_length=100)


def require_difference_ids(ids: list[str]) -> list[str]:
    bad = [item for item in ids if not DIFFERENCE_ID_RE.fullmatch(str(item))]
    if bad:
        raise ValueError(f"difference ids must look like D1, D12: {bad}")
    return sorted(set(ids), key=lambda item: int(item[1:]))


def require_base_commit(commit: str) -> str:
    if not isinstance(commit, str) or not _COMMIT_RE.fullmatch(commit):
        raise ValueError("port overlay requires an exact 40-character base commit")
    return commit


def safe_path_component(name: str, label: str) -> str:
    if (
        not isinstance(name, str)
        or not _SAFE_COMPONENT_RE.fullmatch(name)
        or name in {".", ".."}
    ):
        raise ValueError(f"unsafe {label}: {name!r}")
    return name


def _safe_patch_path(raw: str) -> bool:
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        return False
    if path.parts[0] in {".git", ".env"} or any(part.startswith(".env") for part in path.parts):
        return False
    return True


def validate_unified_patch(patch: str) -> list[str]:
    """Return patched repository-relative paths or raise on an unsafe/invalid diff."""
    if not isinstance(patch, str) or not patch.strip():
        raise ValueError("overlay patch must be non-empty")
    if len(patch.encode("utf-8")) > _MAX_PATCH_BYTES:
        raise ValueError(f"overlay patch exceeds {_MAX_PATCH_BYTES} bytes")
    if "GIT binary patch" in patch or "Binary files " in patch:
        raise ValueError("binary overlay patches are not supported")
    for line in patch.splitlines():
        stripped = line.strip()
        if stripped.startswith("new file mode 120000") or stripped.startswith("new file mode 160000"):
            raise ValueError("symlink and gitlink overlay patches are not supported")
        if stripped.startswith("new mode 120000") or stripped.startswith("new mode 160000"):
            raise ValueError("symlink and gitlink overlay patches are not supported")
        if "mode 120000" in stripped or "mode 160000" in stripped:
            raise ValueError("symlink and gitlink overlay patches are not supported")
    pairs = _DIFF_HEADER_RE.findall(patch)
    if not pairs:
        raise ValueError("overlay must be a git unified diff with diff --git headers")
    paths_out: list[str] = []
    for left, right in pairs:
        if left != right:
            raise ValueError("renames are not supported in port overlays")
        if not _safe_patch_path(left):
            raise ValueError(f"unsafe overlay path: {left!r}")
        paths_out.append(left)
    for line in patch.splitlines():
        if line.startswith(("--- ", "+++ ")):
            value = line[4:].split("\t", 1)[0]
            if value == "/dev/null":
                continue
            if not value.startswith(("a/", "b/")) or not _safe_patch_path(value[2:]):
                raise ValueError(f"unsafe unified diff path: {value!r}")
    return paths_out


def directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(file.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(file.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def extract_new_file(patch: str, relpath: str) -> str | None:
    """Return the added file body from a new-file unified diff, or None if not a pure add."""
    lines = patch.splitlines()
    collecting = False
    is_new_file = False
    body: list[str] = []
    for line in lines:
        header = _DIFF_HEADER_RE.match(line)
        if header:
            if collecting:
                break
            collecting = header.group(1) == relpath and header.group(2) == relpath
            is_new_file = False
            body = []
            continue
        if not collecting:
            continue
        if line.startswith("--- ") and line[4:].split("\t", 1)[0] == "/dev/null":
            is_new_file = True
            continue
        if line.startswith("+") and not line.startswith("+++"):
            body.append(line[1:])
        elif line.startswith("\\"):
            continue
        elif line.startswith("-") and not line.startswith("---"):
            return None
    if not collecting or not is_new_file:
        return None
    text = "\n".join(body)
    if body:
        text += "\n"
    return text


def _maybe_validate_adapter_from_patch(patch: str, inference_adapter_path: str | None) -> None:
    if inference_adapter_path is None:
        return
    extracted = extract_new_file(patch, inference_adapter_path)
    if extracted is None:
        return
    validate_adapter_source(extracted)
    if adapter_loads_hub_id(extracted):
        raise InvalidInferenceAdapter(
            "inference adapter loads Hub model_id instead of the saved artifact"
        )


def validate_overlay_bundle(path: Path) -> PortOverlayManifest:
    resolved = path.resolve()
    patch_path = resolved / "overlay.patch"
    manifest_path = resolved / "manifest.json"
    if not resolved.is_dir() or not patch_path.is_file() or not manifest_path.is_file():
        raise ValueError(f"invalid port overlay bundle: {resolved}")
    manifest = PortOverlayManifest.model_validate_json(manifest_path.read_text())
    require_base_commit(manifest.base_commit)
    require_difference_ids(manifest.addresses)
    patch = patch_path.read_text()
    patched_paths = validate_unified_patch(patch)
    actual = hashlib.sha256(patch.encode("utf-8")).hexdigest()
    if actual != manifest.patch_sha256:
        raise ValueError("port overlay patch hash does not match its manifest")
    if manifest.inference_adapter_path is not None:
        if not _safe_patch_path(manifest.inference_adapter_path):
            raise ValueError("overlay inference adapter path is unsafe")
        if manifest.inference_adapter_path not in patched_paths:
            raise ValueError(
                "overlay inference adapter path must be included in the unified diff"
            )
    return manifest


def overlay_path_from_script(code: str) -> Path | None:
    match = _OVERLAY_HEADER_RE.search(code)
    if not match:
        return None
    raw = match.group(1).strip()
    return Path(raw).expanduser().resolve()


def validate_overlay_script(code: str, overlay_dir: Path) -> None:
    """Require the generated wrapper contract that preserves the canonical checkout."""
    expected_header = f"# QUANT_AGENT_OVERLAY_DIR={overlay_dir.resolve()}"
    if expected_header not in code.splitlines()[:20]:
        raise ValueError("generated port script lacks the exact overlay header")
    validate_overlay_bundle(overlay_dir)
    tree = ast.parse(code)
    literals = {
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    required = {"QUANT_AGENT_OVERLAY_DIR", "QUANT_AGENT_METHOD_REPO"}
    missing = sorted(required - literals)
    if missing:
        raise ValueError(
            "generated port script lacks executor-managed overlay environment usage: "
            + ", ".join(missing)
        )


class PortOverlaySession:
    """Single-success overlay writer with validation and atomic file writes."""

    def __init__(
        self,
        *,
        root: Path,
        method_id: str,
        model_id: str,
        base_commit: str,
    ) -> None:
        self.root = root.resolve()
        self.method_id = method_id
        self.model_id = model_id
        self.base_commit = require_base_commit(base_commit)
        self.overlay_dir: Path | None = None
        self.manifest: PortOverlayManifest | None = None

    def write(
        self,
        *,
        patch: str,
        rationale: str,
        evidence_files: list[str] | None = None,
        target_modules: list[str] | None = None,
        inference_adapter_path: str | None = None,
        addresses: list[str] | None = None,
    ) -> dict:
        if self.overlay_dir is not None:
            return {"status": "error", "message": "port overlay is already finalized"}
        try:
            _assert_not_venv_write(self.root)
            addresses = require_difference_ids(addresses or [])
            patched_paths = validate_unified_patch(patch)
            if inference_adapter_path is not None:
                if not _safe_patch_path(inference_adapter_path):
                    raise ValueError("overlay inference adapter path is unsafe")
                if inference_adapter_path not in patched_paths:
                    raise ValueError(
                        "overlay inference adapter path must be included in the unified diff"
                    )
                _maybe_validate_adapter_from_patch(patch, inference_adapter_path)
            patch_sha = hashlib.sha256(patch.encode("utf-8")).hexdigest()
            manifest = PortOverlayManifest(
                method_id=self.method_id,
                model_id=self.model_id,
                base_commit=self.base_commit,
                patch_sha256=patch_sha,
                rationale=rationale,
                evidence_files=evidence_files or [],
                target_modules=target_modules or [],
                inference_adapter_path=inference_adapter_path,
                addresses=addresses,
            )
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

        identity_parts = [
            self.method_id,
            self.model_id,
            self.base_commit,
            patch_sha,
            inference_adapter_path or "",
        ]
        if addresses:
            # Only when set, so bundles written before `addresses` keep their hash.
            identity_parts.append(",".join(addresses))
        bundle_identity = "\0".join(identity_parts)
        bundle_sha = hashlib.sha256(bundle_identity.encode("utf-8")).hexdigest()
        final_dir = self.root / bundle_sha[:16]
        if not final_dir.exists():
            self.root.mkdir(parents=True, exist_ok=True)
            temp_dir = self.root / f".{bundle_sha[:16]}.{secrets.token_hex(6)}.tmp"
            temp_dir.mkdir(mode=0o700)
            try:
                atomic_write_text(temp_dir / "overlay.patch", patch)
                atomic_write_text(temp_dir / "manifest.json", manifest.model_dump_json(indent=2))
                os.replace(temp_dir, final_dir)
            finally:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
        stored_manifest = validate_overlay_bundle(final_dir)
        self.overlay_dir = final_dir
        self.manifest = stored_manifest
        return {
            "status": "ok",
            "overlay_dir": str(final_dir),
            "patch_sha256": patch_sha,
            "patched_paths": patched_paths,
            "manifest": stored_manifest.model_dump(),
        }


def _venv_root() -> Path:
    return paths.VENV_ROOT.resolve()


def _is_under_venvs(path: Path) -> bool:
    resolved = path.resolve()
    venv = _venv_root()
    return resolved == venv or venv in resolved.parents


def _assert_not_venv_write(path: Path) -> None:
    if _is_under_venvs(path):
        raise RuntimeError(f"refusing to write into the .venvs clone: {path}")


def _git_available() -> bool:
    return shutil.which("git") is not None


def _run_git(args: list[str], *, timeout: int = 120) -> None:
    try:
        result = subprocess.run(
            ["git", "-c", "core.symlinks=false", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=child_env(include_hf=False),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"git overlay operation failed to start: {exc}") from exc
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "git operation failed").splitlines()[-20:]
        raise RuntimeError("git overlay operation failed:\n" + "\n".join(tail))


def cleanup_overlay_worktree(worktree: Path, repo: Path | None = None) -> None:
    if repo is not None and repo.exists() and _git_available():
        try:
            subprocess.run(
                ["git", "-C", str(repo), "worktree", "remove", "--force", str(worktree)],
                capture_output=True, text=True, timeout=30, env=child_env(include_hf=False),
            )
            subprocess.run(
                ["git", "-C", str(repo), "worktree", "prune"],
                capture_output=True, text=True, timeout=30, env=child_env(include_hf=False),
            )
        except (OSError, subprocess.SubprocessError):
            pass
    if worktree.exists():
        shutil.rmtree(worktree, ignore_errors=True)


def prepare_overlay_worktree(
    *,
    repo: Path,
    overlay_manifest: PortOverlayManifest,
    overlay_snapshot: Path,
    worktree: Path,
) -> Path:
    """Apply a validated overlay to a detached worktree, never the canonical clone."""
    repo = repo.resolve()
    worktree = worktree.resolve()
    _assert_not_venv_write(worktree)
    if worktree == repo:
        raise RuntimeError("refusing to apply overlay to the canonical clone")
    git_marker = repo / ".git"
    if not git_marker.exists():
        raise RuntimeError(f"canonical method checkout is missing: {repo}")
    commit = require_base_commit(overlay_manifest.base_commit)
    if not _git_available():
        raise RuntimeError("git is required to apply a port overlay")
    try:
        _run_git(["-C", str(repo), "cat-file", "-e", f"{commit}^{{commit}}"])
        _run_git([
            "-C", str(repo), "worktree", "add", "--detach", str(worktree), commit,
        ])
        patch = overlay_snapshot / "overlay.patch"
        _run_git(["-C", str(worktree), "apply", "--check", str(patch)])
        _run_git(["-C", str(worktree), "apply", str(patch)])
        patched = validate_unified_patch(patch.read_text())
        for rel in patched:
            target = worktree / rel
            if target.is_symlink():
                raise RuntimeError(f"overlay produced a symlink: {rel}")
        if overlay_manifest.inference_adapter_path is not None:
            adapter = validate_adapter_file(worktree / overlay_manifest.inference_adapter_path)
            if adapter_loads_hub_id(adapter.read_text()):
                raise InvalidInferenceAdapter(
                    "inference adapter loads Hub model_id instead of the saved artifact"
                )
    except Exception:
        cleanup_overlay_worktree(worktree, repo)
        raise
    return worktree


def check_overlay_bundle(overlay_dir: Path) -> dict:
    """Validate a stored bundle. Does not modify any git checkout."""
    manifest = validate_overlay_bundle(overlay_dir)
    patch = (overlay_dir / "overlay.patch").read_text()
    patched_paths = validate_unified_patch(patch)
    adapter_validated = False
    if manifest.inference_adapter_path is not None:
        if manifest.inference_adapter_path not in patched_paths:
            raise ValueError(
                "overlay inference adapter path must be included in the unified diff"
            )
        extracted = extract_new_file(patch, manifest.inference_adapter_path)
        if extracted is not None:
            validate_adapter_source(extracted)
            if adapter_loads_hub_id(extracted):
                raise InvalidInferenceAdapter(
                    "inference adapter loads Hub model_id instead of the saved artifact"
                )
            adapter_validated = True
    return {
        "status": "ok",
        "overlay_dir": str(overlay_dir.resolve()),
        "patched_paths": patched_paths,
        "adapter_in_patch": manifest.inference_adapter_path in patched_paths
        if manifest.inference_adapter_path
        else None,
        "adapter_validated": adapter_validated,
        "manifest": manifest.model_dump(),
    }


def apply_check(
    overlay_dir: Path,
    *,
    repo: Path | None = None,
    commit: str | None = None,
) -> dict:
    """``git apply --check`` in a temp worktree when repo+commit+git exist.

    If ``repo`` is omitted, validate the patch text only. If ``repo`` is given,
    git must be available; never report apply-check success without git apply.
    Never writes to the canonical ``.venvs`` clone.
    """
    overlay_dir = overlay_dir.resolve()
    bundle = check_overlay_bundle(overlay_dir)
    manifest = PortOverlayManifest.model_validate(bundle["manifest"])
    patch_path = overlay_dir / "overlay.patch"
    target_commit = require_base_commit(commit or manifest.base_commit)
    if repo is None:
        return {
            **bundle,
            "mode": "patch-text",
            "git": False,
            "message": "repo not provided; validated overlay patch text only",
        }
    if not _git_available():
        raise RuntimeError("git is required to apply-check against a repo")

    repo = repo.resolve()
    git_marker = repo / ".git"
    if not git_marker.exists():
        raise RuntimeError(f"canonical method checkout is missing: {repo}")
    with TemporaryDirectory(prefix="quant-overlay-apply-check-") as tmp:
        worktree = Path(tmp) / "worktree"
        _assert_not_venv_write(worktree)
        try:
            _run_git(["-C", str(repo), "cat-file", "-e", f"{target_commit}^{{commit}}"])
            _run_git([
                "-C", str(repo), "worktree", "add", "--detach", str(worktree), target_commit,
            ])
            _run_git(["-C", str(worktree), "apply", "--check", str(patch_path)])
            adapter_validated = False
            if manifest.inference_adapter_path is not None:
                _run_git(["-C", str(worktree), "apply", str(patch_path)])
                validate_adapter_file(worktree / manifest.inference_adapter_path)
                adapter_validated = True
            return {
                **bundle,
                "mode": "git-apply",
                "git": True,
                "commit": target_commit,
                "adapter_validated": adapter_validated or bundle.get("adapter_validated"),
            }
        finally:
            cleanup_overlay_worktree(worktree, repo)
            if (repo / ".git").exists():
                try:
                    subprocess.run(
                        ["git", "-C", str(repo), "worktree", "prune"],
                        capture_output=True, text=True, timeout=30,
                        env=child_env(include_hf=False),
                    )
                except (OSError, subprocess.SubprocessError):
                    pass


def _print(payload: dict) -> int:
    print(json.dumps(payload, indent=2))
    return 0 if payload.get("status") == "ok" else 1


def _cmd_write(args: argparse.Namespace) -> dict:
    slug = safe_path_component(args.slug, "slug")
    strategy = safe_path_component(args.strategy, "strategy")
    root = paths.OVERLAYS_ROOT / slug / strategy
    _assert_not_venv_write(root)
    session = PortOverlaySession(
        root=root,
        method_id=slug,
        model_id=args.model_id,
        base_commit=args.base_commit,
    )
    return session.write(
        patch=args.patch_file.read_text(),
        rationale=args.rationale,
        evidence_files=args.evidence_file,
        target_modules=args.target_module,
        inference_adapter_path=args.inference_adapter_path,
        addresses=args.addresses,
    )


def _cmd_check(args: argparse.Namespace) -> dict:
    return check_overlay_bundle(args.overlay_dir)


def _cmd_apply_check(args: argparse.Namespace) -> dict:
    return apply_check(args.overlay_dir, repo=args.repo, commit=args.commit)


def main() -> int:
    parser = argparse.ArgumentParser(description="Write and validate port overlay bundles")
    sub = parser.add_subparsers(dest="command", required=True)

    write_p = sub.add_parser("write", help="Write a content-addressed overlay bundle")
    write_p.add_argument("--slug", required=True)
    write_p.add_argument("--strategy", required=True)
    write_p.add_argument("--model-id", required=True)
    write_p.add_argument("--base-commit", required=True)
    write_p.add_argument("--patch-file", type=Path, required=True)
    write_p.add_argument("--rationale", required=True)
    write_p.add_argument("--inference-adapter-path")
    write_p.add_argument("--evidence-file", action="append", default=[])
    write_p.add_argument("--target-module", action="append", default=[])
    write_p.add_argument(
        "--addresses", action="append", default=[],
        help="Difference id (D1, D2, ...) from the port spec this patch handles",
    )

    check_p = sub.add_parser("check", help="Validate an overlay bundle without applying it")
    check_p.add_argument("--overlay-dir", type=Path, required=True)

    apply_p = sub.add_parser(
        "apply-check",
        help="git apply --check in a temp worktree when repo+commit are given",
    )
    apply_p.add_argument("--overlay-dir", type=Path, required=True)
    apply_p.add_argument("--repo", type=Path)
    apply_p.add_argument("--commit")

    args = parser.parse_args()
    try:
        if args.command == "write":
            payload = _cmd_write(args)
        elif args.command == "check":
            payload = _cmd_check(args)
        else:
            payload = _cmd_apply_check(args)
    except (ValueError, InvalidInferenceAdapter, RuntimeError, OSError) as exc:
        payload = {"status": "error", "message": str(exc)}
    return _print(payload)


if __name__ == "__main__":
    raise SystemExit(main())

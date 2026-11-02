#!/usr/bin/env python3
"""Clone a GitHub repo into .venvs/<slug>/repo, with its git submodules.

Submodules are fetched only from GitHub (or a URL relative to the GitHub
origin); any other URL is skipped and listed in ``submodules_skipped``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
from contextlib import contextmanager

from env import child_env, host_execution_policy, require_host_execution
from paths import VENV_ROOT, require_slug

_GITHUB_URL_RE = re.compile(r"^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/?$")
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,79}$")
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_RELATIVE_URL_RE = re.compile(r"^(\.\./)+[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)?/?$")
_GITMODULES_SECTION_RE = re.compile(r'^\[submodule "(.+)"\]$')
_SUBMODULE_DEPTH = 3
_CLONE_TIMEOUT = 300
_GIT_TIMEOUT = 15


def _normalize_url(repo_url: str) -> str:
    return repo_url.strip().rstrip("/")


def _validate_slug(slug: str) -> str | None:
    if not _SLUG_RE.fullmatch(slug):
        return f"invalid slug: {slug!r}"
    return None


@contextmanager
def _slug_lock(slug: str):
    lock_dir = VENV_ROOT / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{slug}.lock"
    with lock_path.open("a+") as lock:
        os.chmod(lock_path, 0o600)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _git(argv: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    require_host_execution("git")
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=child_env(include_hf=False),
    )


def _head_sha(repo: Path) -> str | None:
    head = _git(["git", "-C", str(repo), "rev-parse", "HEAD"], _GIT_TIMEOUT)
    sha = (head.stdout or "").strip().lower()
    if head.returncode != 0 or not _SHA_RE.fullmatch(sha):
        return None
    return sha


def parse_gitmodules(text: str) -> list[dict]:
    """Submodules declared in a .gitmodules file, as {name, path, url}."""
    modules: list[dict] = []
    current: dict | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";")):
            continue
        section = _GITMODULES_SECTION_RE.match(line)
        if section:
            current = {"name": section.group(1)}
            modules.append(current)
        elif line.startswith("["):
            current = None
        elif current is not None and "=" in line:
            key, value = (part.strip() for part in line.split("=", 1))
            if key in ("path", "url"):
                current[key] = value
    return [m for m in modules if m.get("path") and m.get("url")]


def submodule_skip_reason(module: dict) -> str | None:
    path = Path(module["path"])
    if path.is_absolute() or ".." in path.parts:
        return "path escapes the checkout"
    url = module["url"]
    if _GITHUB_URL_RE.match(url) or _RELATIVE_URL_RE.match(url):
        return None
    return "url is not GitHub or relative to the GitHub origin"


def _update_submodule(repo: Path, path: str) -> subprocess.CompletedProcess[str]:
    base = ["git", "-C", str(repo), "-c", "protocol.file.allow=never", "-c", "core.symlinks=false",
            "submodule", "update", "--init"]
    result = _git([*base, "--depth", "1", "--", path], _CLONE_TIMEOUT)
    if result.returncode != 0:
        # The pinned commit may not be reachable from a shallow fetch.
        result = _git([*base, "--", path], _CLONE_TIMEOUT)
    return result


def init_submodules(repo: Path, depth: int = 0, prefix: str = "") -> dict:
    """Fetch every allowed submodule, then theirs; idempotent on a ready checkout."""
    out: dict = {"submodules": [], "submodules_skipped": []}
    gitmodules = repo / ".gitmodules"
    if not gitmodules.is_file():
        return out
    for module in parse_gitmodules(gitmodules.read_text(errors="replace")):
        shown = prefix + module["path"]
        reason = submodule_skip_reason(module)
        if reason is None and depth >= _SUBMODULE_DEPTH:
            reason = f"nested deeper than {_SUBMODULE_DEPTH}"
        if reason:
            out["submodules_skipped"].append({"path": shown, "url": module["url"], "reason": reason})
            continue
        try:
            result = _update_submodule(repo, module["path"])
        except subprocess.TimeoutExpired:
            return {**out, "error": f"submodule {shown} timed out after {_CLONE_TIMEOUT}s"}
        if result.returncode != 0:
            err = (result.stderr or result.stdout or "submodule update failed").strip()
            return {**out, "error": f"submodule {shown}: {err}"}
        out["submodules"].append(shown)
        nested = init_submodules(repo / module["path"], depth + 1, f"{shown}/")
        out["submodules"] += nested["submodules"]
        out["submodules_skipped"] += nested["submodules_skipped"]
        if "error" in nested:
            return {**out, "error": nested["error"]}
    return out


def _with_submodules(repo: Path, payload: dict) -> dict:
    result = init_submodules(repo)
    if "error" in result:
        return {"status": "error", "error": result["error"], "path": str(repo)}
    if result["submodules"]:
        payload["submodules"] = result["submodules"]
    if result["submodules_skipped"]:
        payload["submodules_skipped"] = result["submodules_skipped"]
    return payload


def clone_repo(slug: str, repo_url: str) -> dict:
    try:
        require_slug(slug)
    except ValueError as exc:
        return {"status": "error", "error": str(exc)}

    raw = repo_url.strip()
    if not _GITHUB_URL_RE.match(raw):
        return {
            "status": "error",
            "error": f"repo_url is not a plain GitHub repo URL: {repo_url!r}",
        }
    expected = _normalize_url(raw)

    with _slug_lock(slug):
        venv_dir = VENV_ROOT / slug
        repo = venv_dir / "repo"
        venv_dir.mkdir(parents=True, exist_ok=True)

        if repo.exists() and (repo / ".git").exists():
            origin = _git(["git", "-C", str(repo), "remote", "get-url", "origin"], _GIT_TIMEOUT)
            actual = _normalize_url((origin.stdout or "").strip())
            if origin.returncode != 0 or actual != expected:
                return {
                    "status": "error",
                    "error": f"existing checkout origin mismatch: expected {expected!r}, got {actual!r}",
                }
            sha = _head_sha(repo)
            if sha is None:
                return {"status": "error", "error": "existing checkout HEAD is not a 40-char SHA"}
            return _with_submodules(repo, {
                "status": "ok",
                "path": str(repo),
                "commit_sha": sha,
                "repo_url": expected,
                "already_present": True,
            })

        if repo.exists():
            shutil.rmtree(repo)

        try:
            result = _git(
                ["git", "-c", "core.symlinks=false", "clone", "--depth", "1", raw, str(repo)],
                _CLONE_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return {"status": "error", "error": f"git clone timed out after {_CLONE_TIMEOUT}s"}
        if result.returncode != 0:
            err = (result.stderr or result.stdout or "clone failed").strip()
            return {"status": "error", "error": err}

        origin = _git(["git", "-C", str(repo), "remote", "get-url", "origin"], _GIT_TIMEOUT)
        actual = _normalize_url((origin.stdout or "").strip())
        if origin.returncode != 0 or actual != expected:
            return {
                "status": "error",
                "error": f"cloned origin mismatch: expected {expected!r}, got {actual!r}",
            }
        sha = _head_sha(repo)
        if sha is None:
            return {"status": "error", "error": "cloned HEAD is not a 40-char SHA"}
        return _with_submodules(repo, {
            "status": "ok",
            "path": str(repo),
            "commit_sha": sha,
            "repo_url": expected,
            "already_present": False,
        })


def main() -> int:
    parser = argparse.ArgumentParser(description="Clone a GitHub repo into .venvs/<slug>/repo")
    parser.add_argument("--slug", required=True)
    parser.add_argument("--repo-url", required=True)
    parser.add_argument("--allow-unsafe-host-execution", action="store_true")
    args = parser.parse_args()
    try:
        with host_execution_policy(args.allow_unsafe_host_execution):
            payload = clone_repo(args.slug, args.repo_url)
    except Exception as exc:
        payload = {"status": "error", "error": str(exc)}
    print(json.dumps(payload, indent=2))
    return 0 if payload.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

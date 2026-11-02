#!/usr/bin/env python3
"""Create .venvs/<slug> and install torch, runtime pins, and extra pip/python steps."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import fcntl
import json
import os
import re
import shlex
import subprocess
from contextlib import contextmanager

from env import child_env, host_execution_policy, require_host_execution
from paths import VENV_ROOT, require_slug
from runtime_pins import RUNTIME_PACKAGES
from torch_spec import detect_torch_spec

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,79}$")
_INSTALL_TIMEOUT = 900
_OUTPUT_TAIL_LINES = 60
_SAFE_ENV_ASSIGNMENTS = {"TORCH_CUDA_ARCH_LIST", "MAX_JOBS"}


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


def _tail(text: str, n: int = _OUTPUT_TAIL_LINES) -> str:
    lines = text.splitlines()
    if len(lines) <= n:
        return text
    return "...(%d earlier lines)\n%s" % (len(lines) - n, "\n".join(lines[-n:]))


def _parse_install_step(command: str, python: Path) -> tuple[list[str], dict[str, str]]:
    """Turn a narrow command string into argv without invoking a shell."""
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command must be a non-empty string")
    if any(token in command for token in ("\n", "\r", ";", "&&", "||", "|", ">", "<", "`", "$(")):
        raise ValueError("shell operators and substitutions are not allowed")
    try:
        parts = shlex.split(command, posix=True)
    except ValueError as e:
        raise ValueError(f"invalid command quoting: {e}") from e
    if not parts:
        raise ValueError("command must be non-empty")

    extra: dict[str, str] = {}
    while parts and "=" in parts[0] and not parts[0].startswith(("-", "./", "/")):
        key, value = parts.pop(0).split("=", 1)
        if key not in _SAFE_ENV_ASSIGNMENTS:
            raise ValueError(f"environment assignment {key!r} is not allowed")
        extra[key] = value
    if not parts:
        raise ValueError("command contains no executable")

    executable, *args = parts
    if executable in {"python", "python3", str(python)}:
        return _with_no_build_isolation_for_local_install([str(python), *args]), extra
    if executable in {"pip", "pip3"}:
        return _with_no_build_isolation_for_local_install([str(python), "-m", "pip", *args]), extra
    raise ValueError(f"executable {executable!r} is not allowed; use python or pip")


def _pip_install_args(argv: list[str]) -> list[str] | None:
    """Return tokens after `pip install` in a python -m pip argv, else None."""
    try:
        pip_at = argv.index("pip")
        install_at = argv.index("install", pip_at + 1)
    except ValueError:
        return None
    if pip_at < 1 or argv[pip_at - 1] != "-m":
        return None
    return argv[install_at + 1 :]


def _is_local_project_install(argv: list[str]) -> bool:
    """True for `pip install -e .` / local paths, not PyPI names or -r files."""
    args = _pip_install_args(argv)
    if args is None:
        return False
    skip_next = False
    editable = False
    targets: list[str] = []
    for token in args:
        if skip_next:
            skip_next = False
            continue
        if token in {"-e", "--editable"}:
            editable = True
            continue
        if token.startswith("--editable="):
            editable = True
            targets.append(token.split("=", 1)[1])
            continue
        if token in {"-r", "--requirement", "-c", "--constraint"}:
            skip_next = True
            continue
        if token.startswith("-"):
            continue
        targets.append(token)
    if editable:
        return True
    return any(
        target in {".", "./"}
        or target.startswith(("./", "/", "file:"))
        for target in targets
    )


def _with_no_build_isolation_for_local_install(argv: list[str]) -> list[str]:
    """Local setup.py often `import torch`; pip isolation hides the venv torch pin."""
    if "--no-build-isolation" in argv:
        return argv
    if not _is_local_project_install(argv):
        return argv
    out = list(argv)
    out.insert(out.index("install") + 1, "--no-build-isolation")
    return out


def _run_argv(
    cmd: list[str],
    cwd: Path | None,
    timeout: int,
    env_extra: dict[str, str] | None = None,
    scratch_home: Path | None = None,
) -> dict:
    require_host_execution("subprocess command")
    env = child_env(env_extra, include_hf=False, scratch_home=scratch_home)
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as e:
        return {
            "ok": False,
            "exit_code": -1,
            "stdout": _tail(e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")),
            "stderr": f"timed out after {timeout}s",
        }
    return {
        "ok": result.returncode == 0,
        "exit_code": result.returncode,
        "stdout": _tail(result.stdout or ""),
        "stderr": _tail(result.stderr or ""),
    }


def _baseline_packages(python: Path) -> list[list[str]]:
    spec = detect_torch_spec()
    return [
        # Python 3.12+ venvs ship without setuptools; setup.py repos need it
        # under --no-build-isolation.
        [str(python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        spec.pip_install_argv(str(python)),
        [str(python), "-m", "pip", "install", *RUNTIME_PACKAGES],
    ]


def _is_requirements_install(argv: list[str]) -> bool:
    args = _pip_install_args(argv)
    if args is None:
        return False
    return "-r" in args or "--requirement" in args


def _run_torch_pin(python: Path, repo: Path, scratch: Path) -> dict:
    spec = detect_torch_spec()
    argv = spec.pip_install_argv(str(python))
    ran = _run_argv(
        argv,
        cwd=repo,
        timeout=_INSTALL_TIMEOUT,
        scratch_home=scratch,
    )
    return {"step": f"reapply-torch-pin {spec.torch_pin}|{spec.cuda_tag}", "argv": argv, **ran}


def install_venv(slug: str, install_steps: list[str] | None = None) -> dict:
    try:
        require_slug(slug)
    except ValueError as exc:
        return {"status": "error", "error": str(exc)}

    venv_dir = VENV_ROOT / slug
    repo = venv_dir / "repo"
    py = venv_dir / "bin" / "python"
    scratch = venv_dir / ".scratch"
    steps_in = list(install_steps or [])

    with _slug_lock(slug):
        if not repo.exists():
            return {
                "status": "error",
                "error": f"repo not cloned: {repo}. Run clone.py first.",
            }

        if not py.exists():
            create = _run_argv(
                [sys.executable, "-m", "venv", str(venv_dir)],
                cwd=None,
                timeout=120,
                scratch_home=scratch,
            )
            if not create["ok"]:
                return {"status": "error", "stage": "venv-create", **create}

        parsed_steps: list[tuple[str, list[str], dict[str, str]]] = [
            (shlex.join(argv), argv, {}) for argv in _baseline_packages(py)
        ]
        for step in steps_in:
            try:
                argv, extra = _parse_install_step(step, py)
            except ValueError as e:
                return {"status": "error", "stage": "command-policy", "error": str(e)}
            parsed_steps.append((step, argv, extra))

        results: list[dict] = []
        for step, argv, extra in parsed_steps:
            if _is_local_project_install(argv):
                pin_ran = _run_torch_pin(py, repo, scratch)
                results.append(pin_ran)
                if not pin_ran["ok"]:
                    return {"status": "error", "stage": "install", "results": results}
            ran = _run_argv(
                argv,
                cwd=repo,
                timeout=_INSTALL_TIMEOUT,
                env_extra=extra,
                scratch_home=scratch,
            )
            results.append({"step": step, **ran})
            if not ran["ok"]:
                return {"status": "error", "stage": "install", "results": results}
            if _is_requirements_install(argv):
                pin_ran = _run_torch_pin(py, repo, scratch)
                results.append(pin_ran)
                if not pin_ran["ok"]:
                    return {"status": "error", "stage": "install", "results": results}
        return {
            "status": "ok",
            "python": str(py),
            "repo": str(repo),
            "results": results,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a method venv and run pip/python install steps")
    parser.add_argument("--slug", required=True)
    parser.add_argument("--allow-unsafe-host-execution", action="store_true")
    parser.add_argument("--step", action="append", default=[])
    args = parser.parse_args()
    try:
        with host_execution_policy(args.allow_unsafe_host_execution):
            payload = install_venv(args.slug, args.step)
    except Exception as exc:
        payload = {"status": "error", "error": str(exc)}
    print(json.dumps(payload, indent=2))
    return 0 if payload.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

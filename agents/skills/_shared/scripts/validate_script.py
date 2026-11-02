#!/usr/bin/env python3
"""AST + overlay-header validation for generated quantization scripts.

Does not execute third-party code unless --dry-import is passed with
--allow-unsafe-host-execution.
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from env import child_env, host_execution_policy, require_host_execution
from overlay import validate_overlay_script
from paths import venv_python

# port_smoke.py points the wrapper at a 2-layer slice through these.
SMOKE_HOOKS = ("QUANT_AGENT_SMOKE_MODEL_DIR", "QUANT_AGENT_SMOKE_OUTPUT_DIR")


def _string_literals(tree: ast.AST) -> set[str]:
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def validate_script(
    code: str,
    *,
    model_id: str,
    output_dir: str,
    overlay_dir: Path | None = None,
    require_smoke_hooks: bool = False,
) -> dict:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {"ok": False, "stage": "ast", "message": str(exc)}
    literals = _string_literals(tree)
    missing: list[str] = []
    if model_id not in literals:
        missing.append("model_id")
    if output_dir not in literals:
        missing.append("output_dir")
    if require_smoke_hooks:
        missing.extend(hook for hook in SMOKE_HOOKS if hook not in literals)
    if missing:
        return {
            "ok": False,
            "stage": "literals",
            "message": "script is missing exact string literals: " + ", ".join(missing),
        }
    if overlay_dir is not None:
        try:
            validate_overlay_script(code, overlay_dir)
        except Exception as exc:
            return {"ok": False, "stage": "overlay", "message": str(exc)}
    return {"ok": True, "stage": "ok"}


def dry_import(script: Path, *, slug: str) -> dict:
    require_host_execution("script dry-import")
    py = venv_python(slug)
    if not py.is_file():
        return {"ok": False, "stage": "dry-import", "message": f"venv python missing: {py}"}
    probe = subprocess.run(
        [
            str(py),
            "-c",
            "import runpy, sys; runpy.run_path(sys.argv[1], run_name='__not_main__')",
            str(script),
        ],
        capture_output=True,
        text=True,
        env=child_env({"PYTHONDONTWRITEBYTECODE": "1"}, include_hf=False),
        timeout=60,
    )
    if probe.returncode != 0:
        tail = (probe.stderr or probe.stdout or "").strip().splitlines()[-20:]
        return {
            "ok": False,
            "stage": "dry-import",
            "message": "\n".join(tail) or f"dry-import exit {probe.returncode}",
        }
    return {"ok": True, "stage": "dry-import"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a generated quantization script")
    parser.add_argument("script", type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overlay-dir", type=Path)
    parser.add_argument("--smoke-hooks", action="store_true", help="require the port_smoke.py hooks")
    parser.add_argument("--dry-import", action="store_true")
    parser.add_argument("--slug", help="venv slug for --dry-import")
    parser.add_argument("--allow-unsafe-host-execution", action="store_true")
    args = parser.parse_args()
    code = args.script.read_text()
    payload = validate_script(
        code,
        model_id=args.model_id,
        output_dir=args.output_dir,
        overlay_dir=args.overlay_dir,
        require_smoke_hooks=args.smoke_hooks,
    )
    if payload["ok"] and args.dry_import:
        if not args.slug:
            payload = {
                "ok": False,
                "stage": "dry-import",
                "message": "--slug is required with --dry-import",
            }
        else:
            try:
                with host_execution_policy(args.allow_unsafe_host_execution):
                    payload = dry_import(args.script, slug=args.slug)
            except Exception as exc:
                payload = {"ok": False, "stage": "dry-import", "message": str(exc)}
    print(json.dumps(payload, indent=2))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

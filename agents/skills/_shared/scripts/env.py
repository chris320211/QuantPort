"""Host-execution gate and secret-free child environments."""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from threading import RLock

_HOST_EXECUTION_ALLOWED = False
_LOCK = RLock()

_CHILD_ENV_ALLOWLIST: tuple[str, ...] = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "TERM",
    "LD_LIBRARY_PATH",
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "TORCH_CUDA_ARCH_LIST",
    "HF_HOME",
    "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "XDG_CACHE_HOME",
)

_HOME_CACHE_KEYS = (
    "HOME",
    "HF_HOME",
    "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "XDG_CACHE_HOME",
)

_EXTRA_ALLOWED = frozenset({
    "QUANT_AGENT_OVERLAY_DIR",
    "QUANT_AGENT_METHOD_REPO",
    "PYTHONPATH",
    "MAX_JOBS",
    "TORCH_CUDA_ARCH_LIST",
    "MEASURE_MODEL_PATH",
    "MEASURE_MODEL_ID",
    "MEASURE_OUTPUT_JSON",
    "MEASURE_DTYPE",
    "MEASURE_ADAPTER_PATH",
    "MEASURE_TRUST_REMOTE_CODE",
    "MEASURE_EVAL_PATH",
    "MEASURE_MAX_SEQ_LEN",
    "MEASURE_MAX_TOKENS",
    "MEASURE_STRIDE",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONUNBUFFERED",
    "VLLM_NO_USAGE_STATS",
    "DO_NOT_TRACK",
})


@contextmanager
def host_execution_policy(allowed: bool):
    global _HOST_EXECUTION_ALLOWED
    with _LOCK:
        previous = _HOST_EXECUTION_ALLOWED
        _HOST_EXECUTION_ALLOWED = bool(allowed)
        try:
            yield
        finally:
            _HOST_EXECUTION_ALLOWED = previous


def require_host_execution(operation: str) -> None:
    if not _HOST_EXECUTION_ALLOWED:
        raise RuntimeError(
            f"{operation} would execute third-party/generated code on the host. "
            "Re-run with --allow-unsafe-host-execution only on an isolated disposable machine."
        )


def child_env(
    extra: dict[str, str] | None = None,
    *,
    include_hf: bool = False,
    scratch_home: Path | None = None,
) -> dict[str, str]:
    env = {k: os.environ[k] for k in _CHILD_ENV_ALLOWLIST if k in os.environ}
    if include_hf:
        tok = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
        if tok:
            env["HUGGINGFACE_HUB_TOKEN"] = tok
            env["HF_TOKEN"] = tok
    else:
        for key in _HOME_CACHE_KEYS:
            env.pop(key, None)
        env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
        if scratch_home is not None:
            scratch = Path(scratch_home)
            scratch.mkdir(parents=True, exist_ok=True)
            (scratch / "hf").mkdir(exist_ok=True)
            (scratch / "cache").mkdir(exist_ok=True)
            env["HOME"] = str(scratch)
            env["HF_HOME"] = str(scratch / "hf")
            env["XDG_CACHE_HOME"] = str(scratch / "cache")
            env["HUGGINGFACE_HUB_CACHE"] = str(scratch / "hf")
            env["TRANSFORMERS_CACHE"] = str(scratch / "hf")
    if extra:
        for key, value in extra.items():
            if key not in _EXTRA_ALLOWED:
                raise ValueError(f"child_env extra key is not allowed: {key}")
            env[key] = value
    return env

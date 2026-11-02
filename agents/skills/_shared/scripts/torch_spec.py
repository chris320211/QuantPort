"""Detect the right torch build for the local GPU and CUDA toolkit.

Extension builds compare nvcc's CUDA version to ``torch.version.cuda``. Driver
CUDA (nvidia-smi) is not enough: a 12.8 toolkit with a cu130 wheel fails.

Match **nvcc / CUDA toolkit** first, then GPU arch as a fallback for hosts
without a toolkit (laptop dry runs).

Known toolkit pins (wheels on download.pytorch.org):

- CUDA 12.1/12.2 → torch 2.3.1 cu121
- CUDA 12.4 → torch 2.4.1 cu124 (Hopper default when nvcc is 12.4)
- CUDA 12.6 → torch 2.7.1 cu126
- CUDA 12.8 → torch 2.7.1 cu128
- CUDA 13.0+ → torch 2.9.1 cu130 (first CUDA 13 builds; oldest is the safest
  for older method repos)

A toolkit newer than every known major uses the newest known pin rather than
the CUDA 12.1 default.

Override via env var ``QUANT_AGENT_TORCH_SPEC`` (format: ``torch==X.Y.Z|cuZZZ``)
so users can pin exotic combinations without editing the code.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class TorchSpec:
    torch_pin: str   # e.g. "torch==2.3.1"
    cuda_tag: str    # e.g. "cu121" — used to build the index-url

    @property
    def index_url(self) -> str:
        return f"https://download.pytorch.org/whl/{self.cuda_tag}"

    def pip_install(self) -> str:
        return f"pip install --index-url {self.index_url} {self.torch_pin}"

    def pip_install_argv(self, python: str) -> list[str]:
        """Safe argv form used by subprocess callers (no shell interpolation)."""
        return [python, "-m", "pip", "install", "--index-url", self.index_url, self.torch_pin]


_DEFAULT_SPEC = TorchSpec(torch_pin="torch==2.3.1", cuda_tag="cu121")
_HOPPER_SPEC = TorchSpec(torch_pin="torch==2.4.1", cuda_tag="cu124")

# Exact nvcc major.minor → wheel. Unknown minors round down within the same major.
_TOOLKIT_SPECS: dict[tuple[int, int], TorchSpec] = {
    (12, 1): TorchSpec(torch_pin="torch==2.3.1", cuda_tag="cu121"),
    (12, 2): TorchSpec(torch_pin="torch==2.3.1", cuda_tag="cu121"),
    (12, 4): TorchSpec(torch_pin="torch==2.4.1", cuda_tag="cu124"),
    (12, 6): TorchSpec(torch_pin="torch==2.7.1", cuda_tag="cu126"),
    (12, 8): TorchSpec(torch_pin="torch==2.7.1", cuda_tag="cu128"),
    (13, 0): TorchSpec(torch_pin="torch==2.9.1", cuda_tag="cu130"),
}

_NVCC_RELEASE_RE = re.compile(r"release\s+(\d+)\.(\d+)")


def _parse_override(raw: str) -> TorchSpec | None:
    raw = raw.strip()
    if "|" not in raw:
        return None
    pin, tag = raw.split("|", 1)
    pin = pin.strip()
    tag = tag.strip()
    if not re.fullmatch(r"torch==\d+\.\d+\.\d+(?:[A-Za-z0-9_.+-]*)?", pin):
        return None
    if not re.fullmatch(r"cu\d{3}", tag):
        return None
    return TorchSpec(torch_pin=pin, cuda_tag=tag)


def _compute_capability() -> float | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    first = r.stdout.strip().splitlines()
    if not first:
        return None
    try:
        return float(first[0].strip())
    except ValueError:
        return None


def _nvcc_release() -> tuple[int, int] | None:
    nvcc = shutil.which("nvcc")
    if not nvcc:
        return None
    try:
        r = subprocess.run(
            [nvcc, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    match = _NVCC_RELEASE_RE.search(r.stdout or "")
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def spec_for_toolkit(major: int, minor: int) -> TorchSpec | None:
    exact = _TOOLKIT_SPECS.get((major, minor))
    if exact is not None:
        return exact
    known = sorted(k for k in _TOOLKIT_SPECS if k[0] == major and k <= (major, minor))
    if known:
        return _TOOLKIT_SPECS[known[-1]]
    newest = max(_TOOLKIT_SPECS)
    if (major, minor) > newest:
        return _TOOLKIT_SPECS[newest]
    return None


def detect_torch_spec() -> TorchSpec:
    override_raw = os.environ.get("QUANT_AGENT_TORCH_SPEC", "").strip()
    if override_raw:
        parsed = _parse_override(override_raw)
        if parsed is not None:
            return parsed

    toolkit = _nvcc_release()
    if toolkit is not None:
        matched = spec_for_toolkit(*toolkit)
        if matched is not None:
            return matched

    cc = _compute_capability()
    if cc is not None and cc >= 9.0:
        return _HOPPER_SPEC
    return _DEFAULT_SPEC

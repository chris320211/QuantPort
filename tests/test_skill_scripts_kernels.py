"""Offline tests for the shared kernel library and its matcher; the GPU test runs only on CUDA."""
from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "agents" / "skills" / "_shared" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import kernels as kernels_mod  # noqa: E402

W8A8_CONFIG = {
    "format": "w8a8_int8", "w_bits": 8, "a_bits": 8, "weight_quant": "per_channel",
    "act_quant": "per_token", "w_sym": True, "a_sym": True, "quant_method": "smoothquant",
}
FLATQUANT_CONFIG = {
    "format": "packed_int4", "quant_method": "flatquant", "w_bits": 4, "a_bits": 4,
    "online_transforms": True,
}


def test_library_checks_clean():
    result = kernels_mod.check()
    assert result["status"] == "ok", result["problems"]
    assert {"w8a8_int8_triton", "flatquant_packed_int4_deploy"} <= set(result["entries"])


def test_module_entries_expose_their_entry_points():
    for folder, entry in kernels_mod.load_entries():
        if entry.kind != "module":
            continue
        tree = ast.parse((folder / entry.module).read_text())
        defined = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        assert set(entry.entry_points) <= defined, entry.name


def test_match_is_by_saved_format_not_method_or_model():
    other_method = {**W8A8_CONFIG, "quant_method": "some-new-w8a8-method", "base_model": "any/model"}
    names = [m["name"] for m in kernels_mod.match(other_method)["matches"]]
    assert names == ["w8a8_int8_triton"]
    assert [m["name"] for m in kernels_mod.match(FLATQUANT_CONFIG)["matches"]] == [
        "flatquant_packed_int4_deploy"
    ]


@pytest.mark.parametrize(
    "change",
    [{"a_bits": 4}, {"act_quant": "per_tensor"}, {"w_sym": False}, {"format": "packed_int4"}],
)
def test_match_rejects_a_different_format(change):
    assert kernels_mod.match({**W8A8_CONFIG, **change})["matches"] == []


def test_match_flags_retune_and_unsupported_gpus():
    tuned = kernels_mod.match(W8A8_CONFIG, compute_capability="8.6")["matches"][0]
    assert tuned["retune"] is False and tuned["supported"] is True
    other = kernels_mod.match(W8A8_CONFIG, compute_capability="9.0")["matches"][0]
    assert other["retune"] is True and other["supported"] is True
    old = kernels_mod.match(W8A8_CONFIG, compute_capability="7.5")["matches"][0]
    assert old["supported"] is False


def test_check_reports_missing_files_and_duplicates(tmp_path):
    entry = json.loads((kernels_mod.KERNELS_ROOT / "w8a8_int8" / "kernel.json").read_text())
    for folder in ("a", "b"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "kernel.json").write_text(json.dumps(entry))
    result = kernels_mod.check(tmp_path)
    assert result["status"] == "error"
    assert "duplicate name w8a8_int8_triton" in result["problems"]
    assert any("missing a/triton_w8a8.py" in p for p in result["problems"])


def test_entries_reject_incomplete_kinds():
    with pytest.raises(ValueError, match="module and entry_points"):
        kernels_mod.KernelEntry(name="x", kind="module", match={"format": ["f"]})
    with pytest.raises(ValueError, match="needs port"):
        kernels_mod.KernelEntry(name="x", kind="port", match={"format": ["f"]})


def test_w8a8_kernel_matches_reference_on_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or importlib.util.find_spec("triton") is None:
        pytest.skip("needs CUDA and triton")
    if torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("int8 tl.dot needs sm_80+")
    spec = importlib.util.spec_from_file_location(
        "triton_w8a8", kernels_mod.KERNELS_ROOT / "w8a8_int8" / "triton_w8a8.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    torch.manual_seed(0)
    for m, k, n in ((1, 1536, 256), (37, 1536, 1536), (2048, 8960, 1536)):
        x = torch.randn(m, k, dtype=torch.float16, device="cuda")
        weight = torch.randint(-127, 128, (n, k), dtype=torch.int8, device="cuda")
        scale = (torch.rand(n, 1, device="cuda") * 0.01 + 1e-3).half()
        bias = torch.randn(n, dtype=torch.float16, device="cuda")
        out = mod.w8a8_linear(x, weight, scale, bias).float()
        ref = mod.w8a8_linear_reference(x, weight, scale, bias).float()
        rel = ((out - ref).norm() / ref.norm()).item()
        assert rel < 5e-3, (m, k, n, rel)
    config = mod.tune_config(256, 1536, 1536, iters=2)
    assert config in mod.CONFIG_CANDIDATES

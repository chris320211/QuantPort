"""Offline tests for folding plain-Transformers numbers into a serve report."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "agents" / "skills" / "_shared" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import hf_runtime  # noqa: E402

GPU = "NVIDIA GeForce RTX 3090"


def _side(ppl, prefill, d1, d16, *, quantized, dtype="float16", gpu=GPU, vram=4.0):
    return {
        "runtime": "transformers", "quantized": quantized, "dtype": dtype, "gpu_name": gpu,
        "torch_version": "2.7.1+cu128", "transformers_version": "4.46.3",
        "perplexity": ppl, "tokens_evaluated": 65504, "windows": 32,
        "prefill": {"tokens_per_s": prefill, "micro_batch": 8},
        "decode": [{"batch": 1, "tokens_per_s": d1}, {"batch": 16, "tokens_per_s": d16}],
        "peak_vram_gb": vram,
    }


def _report(dtype=None):
    report = {
        "status": "passed",
        "quantized": {
            "gpu_name": GPU, "perplexity": 9.72, "prefill": {"tokens_per_s": 40000.0},
            "decode": [{"batch": 1, "tokens_per_s": 260.0}, {"batch": 16, "tokens_per_s": 3700.0}],
        },
        "comparison": {"ppl_ratio": 1.015},
    }
    if dtype:
        report["dtype"] = dtype
    return report


def test_compare_reports_both_speedups_against_transformers_baseline():
    block = hf_runtime.compare_hf_runtime(
        _report(),
        _side(10.0, 24000.0, 6.0, 90.0, quantized=True, vram=4.7),
        _side(9.6, 20000.0, 15.0, 225.0, quantized=False, vram=5.8),
    )
    cmp_ = block["comparison"]
    assert cmp_["ppl_ratio"] == pytest.approx(10.0 / 9.6)
    assert cmp_["int8_prefill_speedup"] == pytest.approx(1.2)
    assert cmp_["int8_decode_speedup"] == pytest.approx({"1": 0.4, "16": 0.4})
    assert cmp_["vllm_prefill_speedup"] == pytest.approx(2.0)
    assert cmp_["vllm_decode_speedup"] == pytest.approx({"1": 260 / 15, "16": 3700 / 225})
    assert block["dtype"] == "float16" and block["transformers_version"] == "4.46.3"
    assert block["quantized"]["peak_vram_gb"] == 4.7
    assert "runtime" not in block["baseline"]  # sides keep only the measured numbers


def test_compare_refuses_mismatched_inputs():
    q = _side(10.0, 24000.0, 6.0, 90.0, quantized=True)
    b = _side(9.6, 20000.0, 15.0, 225.0, quantized=False)
    with pytest.raises(ValueError, match="one GPU"):
        hf_runtime.compare_hf_runtime(_report(), q, {**b, "gpu_name": "NVIDIA A10G"})
    with pytest.raises(ValueError, match="dtype"):
        hf_runtime.compare_hf_runtime(_report("bfloat16"), q, b)
    with pytest.raises(ValueError, match="passed"):
        hf_runtime.compare_hf_runtime({**_report(), "status": "failed"}, q, b)
    with pytest.raises(ValueError, match="native artifact"):
        hf_runtime.compare_hf_runtime(_report(), b, q)
    bf16 = hf_runtime.compare_hf_runtime(_report("bfloat16"), q, {**b, "dtype": "bfloat16"})
    assert bf16["dtype"] == "bfloat16"


def test_attach_writes_hf_runtime_into_the_report(tmp_path, monkeypatch, capsys):
    report_path = tmp_path / "vllm.json"
    report_path.write_text(json.dumps(_report()))
    q, b = tmp_path / "q.json", tmp_path / "b.json"
    q.write_text(json.dumps(_side(10.0, 24000.0, 6.0, 90.0, quantized=True)))
    b.write_text(json.dumps(_side(9.6, 20000.0, 15.0, 225.0, quantized=False)))
    monkeypatch.setattr(sys, "argv", [
        "hf_runtime.py", "attach", "--report", str(report_path), "--quantized", str(q), "--baseline", str(b),
    ])
    assert hf_runtime.main() == 0
    report = json.loads(report_path.read_text())
    assert report["status"] == "passed"
    assert report["hf_runtime"]["comparison"]["int8_prefill_speedup"] == pytest.approx(1.2)
    b.write_text(json.dumps(_side(9.6, 20000.0, 15.0, 225.0, quantized=False, gpu="NVIDIA L4")))
    assert hf_runtime.main() == 1
    assert "one GPU" in capsys.readouterr().out

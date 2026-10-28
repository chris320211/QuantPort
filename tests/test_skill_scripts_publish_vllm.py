"""Offline tests for staging a vLLM export as its own Hub repo."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "agents" / "skills" / "_shared" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import paths as paths_mod  # noqa: E402
import publish_vllm as pv  # noqa: E402

LICENSE = {"license": "apache-2.0"}


def _report(status="passed"):
    side = lambda ppl, pre, d1, d16: {  # noqa: E731
        "model_path": "/root/secret-home/model",
        "vllm_version": "0.30.0",
        "gpu_name": "NVIDIA GeForce RTX 3090",
        "perplexity": ppl,
        "tokens_evaluated": 65504,
        "prefill": {"tokens_per_s": pre},
        "decode": [{"batch": 1, "tokens_per_s": d1}, {"batch": 16, "tokens_per_s": d16}],
    }
    return {
        "status": status,
        "issue_codes": [] if status == "passed" else ["vllm_quality_failed"],
        "model_dir": "/root/secret-home/export",
        "baseline_dir": "/root/secret-home/base",
        "quantized": side(10.4, 26000.0, 120.0, 1500.0),
        "fp16_baseline": side(10.0, 20000.0, 100.0, 1400.0),
        "comparison": {
            "ppl_ratio": 1.04, "hf_ppl_ratio": 1.035, "ppl_ratio_drift": 1.0048,
            "prefill_speedup": 1.3, "decode_speedup": {"1": 1.2, "16": 1500 / 1400},
        },
    }


def _export(tmp_path, *, status="exported") -> Path:
    root = tmp_path / "quantized" / "sq-model-vllm"
    root.mkdir(parents=True)
    (root / "config.json").write_text("{}")
    (root / "model.safetensors").write_bytes(b"\0" * 16)
    (root / "tokenizer.json").write_text("{}")
    (root / "vllm_export.json").write_text(json.dumps({
        "status": status, "profile": "w8a8_int8", "ignore": ["lm_head"],
        "source_dir": "/Users/someone/repo/quantized/sq-model", "output_dir": str(root),
    }))
    return root


@pytest.fixture
def out_root(tmp_path, monkeypatch):
    monkeypatch.setattr(paths_mod, "OUT_ROOT", tmp_path / "out")
    return tmp_path / "out"


def _stage(tmp_path, **kw):
    if "export_dir" not in kw:
        kw["export_dir"] = _export(tmp_path)
    args = dict(
        serve_report=_report(), model_id="org/model",
        method_name="SmoothQuant", base_repo_id="me/sq-model", license_info=LICENSE,
    )
    args.update(kw)
    return pv.stage_vllm_bundle(**args)


def test_stage_writes_own_repo_with_card_and_scrubbed_records(tmp_path, out_root):
    manifest = _stage(tmp_path)
    stage = Path(manifest["stage_dir"])
    assert manifest["repo_id"] == "me/sq-model-vllm"
    assert stage == (out_root / "hub" / "sq-model-vllm").resolve()
    assert {"config.json", "model.safetensors", "tokenizer.json", "README.md", "vllm.json", "vllm_export.json"} <= {
        p.name for p in stage.iterdir()
    }
    for name in ("vllm.json", "vllm_export.json"):
        text = (stage / name).read_text()
        assert "/root/secret-home" not in text and "/Users/someone" not in text
    card = (stage / "README.md").read_text()
    assert "base_model: org/model" in card and "license: apache-2.0" in card
    assert "vllm serve me/sq-model-vllm" in card
    assert "NVIDIA GeForce RTX 3090" in card and "vLLM 0.30.0" in card
    assert "| Perplexity | 10.4000 | 10.0000 |" in card
    assert "| Decode tok/s, batch 16 | 1500.0 | 1400.0 |" in card
    assert "1.0048" in card


def test_only_a_passed_check_publishes(tmp_path, out_root):
    with pytest.raises(RuntimeError, match="only a passed check"):
        _stage(tmp_path, serve_report=_report("failed"))
    assert not (out_root / "hub").exists()


def test_unexported_record_is_refused(tmp_path, out_root):
    with pytest.raises(RuntimeError, match="not exported"):
        _stage(tmp_path, export_dir=_export(tmp_path, status="unsupported"))


def test_large_weights_are_uploaded_from_the_export(tmp_path, out_root, monkeypatch):
    monkeypatch.setattr(pv, "_LARGE_FILE_BYTES", 8)
    manifest = _stage(tmp_path)
    assert [Path(p).name for p in manifest["weight_srcs"]] == ["model.safetensors"]
    assert not (Path(manifest["stage_dir"]) / "model.safetensors").exists()


def test_card_section_inserts_before_license_and_replaces_itself():
    card = "---\nlicense: mit\n---\n\n# X\n\n## Pull\n\nstuff\n\n## License\n\nmit\n"
    section = pv.render_card_section(repo_id="me/x-vllm", report=_report())
    once = pv.merge_card_section(card, section)
    assert once.index("## Serve with vLLM") < once.index("## License")
    assert "vllm serve me/x-vllm" in once
    updated = pv.render_card_section(repo_id="me/x-vllm2", report=_report())
    twice = pv.merge_card_section(once, updated)
    assert twice.count(pv.CARD_SECTION_BEGIN) == 1
    assert "me/x-vllm2" in twice and "vllm serve me/x-vllm\n" not in twice
    assert twice.endswith("## License\n\nmit\n")


def test_card_section_appends_without_license_heading():
    merged = pv.merge_card_section("# X\n", pv.render_card_section(repo_id="me/x-vllm", report=_report()))
    assert merged.startswith("# X\n\n" + pv.CARD_SECTION_BEGIN)


def test_card_explains_a_vllm_build_that_beats_the_hf_ratio(tmp_path, out_root):
    report = _report()
    report["comparison"]["ppl_ratio_drift"] = 0.9656
    stage = Path(_stage(tmp_path, serve_report=report)["stage_dir"])
    card = (stage / "README.md").read_text()
    assert "vLLM scores better than the HF benchmark" in card
    assert "vLLM scores better" not in pv.render_vllm_card(
        model_id="m", method_name="M", base_repo_id="a/b", repo_id="a/b-vllm",
        report=_report(), export_record={}, license_info=LICENSE,
    )


def test_card_names_the_baseline_dtype():
    report = {**_report(), "dtype": "bfloat16"}
    card = pv.render_vllm_card(
        model_id="m", method_name="M", base_repo_id="a/b", repo_id="a/b-vllm",
        report=report, export_record={"ignore": ["lm_head"]}, license_info=LICENSE,
    )
    assert "against vLLM bf16 of `m`" in card
    assert "| | vLLM int8 | vLLM bf16 |" in card
    assert "fp16" not in card
    assert "against vLLM bf16;" in pv.render_card_section(repo_id="a/b-vllm", report=report)
    assert "vLLM int8 | vLLM fp16 |" in pv.render_vllm_card(
        model_id="m", method_name="M", base_repo_id="a/b", repo_id="a/b-vllm",
        report=_report(), export_record={}, license_info=LICENSE,
    )


def _with_hf_runtime(report):
    side = lambda ppl, pre, d1, d16, vram: {  # noqa: E731
        "dtype": "float16", "perplexity": ppl, "prefill": {"tokens_per_s": pre, "micro_batch": 8},
        "decode": [{"batch": 1, "tokens_per_s": d1}, {"batch": 16, "tokens_per_s": d16}], "peak_vram_gb": vram,
    }
    return {**report, "hf_runtime": {
        "gpu_name": "NVIDIA GeForce RTX 3090", "dtype": "float16",
        "torch_version": "2.7.1+cu128", "transformers_version": "4.46.3",
        "quantized": side(10.5, 21000.0, 6.0, 90.0, 4.68),
        "baseline": side(10.0, 18000.0, 14.0, 220.0, 5.84),
        "comparison": {
            "ppl_ratio": 1.05, "int8_prefill_speedup": 21000 / 18000,
            "int8_decode_speedup": {"1": 6 / 14, "16": 90 / 220},
            "vllm_prefill_speedup": 26000 / 18000, "vllm_decode_speedup": {"1": 120 / 14, "16": 1500 / 220},
        },
    }}


def test_card_adds_transformers_columns_from_the_same_gpu():
    card = pv.render_vllm_card(
        model_id="org/model", method_name="SmoothQuant", base_repo_id="me/sq-model", repo_id="me/sq-model-vllm",
        report=_with_hf_runtime(_report()), export_record={}, license_info=LICENSE,
    )
    assert "| | vLLM int8 | vLLM fp16 | Transformers int8 | Transformers fp16 |" in card
    assert "| Decode tok/s, batch 1 | 120.0 | 100.0 | 6.0 | 14.0 |" in card
    assert "| Peak VRAM GB | – | – | 4.68 | 5.84 |" in card
    assert "Transformers 4.46.3 (torch 2.7.1+cu128) on the same GPU" in card
    assert "vLLM int8 prefill 1.44×" in card
    assert "compare speed within this table only" not in card
    plain = pv.render_vllm_card(
        model_id="org/model", method_name="SmoothQuant", base_repo_id="me/sq-model", repo_id="me/sq-model-vllm",
        report=_report(), export_record={}, license_info=LICENSE,
    )
    assert "Transformers" not in plain and "compare speed within this table only" in plain


def test_base_card_section_carries_vllm_results_and_serving_advice():
    section = pv.render_card_section(
        repo_id="me/sq-model-vllm", report=_with_hf_runtime(_report()), base_repo_id="me/sq-model", model_id="org/model",
    )
    assert section.startswith(pv.CARD_SECTION_BEGIN) and section.rstrip().endswith(pv.CARD_SECTION_END)
    assert "## vLLM results" in section
    assert "| | vLLM int8 | vLLM fp16 | Transformers int8 | Transformers fp16 |" in section
    assert "For serving, use [`me/sq-model-vllm`]" in section
    assert "## vLLM results" not in pv.render_card_section(repo_id="me/sq-model-vllm", report=_report())
    card = "# X\n\n## License\n\nMIT\n"
    once = pv.merge_card_section(card, section)
    assert pv.merge_card_section(once, section) == once

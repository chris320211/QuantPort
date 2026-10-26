"""Offline tests for Hub staging of quantized artifacts + WikiText-2 metrics."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "agents" / "skills" / "_shared" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import publish as publish_mod  # noqa: E402


def test_render_model_card_uses_benchmark_numbers():
    card = publish_mod.render_model_card(
        request={
            "method_name": "FlatQuant",
            "model_id": "microsoft/Phi-3-mini-4k-instruct",
            "gpu_instance": "g5.2xlarge",
            "gpu_name": "NVIDIA A10G",
            "slug": "flatquant-phi3-mini-4k-g52xlarge",
        },
        job_id="20260919T202736Z-5a1860",
        benchmark={
            "quantized": {
                "perplexity": 7.393646309110877,
                "tokens_per_s": 5105.929253072313,
                "peak_vram_gb": 3.3546090126037598,
            },
            "fp16_baseline": {
                "perplexity": 6.332275669117074,
                "tokens_per_s": 3387.5933448585142,
                "peak_vram_gb": 9.680975437164307,
            },
            "comparison": {
                "ppl_ratio": 1.1676128291713794,
                "quality_ok": True,
                "improved_throughput": True,
                "improved_vram": True,
            },
        },
        repo_id="example/flatquant-phi3-mini-4k-g52xlarge",
    )
    assert "7.393646" in card
    assert "5105.9" in card
    assert "quality_ok=True" in card
    assert "20260919T202736Z-5a1860" in card
    assert "snapshot_download" in card
    assert "microsoft/Phi-3-mini-4k-instruct" in card
    assert "base_model_relation: quantized" in card
    assert "license: other" in card


def test_render_model_card_inherits_base_license():
    card = publish_mod.render_model_card(
        request={"method_name": "SmoothQuant", "model_id": "google/gemma-2-2b-it"},
        job_id="job",
        benchmark={},
        repo_id=None,
        license_info={"license": "gemma"},
    )
    assert "license: gemma\n" in card
    assert "license: mit" not in card


def _fake_completed_job(tmp_path, monkeypatch, output_files):
    from types import SimpleNamespace

    output_dir = tmp_path / "quantized"
    output_dir.mkdir()
    for name in output_files:
        (output_dir / name).write_text("x")
    job_directory = tmp_path / "job"
    job_directory.mkdir()
    (job_directory / "benchmark.json").write_text(
        json.dumps({"comparison": {"quality_ok": True, "improved_vram": True}})
    )
    meta = SimpleNamespace(
        status="completed",
        verification_status="passed",
        output_dir=str(output_dir),
        overlay_path=None,
    )
    monkeypatch.setattr(publish_mod.jobs_mod, "refresh_status", lambda job_id: meta)
    monkeypatch.setattr(publish_mod.jobs_mod, "job_dir", lambda job_id: job_directory)
    monkeypatch.setattr(publish_mod, "resolve_workspace_path", lambda p: Path(p))
    monkeypatch.setattr(publish_mod.paths, "OUT_ROOT", tmp_path / "out")
    return {"method_name": "SmoothQuant", "model_id": "org/base", "slug": "sq-test", "license": "mit"}


def test_stage_refuses_bundle_without_weights(tmp_path, monkeypatch):
    request = _fake_completed_job(tmp_path, monkeypatch, ["config.json", "tokenizer.json"])
    try:
        publish_mod.stage_hub_bundle(job_id="job", request=request)
    except RuntimeError as exc:
        assert "no weight file" in str(exc)
    else:
        raise AssertionError("expected staging to refuse a bundle without weights")
    assert not (tmp_path / "out" / "hub" / "sq-test").exists()


def test_stage_accepts_bundle_with_weights(tmp_path, monkeypatch):
    request = _fake_completed_job(tmp_path, monkeypatch, ["config.json", "model.safetensors"])
    manifest = publish_mod.stage_hub_bundle(job_id="job", request=request)
    assert "model.safetensors" in manifest["copied"]


def test_upload_refuses_bundle_without_weights(tmp_path):
    from env import host_execution_policy

    (tmp_path / "config.json").write_text("{}")
    with host_execution_policy(True):
        try:
            publish_mod.upload_hub_bundle({"stage_dir": str(tmp_path), "weight_srcs": []}, "org/name")
        except RuntimeError as exc:
            assert "no weight file" in str(exc)
        else:
            raise AssertionError("expected upload to refuse a bundle without weights")


def test_upload_requires_org_name_repo_id():
    try:
        publish_mod.upload_hub_bundle({"stage_dir": "/tmp"}, "not-a-repo")
    except Exception as exc:
        assert "org/name" in str(exc) or "host" in str(exc).lower()
    else:
        raise AssertionError("expected upload_hub_bundle to reject a bad repo id or missing host flag")


def test_stage_suffixes_include_safetensors_and_tokenizer_merges():
    assert ".safetensors" in publish_mod._STAGE_SUFFIXES
    assert ".bin" in publish_mod._STAGE_SUFFIXES
    assert "merges.txt" in publish_mod._COPY_NAMES


def test_publish_requires_beneficial_comparison():
    assert publish_mod.library_mod.is_beneficial(
        {"quality_ok": True, "improved_vram": True, "improved_throughput": False}
    )
    assert not publish_mod.library_mod.is_beneficial(
        {"quality_ok": True, "improved_vram": False, "improved_throughput": False}
    )
    assert not publish_mod.library_mod.is_beneficial(
        {"quality_ok": False, "improved_vram": True, "improved_throughput": True}
    )

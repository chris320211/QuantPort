"""Offline tests for method × model × GPU WikiText-2 ranking."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "agents" / "skills" / "_shared" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import library as library_mod  # noqa: E402
import jobs as jobs_mod  # noqa: E402
import paths as paths_mod  # noqa: E402


def test_faster_quality_ok_method_wins_over_broken_fast_method():
    rows = [
        {
            "method_name": "BrokenFast",
            "quality_ok": False,
            "tokens_per_s": 9000.0,
            "peak_vram_gb": 2.0,
            "ppl_ratio": 8.0,
            "packed": True,
        },
        {
            "method_name": "Solid",
            "quality_ok": True,
            "tokens_per_s": 1000.0,
            "peak_vram_gb": 4.0,
            "ppl_ratio": 1.2,
            "packed": False,
        },
    ]
    ranked = library_mod.rank_methods(rows)
    assert ranked[0]["method_name"] == "Solid"
    assert ranked[0]["rank"] == 1
    best = library_mod.pick_best(rows)
    assert best is not None
    assert best["method_name"] == "Solid"


def test_same_board_ranks_two_quality_ok_methods_by_tok_s():
    rows = [
        {
            "method_name": "AWQ",
            "quality_ok": True,
            "tokens_per_s": 4000.0,
            "peak_vram_gb": 5.0,
            "ppl_ratio": 1.10,
            "packed": True,
        },
        {
            "method_name": "FlatQuant",
            "quality_ok": True,
            "tokens_per_s": 5105.9,
            "peak_vram_gb": 3.35,
            "ppl_ratio": 1.17,
            "packed": True,
        },
    ]
    best = library_mod.pick_best(rows)
    assert best is not None
    assert best["method_name"] == "FlatQuant"
    ranked = library_mod.rank_methods(rows)
    assert [row["method_name"] for row in ranked] == ["FlatQuant", "AWQ"]


def test_group_filename_is_model_and_gpu_not_method():
    name = library_mod.group_filename(
        "microsoft/Phi-3-mini-4k-instruct", "g5.2xlarge"
    )
    assert "Phi-3-mini-4k-instruct" in name
    assert "g5.2xlarge" in name
    assert "flatquant" not in name.lower()
    assert "awq" not in name.lower()
    assert library_mod.group_key("org/model", "g5.xlarge") == "org/model|g5.xlarge"


def _write_job(tmp_path: Path, job_id: str, *, packed: bool, tps: float, ratio: float) -> Path:
    output = tmp_path / "quantized" / job_id
    output.mkdir(parents=True)
    if packed:
        (output / "packed_w4a4.pt").write_bytes(b"x")
        (output / "quantization_config.json").write_text(
            json.dumps({"format": "packed_int4"}) + "\n"
        )
    else:
        (output / "pytorch_model.bin").write_bytes(b"x")
        (output / "quantization_config.json").write_text(
            json.dumps({"format": "fakequant_fp16"}) + "\n"
        )
    meta = jobs_mod.JobMeta(
        job_id=job_id,
        method_id="m",
        model_id="microsoft/Phi-3-mini-4k-instruct",
        venv="venv",
        script_path="script.py",
        output_dir=str(output),
        pid=1,
        started_at="2026-01-01T00:00:00Z",
        status="completed",
        verification_status="passed",
        benchmark_status="passed",
    )
    jobs_mod.write_meta(meta)
    job_dir = jobs_mod.job_dir(job_id)
    ppl = 6.0 * ratio
    payload = {
        "quantized": {
            "perplexity": ppl,
            "tokens_per_s": tps,
            "peak_vram_gb": 3.35 if packed else 8.0,
        },
        "fp16_baseline": {
            "perplexity": 6.0,
            "tokens_per_s": 3300.0,
            "peak_vram_gb": 9.68,
        },
        "comparison": {
            "ppl_ratio": ratio,
            "quality_ok": ratio <= 1.5,
            "improved_throughput": tps > 3300.0,
            "improved_vram": True,
        },
    }
    (job_dir / "benchmark.json").write_text(json.dumps(payload) + "\n")
    return output


def test_record_job_replaces_same_method_and_ranks_two_methods(tmp_path, monkeypatch):
    monkeypatch.setattr(paths_mod, "JOBS_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(paths_mod, "LIBRARY_ROOT", tmp_path / "library")
    monkeypatch.setattr(paths_mod, "REPO_ROOT", tmp_path)
    job_a = "20260101T000000Z-aaaaaa"
    job_b = "20260101T000000Z-bbbbbb"
    job_c = "20260101T000000Z-cccccc"
    _write_job(tmp_path, job_a, packed=True, tps=3140.0, ratio=1.17)
    first = library_mod.record_job(
        job_id=job_a,
        request={
            "method_name": "FlatQuant",
            "model_id": "microsoft/Phi-3-mini-4k-instruct",
            "gpu_instance": "g5.2xlarge",
            "gpu_name": "NVIDIA A10G",
            "slug": "flatquant-phi3-mini-4k-g52xlarge",
            "arxiv_id": "2410.09426",
            "repo_url": "https://github.com/ruikangliu/FlatQuant",
        },
        hub_url="https://huggingface.co/example/flatquant-phi3",
    )
    assert first["best"]["method_name"] == "FlatQuant"
    assert first["methods"][0]["packed"] is True
    assert first["methods"][0]["hub_url"].endswith("flatquant-phi3")
    assert first["methods"][0]["paper_url"] == "https://arxiv.org/abs/2410.09426"
    assert first["methods"][0]["repo_url"] == "https://github.com/ruikangliu/FlatQuant"

    _write_job(tmp_path, job_b, packed=True, tps=5105.9, ratio=1.17)
    second = library_mod.record_job(
        job_id=job_b,
        request={
            "method_name": "FlatQuant",
            "model_id": "microsoft/Phi-3-mini-4k-instruct",
            "gpu_instance": "g5.2xlarge",
            "slug": "flatquant-phi3-mini-4k-g52xlarge",
            "arxiv_id": "2410.09426",
            "repo_url": "https://github.com/ruikangliu/FlatQuant",
        },
    )
    methods = {row["method_name"]: row for row in second["methods"]}
    assert len(methods) == 1
    assert methods["FlatQuant"]["job_id"] == job_b
    assert methods["FlatQuant"]["tokens_per_s"] == 5105.9
    assert methods["FlatQuant"]["hub_url"].endswith("flatquant-phi3")

    _write_job(tmp_path, job_c, packed=True, tps=4000.0, ratio=1.10)
    third = library_mod.record_job(
        job_id=job_c,
        request={
            "method_name": "AWQ",
            "model_id": "microsoft/Phi-3-mini-4k-instruct",
            "gpu_instance": "g5.2xlarge",
            "slug": "awq-phi3-mini-4k-g52xlarge",
        },
    )
    names = [row["method_name"] for row in third["methods"]]
    assert names[0] == "FlatQuant"
    assert "AWQ" in names
    assert third["best"]["method_name"] == "FlatQuant"
    index = json.loads((tmp_path / "library" / "index.json").read_text())
    assert index["groups"][0]["n_methods"] == 2
    assert index["groups"][0]["best_method"] == "FlatQuant"
    readme = (tmp_path / "library" / "LIBRARY.md").read_text()
    assert "microsoft/Phi-3-mini-4k-instruct" in readme
    assert "g5.2xlarge" in readme
    assert "**Best:** FlatQuant" in readme
    catalog = json.loads((tmp_path / "library" / "catalog.json").read_text())
    assert len(catalog["rows"]) == 2
    by_method = {row["method_name"]: row for row in catalog["rows"]}
    assert by_method["FlatQuant"]["is_best"] is True
    assert by_method["AWQ"]["is_best"] is False
    assert by_method["FlatQuant"]["hub_repo_id"] == "example/flatquant-phi3"
    matched = library_mod.query(
        model_id="microsoft/Phi-3-mini-4k-instruct",
        gpu_instance="g5.2xlarge",
        method_name="awq",
    )
    assert len(matched) == 1
    assert matched[0]["method_name"] == "AWQ"
    best = library_mod.query(gpu_instance="g5.2xlarge", best_only=True)
    assert [row["method_name"] for row in best] == ["FlatQuant"]
    howto = library_mod.fetch_howto(by_method["FlatQuant"])
    assert howto["status"] == "ok"
    assert "huggingface-cli download example/flatquant-phi3" in howto["command"]
    local = library_mod.fetch_howto({"artifact_dir": "quantized/x"})
    assert local["status"] == "local_only"


def _write_diagnose(job_id: str, payload: dict) -> None:
    (jobs_mod.job_dir(job_id) / "diagnose.json").write_text(json.dumps(payload) + "\n")


def test_record_attempt_keeps_failures_off_the_public_board(tmp_path, monkeypatch):
    monkeypatch.setattr(paths_mod, "JOBS_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(paths_mod, "LIBRARY_ROOT", tmp_path / "library")
    monkeypatch.setattr(paths_mod, "REPO_ROOT", tmp_path)
    job_id = "20260101T000000Z-aaaaaa"
    _write_job(tmp_path, job_id, packed=False, tps=100.0, ratio=8.76)
    _write_diagnose(
        job_id,
        {
            "issue_codes": ["ppl_exploded", "throughput_unchanged"],
            "root_cause": "ppl_ratio_exploded",
            "recommended_action": "none",
            "notes": ["Retry GPU budget exhausted; stop."],
            "error_excerpt": "reusing GPTQ weights spinquant.pt",
            "retry": {"used": 14, "max": 14, "remaining": 0},
            "best_job_id": job_id,
            "best_ppl_ratio": 8.76,
        },
    )
    request = {
        "method_name": "SpinQuant",
        "model_id": "google/gemma-2-2b-it",
        "gpu_instance": "g5.2xlarge",
        "gpu_name": "NVIDIA A10G",
        "slug": "spinquant-gemma2-2b-g52xlarge",
        "arxiv_id": "2405.16406",
        "repo_url": "https://github.com/facebookresearch/SpinQuant",
        "retry_gpu_jobs_used": 14,
        "retry_gpu_jobs_max": 14,
        "best_job_id": job_id,
        "best_ppl_ratio": 8.76,
    }
    result = library_mod.record_attempt(job_id=job_id, request=request)
    assert result["status"] == "recorded"
    assert result["stop_reason"] == "budget_exhausted"
    assert result["method_name"] == "SpinQuant"
    assert library_mod.query() == []
    assert library_mod.catalog_rows() == []
    attempts = library_mod.query_attempts()
    assert len(attempts) == 1
    row = attempts[0]
    assert row["model_id"] == "google/gemma-2-2b-it"
    assert row["issue_codes"] == ["ppl_exploded", "throughput_unchanged"]
    assert row["retry_gpu_jobs_used"] == 14
    assert abs(float(row["ppl_ratio"]) - 8.76) < 1e-6
    md = (tmp_path / "library" / "ATTEMPTS.md").read_text()
    assert "SpinQuant" in md
    assert "budget_exhausted" in md
    library_md = (tmp_path / "library" / "LIBRARY.md").read_text()
    assert "ATTEMPTS.md" in library_md
    assert "SpinQuant" not in library_md
    assert "google/gemma-2-2b-it" not in library_md
    catalog = json.loads((tmp_path / "library" / "catalog.json").read_text())
    assert catalog["rows"] == []


def test_record_attempt_without_benchmark_records_process_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(paths_mod, "JOBS_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(paths_mod, "LIBRARY_ROOT", tmp_path / "library")
    monkeypatch.setattr(paths_mod, "REPO_ROOT", tmp_path)
    job_id = "20260101T000000Z-bbbbbb"
    output = tmp_path / "quantized" / job_id
    output.mkdir(parents=True)
    meta = jobs_mod.JobMeta(
        job_id=job_id,
        method_id="m",
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        venv="venv",
        script_path="script.py",
        output_dir=str(output),
        pid=1,
        started_at="2026-01-01T00:00:00Z",
        status="failed",
        exit_code=1,
        verification_status=None,
        benchmark_status=None,
    )
    jobs_mod.write_meta(meta)
    _write_diagnose(
        job_id,
        {
            "issue_codes": ["process_failed"],
            "root_cause": "process_failed",
            "recommended_action": "none",
            "notes": ["GPU quantize process crashed."],
            "retry": {"used": 1, "max": 14, "remaining": 13},
        },
    )
    result = library_mod.record_attempt(
        job_id=job_id,
        request={
            "method_name": "QuaRot",
            "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
            "gpu_instance": "g5.2xlarge",
            "slug": "quarot-qwen25-15b-g52xlarge",
            "arxiv_id": "2404.00456",
            "repo_url": "https://github.com/spcl/QuaRot",
            "retry_gpu_jobs_used": 1,
            "retry_gpu_jobs_max": 14,
        },
    )
    assert result["stop_reason"] == "process_failed"
    row = library_mod.query_attempts()[0]
    assert row["ppl"] is None
    assert row["issue_codes"] == ["process_failed"]
    assert library_mod.query() == []


def test_record_attempt_rejects_beneficial_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(paths_mod, "JOBS_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(paths_mod, "LIBRARY_ROOT", tmp_path / "library")
    monkeypatch.setattr(paths_mod, "REPO_ROOT", tmp_path)
    job_id = "20260101T000000Z-cccccc"
    _write_job(tmp_path, job_id, packed=True, tps=5105.9, ratio=1.17)
    request = {
        "method_name": "FlatQuant",
        "model_id": "microsoft/Phi-3-mini-4k-instruct",
        "gpu_instance": "g5.2xlarge",
        "slug": "flatquant-phi3-mini-4k-g52xlarge",
        "arxiv_id": "2410.09426",
        "repo_url": "https://github.com/ruikangliu/FlatQuant",
    }
    try:
        library_mod.record_attempt(job_id=job_id, request=request)
    except RuntimeError as exc:
        assert "beneficial" in str(exc)
    else:
        raise AssertionError("expected record_attempt to reject a beneficial run")
    assert library_mod.query_attempts() == []


def test_successful_catalog_clears_matching_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(paths_mod, "JOBS_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(paths_mod, "LIBRARY_ROOT", tmp_path / "library")
    monkeypatch.setattr(paths_mod, "REPO_ROOT", tmp_path)
    failed_id = "20260101T000000Z-dddddd"
    ok_id = "20260101T000000Z-eeeeee"
    _write_job(tmp_path, failed_id, packed=False, tps=100.0, ratio=8.0)
    _write_diagnose(
        failed_id,
        {
            "issue_codes": ["ppl_exploded"],
            "recommended_action": "none",
            "notes": ["Retry GPU budget exhausted; stop."],
            "retry": {"used": 14, "max": 14, "remaining": 0},
        },
    )
    request = {
        "method_name": "FlatQuant",
        "model_id": "microsoft/Phi-3-mini-4k-instruct",
        "gpu_instance": "g5.2xlarge",
        "slug": "flatquant-phi3-mini-4k-g52xlarge",
        "arxiv_id": "2410.09426",
        "repo_url": "https://github.com/ruikangliu/FlatQuant",
    }
    library_mod.record_attempt(job_id=failed_id, request=request)
    assert library_mod.query_attempts()
    _write_job(tmp_path, ok_id, packed=True, tps=5105.9, ratio=1.17)
    library_mod.record_job(
        job_id=ok_id,
        request=request,
        hub_url="https://huggingface.co/example/flatquant-phi3",
    )
    assert library_mod.query_attempts() == []
    assert library_mod.query(method_name="FlatQuant")
    leftover = tmp_path / "library" / "attempts"
    assert list(leftover.glob("*.json")) == []


def test_record_job_rejects_non_beneficial_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(paths_mod, "JOBS_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(paths_mod, "LIBRARY_ROOT", tmp_path / "library")
    monkeypatch.setattr(paths_mod, "REPO_ROOT", tmp_path)
    job_id = "20260101T000000Z-ffffff"
    _write_job(tmp_path, job_id, packed=False, tps=100.0, ratio=8.0)
    request = {
        "method_name": "QuaRot",
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "gpu_instance": "g5.2xlarge",
        "slug": "quarot-qwen25-15b-g52xlarge",
        "arxiv_id": "2404.00456",
        "repo_url": "https://github.com/spcl/QuaRot",
    }
    try:
        library_mod.record_job(job_id=job_id, request=request)
    except RuntimeError as exc:
        assert "quality_ok" in str(exc) or "beat fp16" in str(exc)
    else:
        raise AssertionError("expected record_job to reject a non-beneficial run")
    groups_dir = tmp_path / "library" / "groups"
    assert not groups_dir.exists() or list(groups_dir.glob("*.json")) == []

    job_ok = "20260101T000000Z-eeeeee"
    _write_job(tmp_path, job_ok, packed=True, tps=100.0, ratio=1.1)
    bench = jobs_mod.job_dir(job_ok) / "benchmark.json"
    payload = json.loads(bench.read_text())
    payload["comparison"]["improved_vram"] = False
    payload["comparison"]["improved_throughput"] = False
    bench.write_text(json.dumps(payload) + "\n")
    try:
        library_mod.record_job(
            job_id=job_ok,
            request={**request, "method_name": "SmoothQuant", "slug": "sq"},
        )
    except RuntimeError as exc:
        assert "beat fp16" in str(exc)
    else:
        raise AssertionError("expected record_job to reject quality_ok without efficiency")


def test_contribution_json_is_the_pr_unit(tmp_path, monkeypatch):
    monkeypatch.setattr(paths_mod, "JOBS_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(paths_mod, "LIBRARY_ROOT", tmp_path / "library")
    monkeypatch.setattr(paths_mod, "REPO_ROOT", tmp_path)
    job_id = "20260101T000000Z-dddddd"
    _write_job(tmp_path, job_id, packed=True, tps=5105.9, ratio=1.17)
    library_mod.record_job(
        job_id=job_id,
        request={
            "method_name": "FlatQuant",
            "model_id": "microsoft/Phi-3-mini-4k-instruct",
            "gpu_instance": "g5.2xlarge",
            "slug": "flatquant-phi3-mini-4k-g52xlarge",
        },
    )
    contrib = {
        "schema_version": 1,
        "model_id": "microsoft/Phi-3-mini-4k-instruct",
        "method_name": "AWQ",
        "gpu_instance": "g5.2xlarge",
        "gpu_name": "NVIDIA A10G",
        "quality_ok": True,
        "ppl": 6.6,
        "ppl_ratio": 1.1,
        "tokens_per_s": 4000.0,
        "peak_vram_gb": 4.2,
        "fp16_ppl": 6.0,
        "fp16_tokens_per_s": 3300.0,
        "fp16_peak_vram_gb": 9.68,
        "packed": True,
        "hub_repo_id": "someone/awq-phi3",
        "arxiv_id": "2306.00978",
        "repo_url": "https://github.com/mit-han-lab/llm-awq",
        "dataset": "wikitext-2-raw-v1",
        "max_seq_len": 2048,
    }
    src = tmp_path / "incoming.json"
    src.write_text(json.dumps(contrib) + "\n")
    dest = library_mod.accept_contribution(src)
    assert dest.name.endswith("AWQ.json")
    rows = {row["method_name"]: row for row in library_mod.query()}
    assert set(rows) == {"FlatQuant", "AWQ"}
    assert rows["AWQ"]["hub_repo_id"] == "someone/awq-phi3"
    assert rows["FlatQuant"]["is_best"] is True
    howto = library_mod.fetch_howto(rows["AWQ"])
    assert "huggingface-cli download someone/awq-phi3" in howto["command"]


def test_validate_contribution_requires_hub_and_quality():
    base = {
        "model_id": "org/model",
        "method_name": "AWQ",
        "gpu_instance": "g5.xlarge",
        "quality_ok": True,
        "ppl": 11.0,
        "ppl_ratio": 1.1,
        "tokens_per_s": 100.0,
        "peak_vram_gb": 4.0,
        "fp16_ppl": 10.0,
        "fp16_tokens_per_s": 80.0,
        "fp16_peak_vram_gb": 8.0,
        "hub_repo_id": "org/awq-model",
        "arxiv_id": "2306.00978",
        "repo_url": "https://github.com/mit-han-lab/llm-awq",
    }
    library_mod.validate_contribution(base)
    bad_quality = dict(base, quality_ok=False)
    try:
        library_mod.validate_contribution(bad_quality)
    except ValueError as exc:
        assert "quality_ok" in str(exc)
    else:
        raise AssertionError("expected quality_ok rejection")
    missing_hub = dict(base, hub_repo_id=None)
    try:
        library_mod.validate_contribution(missing_hub)
    except ValueError as exc:
        assert "hub_repo_id" in str(exc)
    else:
        raise AssertionError("expected hub_repo_id rejection")
    missing_paper = dict(base, arxiv_id=None, paper_url=None)
    try:
        library_mod.validate_contribution(missing_paper)
    except ValueError as exc:
        assert "paper" in str(exc)
    else:
        raise AssertionError("expected paper rejection")
    fat = dict(base, tokens_per_s=50.0, peak_vram_gb=9.0)
    try:
        library_mod.validate_contribution(fat)
    except ValueError as exc:
        assert "beat fp16" in str(exc)
    else:
        raise AssertionError("expected efficiency rejection")


def test_pick_best_none_when_nothing_quality_ok():
    assert library_mod.pick_best([{"quality_ok": False, "tokens_per_s": 9}]) is None


def test_quant_skills_point_at_library_index():
    parent = (ROOT / "agents" / "skills" / "quant" / "SKILL.md").read_text()
    bench = (ROOT / "agents" / "skills" / "quant-benchmark" / "SKILL.md").read_text()
    publish = (ROOT / "agents" / "skills" / "quant-publish" / "SKILL.md").read_text()
    assert "library/" in parent
    assert "catalog.json" in parent
    assert "library/" in bench
    assert "contributions" in parent
    assert "--record-attempt" in parent or "record-attempt" in parent
    catalog = (ROOT / "agents" / "skills" / "quant-catalog" / "SKILL.md").read_text()
    assert "contributions" in catalog
    assert "--catalog" in catalog
    assert "--record-attempt" in catalog
    assert (ROOT / "library" / "LIBRARY.md").is_file()
    assert (ROOT / "library" / "ATTEMPTS.md").is_file()
    assert not (ROOT / "library" / "README.md").is_file()
    assert "library/" in publish
    assert "artifact store" in publish.lower() or "not the comparison" in publish.lower()
    sync = (ROOT / "agents" / "skills" / "quant-sync" / "SKILL.md").read_text()
    assert "sync_remotes.py" in sync
    assert "--push" in sync
    assert "quant-sync" in parent


def test_collection_entries_dedupes_and_skips_failed_quality():
    rows = [
        {
            "hub_repo_id": "a/one",
            "quality_ok": True,
            "method_name": "AWQ",
            "model_id": "org/m",
            "gpu_instance": "g5.2xlarge",
            "ppl_ratio": 1.1,
        },
        {
            "hub_repo_id": "a/one",
            "quality_ok": True,
            "method_name": "AWQ-again",
            "model_id": "org/m",
            "gpu_instance": "g5.2xlarge",
            "ppl_ratio": 1.2,
        },
        {
            "hub_repo_id": "b/two",
            "quality_ok": False,
            "method_name": "Broken",
            "model_id": "org/m",
            "gpu_instance": "g5.2xlarge",
            "ppl_ratio": 3.0,
        },
        {
            "hub_repo_id": "c/three",
            "quality_ok": True,
            "method_name": "FlatQuant",
            "model_id": "org/m",
            "gpu_instance": "g6e.xlarge",
            "ppl_ratio": 1.05,
        },
    ]
    items = library_mod.collection_entries(rows)
    assert [item["item_id"] for item in items] == ["a/one", "c/three"]


def test_repo_contributions_join_one_library():
    contrib_dir = ROOT / "library" / "contributions"
    files = list(contrib_dir.glob("*.json"))
    assert files, "expected at least one contribution JSON in the shared library"
    assert not (ROOT / "compare").exists()
    assert (ROOT / "library" / "LIBRARY.md").is_file()
    assert (ROOT / "library" / "ATTEMPTS.md").is_file()
    assert not (ROOT / "library" / "README.md").is_file()
    for path in files:
        library_mod.validate_contribution(json.loads(path.read_text()))
    rows = library_mod.query()
    assert any(row.get("hub_repo_id") for row in rows)
    assert any(row.get("paper_url") and row.get("repo_url") for row in rows)
    methods = {row["method_name"] for row in rows}
    assert "SpinQuant" not in methods
    assert "QuaRot" not in methods
    attempts = {row["method_name"] for row in library_mod.query_attempts()}
    assert "SpinQuant" in attempts
    assert "QuaRot" in attempts


def test_catalog_run_records_standard_links(tmp_path, monkeypatch):
    monkeypatch.setattr(paths_mod, "JOBS_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(paths_mod, "LIBRARY_ROOT", tmp_path / "library")
    monkeypatch.setattr(paths_mod, "REPO_ROOT", tmp_path)
    job_id = "20260101T000000Z-eeeeee"
    _write_job(tmp_path, job_id, packed=True, tps=5105.9, ratio=1.17)
    request = {
        "method_name": "FlatQuant",
        "model_id": "microsoft/Phi-3-mini-4k-instruct",
        "gpu_instance": "g5.2xlarge",
        "gpu_name": "NVIDIA A10G",
        "slug": "flatquant-phi3-mini-4k-g52xlarge",
        "arxiv_id": "2410.09426",
        "repo_url": "https://github.com/ruikangliu/FlatQuant",
        "repo_commit": "9d88ffcb7d2c6bda59fb5c44dad36adc101aadb1",
    }
    result = library_mod.catalog_run(
        job_id=job_id,
        request=request,
        hub_url="https://huggingface.co/you/flatquant-phi3",
    )
    assert result["status"] == "cataloged"
    assert result["paper_url"] == "https://arxiv.org/abs/2410.09426"
    assert result["repo_url"] == "https://github.com/ruikangliu/FlatQuant"
    row = library_mod.query(method_name="FlatQuant")[0]
    assert row["model_id"] == "microsoft/Phi-3-mini-4k-instruct"
    assert row["gpu_instance"] == "g5.2xlarge"
    assert row["hub_repo_id"] == "you/flatquant-phi3"
    readme = (tmp_path / "library" / "LIBRARY.md").read_text()
    assert "arxiv.org/abs/2410.09426" in readme
    assert "github.com/ruikangliu/FlatQuant" in readme
    assert "huggingface.co/you/flatquant-phi3" in readme
    try:
        library_mod.catalog_run(job_id=job_id, request=request, hub_url="")
    except ValueError as exc:
        assert "hub-url" in str(exc)
    else:
        raise AssertionError("expected missing hub url rejection")


def _serve_report(status="passed", ratio=1.04, drift=1.005, issues=None):
    return {
        "status": status,
        "issue_codes": issues or [],
        "quantized": {"vllm_version": "0.30.0", "gpu_name": "NVIDIA GeForce RTX 3090", "model_path": "/root/x"},
        "comparison": {
            "ppl_ratio": ratio,
            "hf_ppl_ratio": 1.035,
            "ppl_ratio_drift": drift,
            "prefill_speedup": 1.3,
            "decode_speedup": {"1": 1.2, "16": 1.1},
        },
    }


def _accept_awq(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(paths_mod, "LIBRARY_ROOT", tmp_path / "library")
    contrib = {
        "model_id": "org/model", "method_name": "SmoothQuant", "gpu_instance": "g5.2xlarge",
        "gpu_name": "NVIDIA A10G", "quality_ok": True, "ppl": 10.5, "ppl_ratio": 1.05,
        "tokens_per_s": 4000.0, "peak_vram_gb": 4.0, "fp16_ppl": 10.0, "fp16_tokens_per_s": 3000.0,
        "fp16_peak_vram_gb": 6.0, "hub_repo_id": "me/sq-model", "arxiv_id": "2211.10438",
        "repo_url": "https://github.com/mit-han-lab/smoothquant",
    }
    src = tmp_path / "incoming.json"
    src.write_text(json.dumps(contrib))
    library_mod.accept_contribution(src)


def test_record_vllm_passed_reaches_catalog_markdown_and_collection(tmp_path, monkeypatch):
    _accept_awq(tmp_path, monkeypatch)
    result = library_mod.record_vllm(
        model_id="org/model", gpu_instance="g5.2xlarge", method_name="SmoothQuant",
        export_record={"status": "exported", "profile": "w8a8_int8"},
        serve_report=_serve_report(), hub_repo_id="me/sq-model-vllm",
    )
    assert result["vllm"]["status"] == "passed"
    row = library_mod.query()[0]
    assert row["vllm"]["hub_repo_id"] == "me/sq-model-vllm"
    assert row["vllm"]["gpu_name"] == "NVIDIA GeForce RTX 3090"
    assert row["tokens_per_s"] == 4000.0  # the ranking numbers are untouched
    md = (tmp_path / "library" / "LIBRARY.md").read_text()
    assert "| vLLM |" in md
    assert "[vLLM](https://huggingface.co/me/sq-model-vllm)" in md
    assert "Measured on NVIDIA GeForce RTX 3090 with vLLM 0.30.0" in md
    items = [e["item_id"] for e in library_mod.collection_entries()]
    assert items == ["me/sq-model", "me/sq-model-vllm"]
    catalog = json.loads((tmp_path / "library" / "catalog.json").read_text())
    assert catalog["schema_version"] == library_mod.SCHEMA_VERSION == 2


def test_record_vllm_unsupported_and_failed_show_why(tmp_path, monkeypatch):
    _accept_awq(tmp_path, monkeypatch)
    library_mod.record_vllm(
        model_id="org/model", gpu_instance="g5.2xlarge", method_name="SmoothQuant",
        export_record={"status": "unsupported", "reason_code": "activation_bits_unsupported"},
    )
    assert "no (`activation_bits_unsupported`)" in (tmp_path / "library" / "LIBRARY.md").read_text()
    library_mod.record_vllm(
        model_id="org/model", gpu_instance="g5.2xlarge", method_name="SmoothQuant",
        export_record={"status": "exported"},
        serve_report=_serve_report("failed", ratio=1.2, drift=1.15, issues=["vllm_ppl_mismatch"]),
    )
    md = (tmp_path / "library" / "LIBRARY.md").read_text()
    assert "failed (`vllm_ppl_mismatch`)" in md
    assert library_mod.collection_entries() == [library_mod.collection_entries()[0]]


def test_record_vllm_needs_an_existing_row(tmp_path, monkeypatch):
    monkeypatch.setattr(paths_mod, "LIBRARY_ROOT", tmp_path / "library")
    import pytest

    with pytest.raises(RuntimeError, match="no library row"):
        library_mod.record_vllm(
            model_id="org/none", gpu_instance="g5.2xlarge", method_name="X",
            export_record={"status": "exported"},
        )


def test_vllm_block_rejects_unbacked_claims():
    import pytest

    with pytest.raises(ValueError, match="only a passed"):
        library_mod.validate_vllm_block({"status": "failed", "issue_codes": ["vllm_quality_failed"], "hub_repo_id": "me/x"})
    with pytest.raises(ValueError, match="drift"):
        library_mod.vllm_block({"status": "exported"}, {**_serve_report(), "status": "passed"} | {"comparison": {**_serve_report()["comparison"], "ppl_ratio_drift": 1.2}})
    with pytest.raises(ValueError, match="vllm_version"):
        library_mod.validate_vllm_block({"status": "passed", "ppl_ratio": 1.0, "gpu_name": "x"})
    pending = library_mod.vllm_block({"status": "exported", "profile": "w8a8_int8"})
    assert pending == {"status": "pending", "profile": "w8a8_int8"}


def test_record_vllm_keeps_the_baseline_dtype(tmp_path, monkeypatch):
    import pytest

    _accept_awq(tmp_path, monkeypatch)
    library_mod.record_vllm(
        model_id="org/model", gpu_instance="g5.2xlarge", method_name="SmoothQuant",
        export_record={"status": "exported", "profile": "w8a8_int8"},
        serve_report={**_serve_report(), "dtype": "bfloat16"}, hub_repo_id="me/sq-model-vllm",
    )
    assert library_mod.query()[0]["vllm"]["dtype"] == "bfloat16"
    assert "PPL ratio 1.0400 vs vLLM bf16" in (tmp_path / "library" / "LIBRARY.md").read_text()
    with pytest.raises(ValueError, match="vllm.dtype"):
        library_mod.validate_vllm_block({"status": "pending", "dtype": "float32"})


def test_record_vllm_keeps_the_same_gpu_transformers_comparison(tmp_path, monkeypatch):
    import pytest

    _accept_awq(tmp_path, monkeypatch)
    hf_runtime = {
        "gpu_name": "NVIDIA GeForce RTX 3090", "dtype": "float16", "transformers_version": "4.46.3",
        "quantized": {"peak_vram_gb": 4.68}, "baseline": {"peak_vram_gb": 5.84},
        "comparison": {
            "ppl_ratio": 1.05, "int8_prefill_speedup": 1.19, "int8_decode_speedup": {"1": 0.41, "16": 0.39},
            "vllm_prefill_speedup": 2.04, "vllm_decode_speedup": {"1": 18.23, "16": 16.68},
        },
    }
    library_mod.record_vllm(
        model_id="org/model", gpu_instance="g5.2xlarge", method_name="SmoothQuant",
        export_record={"status": "exported", "profile": "w8a8_int8"},
        serve_report={**_serve_report(), "hf_runtime": hf_runtime}, hub_repo_id="me/sq-model-vllm",
    )
    block = library_mod.query()[0]["vllm"]["hf_runtime"]
    assert block["vllm_decode_speedup"] == {"1": 18.23, "16": 16.68}
    assert block["peak_vram_gb"] == 4.68 and block["baseline_peak_vram_gb"] == 5.84
    md = (tmp_path / "library" / "LIBRARY.md").read_text()
    assert (
        "Against plain Transformers fp16 on the same GPU: vLLM int8 prefill 2.04×, decode 18.23× batch 1, "
        "16.68× batch 16; the native artifact in Transformers prefill 1.19×, decode 0.41× batch 1, 0.39× batch 16."
    ) in md
    with pytest.raises(ValueError, match="hf_runtime"):
        library_mod.validate_vllm_block({"status": "pending", "hf_runtime": {"ppl_ratio": 1.0}})
    with pytest.raises(ValueError, match="int8_decode_speedup"):
        library_mod.validate_vllm_block({
            "status": "passed", "vllm_version": "0.30.0", "gpu_name": "x", "ppl_ratio": 1.0,
            "hf_runtime": {**library_mod._hf_runtime_summary(hf_runtime), "int8_decode_speedup": {"1": "fast"}},
        })

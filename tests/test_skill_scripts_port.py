"""quant-port planning helpers: arch probe summary, smoke slice/gate, spec, rank, recipes."""
from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "agents" / "skills" / "_shared" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import arch_probe  # noqa: E402
import library as library_mod  # noqa: E402
import paths as paths_mod  # noqa: E402
import port_plan  # noqa: E402
import port_smoke  # noqa: E402
import request as request_mod  # noqa: E402
from overlay import PortOverlaySession, validate_overlay_bundle  # noqa: E402
from port_smoke import SmokeSettings  # noqa: E402
from validate_script import validate_script  # noqa: E402

SLUG = "quarot-qwen25-15b-g52xlarge"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
COMMIT = "a" * 40
PATCH = """diff --git a/quant/modeling.py b/quant/modeling.py
--- a/quant/modeling.py
+++ b/quant/modeling.py
@@ -1 +1,2 @@
 SUPPORTED = [\"llama\"]
+SUPPORTED.append(\"qwen2\")
"""
BIG_PATCH = """diff --git a/quant/modeling.py b/quant/modeling.py
--- a/quant/modeling.py
+++ b/quant/modeling.py
@@ -1 +1,4 @@
 SUPPORTED = [\"llama\"]
+SUPPORTED.append(\"qwen2\")
+SUPPORTED.append(\"qwen2_moe\")
+BIAS = True
diff --git a/quant/rotate.py b/quant/rotate.py
--- a/quant/rotate.py
+++ b/quant/rotate.py
@@ -1 +1,2 @@
 ROTATE = 1
+ROTATE_BIAS = 1
"""


# --- arch probe ---------------------------------------------------------------


def _qwen2_dump(num_layers: int = 2) -> dict:
    modules = [
        {"name": "", "cls": "Qwen2ForCausalLM"},
        {"name": "model", "cls": "Qwen2Model"},
        {"name": "model.embed_tokens", "cls": "Embedding", "params": {"weight": [151936, 1536]}},
        {"name": "model.layers", "cls": "ModuleList"},
    ]
    for idx in range(num_layers):
        base = f"model.layers.{idx}"
        modules += [
            {"name": base, "cls": "Qwen2DecoderLayer"},
            {"name": f"{base}.self_attn", "cls": "Qwen2Attention"},
        ]
        for proj, out_features, bias in (("q_proj", 1536, True), ("k_proj", 256, True), ("v_proj", 256, True), ("o_proj", 1536, False)):
            params = {"weight": [out_features, 1536]}
            if bias:
                params["bias"] = [out_features]
            modules.append({
                "name": f"{base}.self_attn.{proj}", "cls": "Linear", "params": params,
                "in_features": 1536, "out_features": out_features,
            })
        modules.append({"name": f"{base}.mlp", "cls": "Qwen2MLP"})
        for proj, inf, outf in (("gate_proj", 1536, 8960), ("up_proj", 1536, 8960), ("down_proj", 8960, 1536)):
            modules.append({
                "name": f"{base}.mlp.{proj}", "cls": "Linear", "params": {"weight": [outf, inf]},
                "in_features": inf, "out_features": outf,
            })
        for norm in ("input_layernorm", "post_attention_layernorm"):
            modules.append({"name": f"{base}.{norm}", "cls": "Qwen2RMSNorm", "params": {"weight": [1536]}})
    modules += [
        {"name": "model.norm", "cls": "Qwen2RMSNorm", "params": {"weight": [1536]}},
        {"name": "lm_head", "cls": "Linear", "params": {"weight": [151936, 1536]}, "in_features": 1536, "out_features": 151936},
    ]
    return {
        "model_class": "Qwen2ForCausalLM",
        "tied_embeddings": True,
        "config": {
            "model_type": "qwen2", "hidden_size": 1536, "num_attention_heads": 12,
            "num_key_value_heads": 2, "num_hidden_layers": num_layers, "hidden_act": "silu",
            "tie_word_embeddings": True, "irrelevant_key": 1,
        },
        "modules": modules,
    }


def test_summarize_flags_where_qwen2_departs_from_llama():
    summary = arch_probe.summarize(_qwen2_dump(), model_id=MODEL_ID, snapshot="/snap")
    assert summary["layer_container"] == "model.layers"
    assert summary["num_layers"] == 2
    assert summary["layer_groups"] == [[0, 1]]
    assert summary["layer0"]["attention"] == "Qwen2Attention"
    assert summary["layer0"]["mlp"] == "Qwen2MLP"
    assert "irrelevant_key" not in summary["config"]
    flags = set(summary["flags"])
    assert {"attention_bias", "gqa", "tied_embeddings"} <= flags
    assert not flags & {"mlp_bias", "qk_norm", "moe", "heterogeneous_layers", "non_silu_activation"}
    linears = {lin["name"]: lin for lin in summary["layer0"]["linears"]}
    assert linears["self_attn.q_proj"]["bias"] and linears["self_attn.q_proj"]["in_attention"]
    assert not linears["mlp.down_proj"]["bias"] and not linears["mlp.down_proj"]["in_attention"]
    assert {entry["name"] for entry in summary["top_level"]} >= {"model.embed_tokens", "lm_head", "model.norm"}


def test_summarize_collapses_experts_and_marks_moe():
    raw = _qwen2_dump(1)
    raw["modules"] = [m for m in raw["modules"] if ".mlp." not in m["name"]]
    for expert in range(3):
        raw["modules"].append({
            "name": f"model.layers.0.mlp.experts.{expert}.down_proj", "cls": "Linear",
            "params": {"weight": [1536, 1024]}, "in_features": 1024, "out_features": 1536,
        })
    summary = arch_probe.summarize(raw)
    assert "moe" in summary["flags"]
    assert summary["layer0"]["num_experts"] == 3
    names = [lin["name"] for lin in summary["layer0"]["linears"]]
    assert names.count("mlp.experts.*.down_proj") == 1


def test_wants_remote_code(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
    assert not arch_probe.wants_remote_code(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"auto_map": {"AutoModel": "x.Y"}}))
    assert arch_probe.wants_remote_code(tmp_path)


# --- smoke slice ----------------------------------------------------------------


def test_truncate_config_cuts_layer_count_and_per_layer_lists():
    config = {
        "num_hidden_layers": 4,
        "layer_types": ["sliding", "full", "sliding", "full"],
        "hidden_size": 8,
        "text_config": {"num_hidden_layers": 4},
    }
    out = port_smoke.truncate_config(config, 2)
    assert out["num_hidden_layers"] == 2
    assert out["layer_types"] == ["sliding", "full"]
    assert out["text_config"]["num_hidden_layers"] == 2
    assert config["num_hidden_layers"] == 4  # input untouched
    assert port_smoke.truncate_config({"n_layer": 1}, 2)["n_layer"] == 1
    with pytest.raises(ValueError):
        port_smoke.truncate_config({"hidden_size": 8}, 2)


@pytest.mark.parametrize(
    ("key", "kept"),
    [
        ("model.layers.0.self_attn.q_proj.weight", True),
        ("model.layers.1.mlp.down_proj.weight", True),
        ("model.layers.2.mlp.down_proj.weight", False),
        ("model.layers.10.input_layernorm.weight", False),
        ("transformer.h.3.attn.c_attn.weight", False),
        ("transformer.h.1.attn.c_attn.weight", True),
        ("model.embed_tokens.weight", True),
        ("lm_head.weight", True),
        ("model.norm.weight", True),
    ],
)
def test_keep_key(key, kept):
    assert port_smoke.keep_key(key, 2) is kept


def test_safetensors_keys_reads_the_header_without_torch(tmp_path):
    header = {
        "__metadata__": {"format": "pt"},
        "a.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]},
        "b.weight": {"dtype": "F16", "shape": [1], "data_offsets": [2, 4]},
    }
    raw = json.dumps(header).encode()
    path = tmp_path / "model.safetensors"
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * 4)
    assert sorted(port_smoke.safetensors_keys(path)) == ["a.weight", "b.weight"]


# --- smoke gate -------------------------------------------------------------------


def _stat(**kwargs):
    base = {"cls": "Linear", "is_plain_linear": True, "int_storage": False, "unique_per_row_max": 1536, "in_features": 1536}
    return {**base, **kwargs}


def test_classify_tells_packed_fake_quant_wrapped_float_missing():
    assert port_smoke.classify(None, weight_bits=4, group_size=None) == "missing"
    assert port_smoke.classify(_stat(int_storage=True), weight_bits=4, group_size=None) == "packed"
    assert port_smoke.classify(_stat(unique_per_row_max=16), weight_bits=4, group_size=None) == "fake_quant"
    # Group-wise: up to 2**bits values per group.
    assert port_smoke.classify(_stat(unique_per_row_max=190), weight_bits=4, group_size=128) == "fake_quant"
    assert port_smoke.classify(_stat(unique_per_row_max=1500), weight_bits=4, group_size=None) == "float"
    assert port_smoke.classify(_stat(is_plain_linear=False, cls="QLinear", unique_per_row_max=None), weight_bits=4, group_size=None) == "wrapped"


def _check(**kwargs):
    base = {
        "finite": True,
        "shape_match": True,
        "rel_logit_error": 0.1,
        "top1_agreement": 0.9,
        "kl": 0.02,
        "targets": {
            "model.layers.0.self_attn.q_proj": _stat(int_storage=True),
            "model.layers.0.mlp.down_proj": _stat(unique_per_row_max=16, in_features=8960),
        },
    }
    return {**base, **kwargs}


def test_evaluate_passes_a_fully_quantized_close_slice():
    result = port_smoke.evaluate(_check(), SmokeSettings(weight_bits=4))
    assert result["status"] == "passed", result
    assert result["metrics"]["coverage"] == 1.0
    assert result["metrics"]["classes"]["packed"] == 1
    assert result["metrics"]["classes"]["fake_quant"] == 1


def test_evaluate_fails_float_linears_large_error_and_nan():
    targets = _check()["targets"] | {"model.layers.1.self_attn.k_proj": _stat()}
    result = port_smoke.evaluate(_check(targets=targets), SmokeSettings(weight_bits=4))
    assert result["status"] == "failed"
    assert any(reason.startswith("coverage") for reason in result["reasons"])
    assert result["uncovered"] == ["model.layers.1.self_attn.k_proj"]
    assert port_smoke.evaluate(_check(targets=targets), SmokeSettings(weight_bits=4, min_coverage=0.5))["status"] == "passed"

    big = port_smoke.evaluate(_check(rel_logit_error=0.9), SmokeSettings(weight_bits=4))
    assert big["status"] == "failed" and big["reasons"][0].startswith("rel_logit_error")

    nan = port_smoke.evaluate(_check(finite=False, rel_logit_error=None), SmokeSettings(weight_bits=4))
    assert "non_finite_logits" in nan["reasons"]

    empty = port_smoke.evaluate(_check(targets={}), SmokeSettings())
    assert "no_target_linears" in empty["reasons"]


def test_load_settings_reads_the_spec_smoke_block_and_overrides(tmp_path):
    spec = tmp_path / "differences.json"
    spec.write_text(json.dumps({"smoke": {"weight_bits": 4, "group_size": 128}}))
    settings = port_smoke.load_settings(spec, {"max_rel_error": 0.3, "group_size": None})
    assert (settings.weight_bits, settings.group_size, settings.max_rel_error) == (4, 128, 0.3)
    assert port_smoke.load_settings(tmp_path / "missing.json", {}).weight_bits == 8


def test_smoke_dir_mirrors_the_overlay_layout(tmp_path, monkeypatch):
    monkeypatch.setattr(paths_mod, "SMOKE_ROOT", tmp_path / "out" / "smoke")
    bundle = tmp_path / "out" / "overlays" / SLUG / "bias_rotate" / "0123456789abcdef"
    assert port_smoke.smoke_dir(SLUG, bundle) == tmp_path / "out" / "smoke" / SLUG / "bias_rotate" / "0123456789abcdef"


# --- overlay addresses and wrapper hooks --------------------------------------------


def test_manifest_addresses_are_sorted_validated_and_change_the_hash(tmp_path):
    plain = PortOverlaySession(root=tmp_path / "a", method_id=SLUG, model_id=MODEL_ID, base_commit=COMMIT)
    plain_result = plain.write(patch=PATCH, rationale="add qwen2")
    addressed = PortOverlaySession(root=tmp_path / "a", method_id=SLUG, model_id=MODEL_ID, base_commit=COMMIT)
    result = addressed.write(patch=PATCH, rationale="add qwen2", addresses=["D3", "D1", "D3"])
    assert result["status"] == "ok"
    assert result["manifest"]["addresses"] == ["D1", "D3"]
    assert Path(result["overlay_dir"]).name != Path(plain_result["overlay_dir"]).name
    assert validate_overlay_bundle(Path(result["overlay_dir"])).addresses == ["D1", "D3"]
    # No addresses: same hash as before the field existed.
    assert plain_result["manifest"]["addresses"] == []

    bad = PortOverlaySession(root=tmp_path / "b", method_id=SLUG, model_id=MODEL_ID, base_commit=COMMIT)
    rejected = bad.write(patch=PATCH, rationale="x", addresses=["bias"])
    assert rejected["status"] == "error" and "difference ids" in rejected["message"]


def test_validate_script_smoke_hooks_are_opt_in():
    code = f'MODEL_ID = "{MODEL_ID}"\nOUTPUT_DIR = "./quantized/{SLUG}"\n'
    assert validate_script(code, model_id=MODEL_ID, output_dir=f"./quantized/{SLUG}")["ok"]
    missing = validate_script(code, model_id=MODEL_ID, output_dir=f"./quantized/{SLUG}", require_smoke_hooks=True)
    assert not missing["ok"]
    assert "QUANT_AGENT_SMOKE_MODEL_DIR" in missing["message"]
    hooked = code + (
        'import os\nMODEL = os.environ.get("QUANT_AGENT_SMOKE_MODEL_DIR") or MODEL_ID\n'
        'OUT = os.environ.get("QUANT_AGENT_SMOKE_OUTPUT_DIR") or OUTPUT_DIR\n'
    )
    assert validate_script(hooked, model_id=MODEL_ID, output_dir=f"./quantized/{SLUG}", require_smoke_hooks=True)["ok"]


def test_request_accepts_port_spec_path():
    assert "port_spec_path" in request_mod.ALLOWED_KEYS


# --- spec -------------------------------------------------------------------------


def _spec(**kwargs) -> dict:
    base = {
        "schema_version": 1,
        "slug": SLUG,
        "method_name": "QuaRot",
        "model_id": MODEL_ID,
        "model_type": "qwen2",
        "mapping": [
            {"role": "attention input projections", "steps": ["rotate", "quantize"],
             "modules": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"], "status": "mapped"},
            {"role": "linear biases", "steps": ["rotate"], "modules": ["self_attn.q_proj.bias"], "status": "extra"},
            {"role": "embeddings", "steps": ["rotate"], "modules": ["model.embed_tokens", "lm_head"], "status": "partial"},
            {"role": "value/output pair", "steps": ["rotate"], "modules": ["self_attn.v_proj", "self_attn.o_proj"],
             "status": "partial", "note": "2 kv heads"},
        ],
        "differences": [
            {"id": "D1", "kind": "attention_bias", "role": "linear biases", "severity": "must",
             "detail": "q/k/v bias not rotated", "evidence": ["arch.json flags attention_bias"],
             "repo_support": "missing"},
            {"id": "D2", "kind": "tied_embeddings", "role": "embeddings", "severity": "must",
             "detail": "embedding tied to lm_head", "evidence": ["arch.json tied_embeddings"],
             "repo_support": "partial"},
            {"id": "D3", "kind": "gqa", "role": "value/output pair", "severity": "should", "detail": "2 kv heads"},
        ],
        "smoke": {"weight_bits": 4},
    }
    return {**base, **kwargs}


METHOD_MD = """# QuaRot

## Steps
1. Fuse RMSNorm scales into the next linears (paper 4.1, quarot/rotation.py:40).
2. Rotate the residual stream with a random Hadamard Q on every reader and writer.
3. Quantize weights with GPTQ and activations per token to 4 bits.

## Tensor roles
Residual stream readers (q/k/v, gate/up) rotate on the input side; writers
(o_proj, down_proj, embeddings) rotate on the output side. LM head reads it.

## Invariants
Q is orthogonal, so Q Q^T = I keeps the function unchanged before quantization,
provided every reader and writer of the residual stream sees the same Q.

## Calibration
128 WikiText-2 samples of 2048 tokens for GPTQ; no training iterations.

## Save and reload
Packed int4 weights with fp16 scales in Linear4bit; reload rebuilds the
Hadamard online rotation before down_proj.
"""


def _write_spec(tmp_path, data=None, method: str | None = METHOD_MD) -> Path:
    path = tmp_path / "out" / "analysis" / SLUG / "differences.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data or _spec()))
    if method is not None:
        (path.parent / "method.md").write_text(method)
    return path


def test_spec_check_counts_and_matches_the_request(tmp_path):
    path = _write_spec(tmp_path)
    result = port_plan.check_spec(path, {"slug": SLUG, "model_id": MODEL_ID})
    assert result["status"] == "ok"
    assert result["must"] == ["D1", "D2"]
    assert result["severity"] == {"must": 2, "should": 1, "info": 0}
    assert result["missing_in_repo"] == ["D1", "D2"]
    assert result["smoke"]["weight_bits"] == 4
    assert result["method"].endswith("method.md")
    assert result["mapping"]["extra"] == ["linear biases"]
    assert result["mapping"]["partial"] == ["embeddings", "value/output pair"]
    with pytest.raises(ValueError):
        port_plan.check_spec(path, {"slug": SLUG, "model_id": "other/model"})


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s["differences"].append(dict(s["differences"][0])),  # duplicate id
        lambda s: s["differences"][0].update(id="1"),
        lambda s: s["differences"][0].update(severity="critical"),
        lambda s: s["differences"][0].update(evidence=[]),  # must without evidence
        lambda s: s.update(unexpected=True),
        lambda s: s.update(slug="Bad Slug"),
        lambda s: s["smoke"].update(max_rel_error=0),
        lambda s: s.update(mapping=[]),
        lambda s: s["mapping"].append(dict(s["mapping"][0])),  # duplicate role
        lambda s: s["mapping"][0].update(status="fine"),
        lambda s: s["differences"][0].update(role="not a mapped role"),
        lambda s: s["differences"].pop(),  # partial role left without a difference
    ],
)
def test_spec_rejects_bad_input(tmp_path, mutate):
    data = _spec()
    mutate(data)
    with pytest.raises(ValidationError):
        port_plan.load_spec(_write_spec(tmp_path, data))


def test_spec_check_requires_a_real_method_model(tmp_path):
    with pytest.raises(ValueError, match="write the method model"):
        port_plan.check_spec(_write_spec(tmp_path, method=None))
    without_invariants = METHOD_MD.replace("## Invariants", "## Thoughts")
    with pytest.raises(ValueError, match="lacks sections.*Invariants"):
        port_plan.check_spec(_write_spec(tmp_path, method=without_invariants))
    thin = METHOD_MD.replace("128 WikiText-2 samples of 2048 tokens for GPTQ; no training iterations.", "TBD")
    with pytest.raises(ValueError, match="too thin.*Calibration"):
        port_plan.check_spec(_write_spec(tmp_path, method=thin))
    lower = METHOD_MD.replace("## Save and reload", "## save and reload")
    assert port_plan.check_spec(_write_spec(tmp_path, method=lower))["status"] == "ok"


# --- rank -------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(paths_mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(paths_mod, "OVERLAYS_ROOT", tmp_path / "out" / "overlays")
    monkeypatch.setattr(paths_mod, "SMOKE_ROOT", tmp_path / "out" / "smoke")
    monkeypatch.setattr(paths_mod, "ANALYSIS_ROOT", tmp_path / "out" / "analysis")
    monkeypatch.setattr(paths_mod, "LIBRARY_ROOT", tmp_path / "library")
    monkeypatch.setattr(paths_mod, "JOBS_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(request_mod, "REQUESTS_ROOT", tmp_path / "out" / "requests")
    return tmp_path


def _candidate(
    root: Path,
    plan: str,
    *,
    patch: str = PATCH,
    addresses: list[str] | None = None,
    status: str = "passed",
    coverage: float = 1.0,
    rel: float = 0.1,
    settings: dict | None = None,
    smoke: bool = True,
) -> Path:
    session = PortOverlaySession(
        root=root / "out" / "overlays" / SLUG / plan, method_id=SLUG, model_id=MODEL_ID, base_commit=COMMIT,
    )
    result = session.write(patch=patch, rationale=plan, addresses=addresses)
    assert result["status"] == "ok", result
    bundle = Path(result["overlay_dir"])
    script = bundle.parent / "quantize.py"
    code = f"# QUANT_AGENT_OVERLAY_DIR={bundle}\nMODEL_ID = \"{MODEL_ID}\"\n"
    script.write_text(code)
    if smoke:
        dest = port_smoke.smoke_dir(SLUG, bundle)
        dest.mkdir(parents=True)
        (dest / "smoke.json").write_text(json.dumps({
            "status": status,
            "stage": "check",
            "reasons": [] if status == "passed" else ["coverage 0.50 < 1.0"],
            "overlay_dir": str(bundle),
            "script": str(script),
            "script_sha256": hashlib.sha256(code.encode()).hexdigest(),
            "settings": settings or SmokeSettings(weight_bits=4).model_dump(),
            "metrics": {"coverage": coverage, "rel_logit_error": rel},
        }))
    return bundle


def test_rank_orders_by_gate_then_must_coverage_then_metrics_then_size(workspace):
    spec = port_plan.PortSpec.model_validate(_spec())
    small_no_must = _candidate(workspace, "dispatch_only", addresses=["D3"], rel=0.01)
    full_big = _candidate(workspace, "bias_rotate", patch=BIG_PATCH, addresses=["D1", "D2"], rel=0.2)
    full_small = _candidate(workspace, "bias_rotate_min", addresses=["D1", "D2", "D3"], rel=0.2)
    failed = _candidate(workspace, "adapter_reload", addresses=["D1", "D2"], status="failed", coverage=0.5)
    unsmoked = _candidate(workspace, "unsmoked", addresses=["D1", "D2"], smoke=False)

    result = port_plan.rank(SLUG, spec, [failed, small_no_must, full_big, full_small, unsmoked])
    assert result["status"] == "ok"
    assert result["ranked"] == [str(full_small), str(full_big), str(small_no_must)]
    assert result["winner_overlay_dir"] == str(full_small)
    assert result["winner_script"] == str(full_small.parent / "quantize.py")
    by_dir = {entry["overlay_dir"]: entry for entry in result["candidates"]}
    assert by_dir[str(failed)]["reasons"] == ["smoke_failed"]
    assert by_dir[str(unsmoked)]["reasons"] == ["not_smoked"]
    assert by_dir[str(small_no_must)]["must_missing"] == ["D1", "D2"]
    assert by_dir[str(full_big)]["files_touched"] == 2
    assert by_dir[str(full_big)]["patch_lines"] == 4


def test_rank_drops_stale_smokes(workspace):
    spec = port_plan.PortSpec.model_validate(_spec())
    other_settings = _candidate(
        workspace, "loose", addresses=["D1", "D2"],
        settings=SmokeSettings(weight_bits=4, max_rel_error=5).model_dump(),
    )
    edited = _candidate(workspace, "edited", addresses=["D1", "D2"])
    (edited.parent / "quantize.py").write_text(f"# QUANT_AGENT_OVERLAY_DIR={edited}\n# edited after smoke\n")
    elsewhere = _candidate(workspace, "elsewhere", addresses=["D1", "D2"])
    script = elsewhere.parent / "quantize.py"
    code = f"# QUANT_AGENT_OVERLAY_DIR={edited}\n"
    script.write_text(code)
    smoke_json = port_smoke.smoke_dir(SLUG, elsewhere) / "smoke.json"
    smoke = json.loads(smoke_json.read_text())
    smoke["script_sha256"] = hashlib.sha256(code.encode()).hexdigest()
    smoke_json.write_text(json.dumps(smoke))

    result = port_plan.rank(SLUG, spec, [other_settings, edited, elsewhere])
    assert result["status"] == "no_passing_overlay"
    assert result["ranked"] == [] and result["winner_overlay_dir"] is None
    reasons = {Path(entry["overlay_dir"]).parent.name: entry["reasons"] for entry in result["candidates"]}
    assert reasons["loose"] == ["smoke_settings_changed"]
    assert reasons["edited"] == ["script_changed_since_smoke"]
    assert reasons["elsewhere"] == ["script_points_at_another_overlay"]


def test_rank_write_stores_the_winner_in_the_request(workspace):
    spec_path = _write_spec(workspace)
    winner = _candidate(workspace, "bias_rotate", addresses=["D1", "D2"])
    request = {
        "method_name": "QuaRot", "model_id": MODEL_ID, "gpu_instance": "g5.2xlarge",
        "arxiv_id": "2404.00456", "paper_path": "p", "repo_url": "https://github.com/spcl/QuaRot",
        "repo_path": "r", "repo_commit": COMMIT, "hf_snapshot_path": "s", "slug": SLUG,
    }
    path = request_mod.write_request(request)
    assert port_plan.discover_overlays(SLUG) == [winner]
    result = port_plan.rank(SLUG, port_plan.load_spec(spec_path), port_plan.discover_overlays(SLUG))
    port_plan.write_winner(path, result, spec_path)
    stored = request_mod.load_request(path)
    assert stored["winner_overlay_dir"] == str(winner)
    assert stored["winner_script"] == str(winner.parent / "quantize.py")
    assert stored["ranked"] == [str(winner)]
    assert stored["port_spec_path"] == str(spec_path)


# --- recipes and library ports -------------------------------------------------------


def _write_row(root: Path, folder: str, stem: str, row: dict, port: dict | None = None) -> None:
    rows = root / "library" / folder
    rows.mkdir(parents=True, exist_ok=True)
    if port is not None:
        ref = f"library/ports/{'attempts/' if folder == 'attempts' else ''}{stem}"
        row = {**row, "port": ref}
        port_dir = root / ref
        port_dir.mkdir(parents=True)
        (port_dir / "port.json").write_text(json.dumps(port))
        (port_dir / "manifest.json").write_text(json.dumps({"rationale": "rotate bias", "addresses": ["D1"]}))
        (port_dir / "differences.json").write_text(json.dumps(_spec()))
        (port_dir / "method.md").write_text(METHOD_MD)
    (rows / f"{stem}.json").write_text(json.dumps(row))


def test_recipes_find_positive_and_negative_rows_by_model_or_family(workspace):
    _write_row(workspace, "attempts", "Qwen_Qwen2.5-1.5B-Instruct__g5.2xlarge__QuaRot", {
        "model_id": MODEL_ID, "method_name": "QuaRot", "gpu_instance": "g5.2xlarge",
        "quality_ok": False, "root_cause": "packed_quality_gap", "issue_codes": ["packed_quality_gap"],
        "best_ppl_ratio": 1.504, "notes": ["n1", "n2", "n3", "n4", "n5"],
    })
    _write_row(workspace, "contributions", "Qwen_Qwen2.5-7B-Instruct__g5.2xlarge__QuaRot", {
        "model_id": "Qwen/Qwen2.5-7B-Instruct", "method_name": "QuaRot", "gpu_instance": "g5.2xlarge",
        "quality_ok": True, "ppl_ratio": 1.08, "packed": True,
    }, port={"model_type": "qwen2"})
    _write_row(workspace, "contributions", "meta-llama_Llama-3.2-1B__g5.2xlarge__QuaRot", {
        "model_id": "meta-llama/Llama-3.2-1B", "method_name": "QuaRot", "quality_ok": True,
    }, port={"model_type": "llama"})
    _write_row(workspace, "contributions", "Qwen_Qwen2.5-1.5B-Instruct__g5.2xlarge__SmoothQuant", {
        "model_id": MODEL_ID, "method_name": "SmoothQuant", "quality_ok": True,
    })

    result = port_plan.recipes(method="quarot", model_type="qwen2", model_id="Qwen/Qwen2.5-0.5B-Instruct")
    assert [(r["outcome"], r["match"], r["model_id"]) for r in result["recipes"]] == [
        ("positive", "model_type", "Qwen/Qwen2.5-7B-Instruct"),
    ]
    positive = result["recipes"][0]
    assert positive["port"]["addresses"] == ["D1"]
    assert positive["port"]["differences"][0]["id"] == "D1"
    assert positive["port"]["differences"][0]["role"] == "linear biases"
    assert positive["port"]["mapping"][1] == {"role": "linear biases", "status": "extra", "modules": ["self_attn.q_proj.bias"]}
    assert positive["port"]["method"].endswith("/method.md")

    same_model = port_plan.recipes(method="QuaRot", model_type="qwen2", model_id=MODEL_ID)["recipes"]
    assert [r["outcome"] for r in same_model] == ["negative", "positive"]
    negative = same_model[0]
    assert negative["root_cause"] == "packed_quality_gap"
    assert negative["best_ppl_ratio"] == 1.504
    assert len(negative["notes"]) == 4


def test_snapshot_model_type(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
    assert port_plan.snapshot_model_type({"hf_snapshot_path": str(tmp_path)}) == "qwen2"
    assert port_plan.snapshot_model_type({}) is None


def test_library_port_carries_model_type_and_the_spec(workspace):
    job_id = "20260101T000000Z-aaaaa1"
    job = workspace / "jobs" / job_id
    (job / "overlay").mkdir(parents=True)
    (job / "overlay" / "overlay.patch").write_text(PATCH)
    (job / "overlay" / "manifest.json").write_text(json.dumps({
        "base_commit": COMMIT, "patch_sha256": hashlib.sha256(PATCH.encode()).hexdigest(), "addresses": ["D1"],
    }))
    (job / "script.py").write_text(f'MODEL_ID = "{MODEL_ID}"\n')
    snapshot = workspace / "snap"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
    spec_path = _write_spec(workspace, _spec(notes=f"probe at {workspace}/out/analysis"))
    request = {
        "method_name": "QuaRot", "model_id": MODEL_ID, "gpu_instance": "g5.2xlarge", "slug": SLUG,
        "repo_url": "https://github.com/spcl/QuaRot", "repo_commit": COMMIT, "hf_snapshot_path": str(snapshot),
    }
    port = library_mod.build_port(job_id=job_id, request=request, kind="attempt")
    assert port["status"] == "ready", port
    record = json.loads(port["files"]["port.json"])
    assert record["model_type"] == "qwen2"
    assert {"differences.json", "method.md"} <= set(record["files"])
    assert port["files"]["method.md"].decode() == METHOD_MD
    stored_spec = port["files"]["differences.json"].decode()
    assert str(workspace) not in stored_spec and "<repo>/out/analysis" in stored_spec

    spec_path.write_text(json.dumps(_spec(notes="hf_" + "a" * 34)))
    assert library_mod.build_port(job_id=job_id, request=request, kind="attempt") == {
        "status": "skipped", "reason": "secret_like_content",
    }
    spec_path.unlink()
    (spec_path.parent / "method.md").unlink()
    port = library_mod.build_port(job_id=job_id, request=request, kind="attempt")
    assert not {"differences.json", "method.md"} & set(json.loads(port["files"]["port.json"])["files"])

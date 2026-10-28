"""Offline tests for the vLLM export: header rewrite, profile detection, refusals."""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "agents" / "skills" / "_shared" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import export_vllm as export_mod  # noqa: E402

_PACK = {"F16": "e", "F32": "f", "I8": "b", "U8": "B"}


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], list]]) -> None:
    header: dict = {}
    blobs = []
    offset = 0
    for name, (dtype, shape, values) in tensors.items():
        blob = struct.pack(f"<{len(values)}{_PACK[dtype]}", *values)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(blob)]}
        blobs.append(blob)
        offset += len(blob)
    encoded = json.dumps(header).encode()
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for blob in blobs:
            handle.write(blob)


def read_tensor_bytes(path: Path, name: str) -> tuple[dict, bytes]:
    tensors, _ = export_mod.read_safetensors_header(path)
    info = next(t for t in tensors if t.name == name)
    with path.open("rb") as handle:
        handle.seek(info.start)
        return {"dtype": info.dtype, "shape": info.shape}, handle.read(info.nbytes)


def make_w8a8_artifact(root: Path, *, tied: bool = True, qcfg: dict | None = None) -> Path:
    """Tiny Llama-like model: vocab 6, hidden 4, one layer with int8 q_proj / o_proj."""
    root.mkdir(parents=True)
    embed = [0.5, -0.25, 1.0, 0.0] * 6
    weight_o = [1.0, -2.0, 0.5, 0.25] * 4  # fp16 reference for o_proj
    scale_o = [max(abs(v) for v in weight_o[i * 4:(i + 1) * 4]) / 127 for i in range(4)]
    q_o = [round(weight_o[i] / scale_o[i // 4]) for i in range(16)]
    write_safetensors(
        root / "model.safetensors",
        {
            "model.embed_tokens.weight": ("F16", [6, 4], embed),
            "model.layers.0.input_layernorm.weight": ("F16", [4], [1.0] * 4),
            "model.layers.0.self_attn.q_proj.weight": ("I8", [4, 4], [1, -2, 3, -4] * 4),
            "model.layers.0.self_attn.q_proj.weight_scales": ("F16", [4], [0.01, 0.02, 0.03, 0.04]),
            "model.layers.0.self_attn.q_proj.bias": ("F16", [4], [0.0] * 4),
            "model.layers.0.self_attn.o_proj.weight": ("I8", [4, 4], q_o),
            "model.layers.0.self_attn.o_proj.weight_scales": ("F16", [4], scale_o),
            "model.layers.0.mlp.gate_proj.weight": ("F16", [4, 4], [0.1] * 16),
            "model.norm.weight": ("F16", [4], [1.0] * 4),
            "lm_head.weight": ("F16", [6, 4], embed if tied else [0.75] * 24),
        },
    )
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "torch_dtype": "float16",
                "tie_word_embeddings": tied,
                "auto_map": {"AutoModelForCausalLM": "modeling_x.Model"},
            }
        )
    )
    (root / "quantization_config.json").write_text(
        json.dumps(qcfg if qcfg is not None else {"format": "packed_w8a8", "act_quant": "per_token", "weight_quant": "per_channel"})
    )
    (root / "tokenizer.json").write_text("{}")
    (root / "generation_config.json").write_text("{}")
    (root / "LICENSE").write_text("license")
    (root / "quant_agent_inference_adapter.py").write_text("raise SystemExit\n")
    (root / "overlay.patch").write_text("")
    write_reference(root.parent / "ref", weight_o)
    return root


def write_reference(root: Path, weight_o: list[float]) -> None:
    root.mkdir(exist_ok=True)
    write_safetensors(
        root / "model.safetensors",
        {
            "model.layers.0.self_attn.o_proj.weight": ("F16", [4, 4], weight_o),
            # q_proj is "smoothed" in the artifact, so it differs from the reference.
            "model.layers.0.self_attn.q_proj.weight": ("F16", [4, 4], [9.0] * 16),
        },
    )


def test_w8a8_export_renames_scales_and_writes_compressed_tensors_config(tmp_path):
    src = make_w8a8_artifact(tmp_path / "art")
    out = tmp_path / "out"
    record = export_mod.export(src, out)

    assert record["status"] == "exported"
    assert record["vllm_servable"] is True
    assert record["profile"] == "w8a8_int8"
    assert record["n_quantized_linears"] == 2
    meta, data = read_tensor_bytes(out / "model.safetensors", "model.layers.0.self_attn.q_proj.weight_scale")
    assert meta == {"dtype": "F16", "shape": [4, 1]}
    _, original = read_tensor_bytes(src / "model.safetensors", "model.layers.0.self_attn.q_proj.weight_scales")
    assert data == original
    _, q_new = read_tensor_bytes(out / "model.safetensors", "model.layers.0.self_attn.q_proj.weight")
    _, q_old = read_tensor_bytes(src / "model.safetensors", "model.layers.0.self_attn.q_proj.weight")
    assert q_new == q_old
    names = {t.name for t in export_mod.read_safetensors_header(out / "model.safetensors")[0]}
    assert "model.layers.0.self_attn.q_proj.weight_scales" not in names
    assert "lm_head.weight" not in names  # tied + identical
    assert "model.layers.0.self_attn.q_proj.bias" in names

    config = json.loads((out / "config.json").read_text())
    assert "auto_map" not in config
    qc = config["quantization_config"]
    assert qc["quant_method"] == "compressed-tensors"
    assert qc["format"] == "int-quantized"
    group = qc["config_groups"]["group_0"]
    assert group["weights"]["strategy"] == "channel"
    assert group["input_activations"] == {**group["input_activations"], "strategy": "token", "dynamic": True, "num_bits": 8}
    assert qc["ignore"] == ["lm_head", "model.layers.0.mlp.gate_proj"]

    files = {p.name for p in out.iterdir()}
    assert {"tokenizer.json", "generation_config.json", "LICENSE", "vllm_export.json"} <= files
    assert not any(p.suffix in {".py", ".pt", ".patch"} for p in out.iterdir())
    assert not any(p.name.startswith(".") for p in tmp_path.iterdir())


def test_untied_lm_head_is_kept(tmp_path):
    src = make_w8a8_artifact(tmp_path / "art", tied=False)
    export_mod.export(src, tmp_path / "out")
    names = {t.name for t in export_mod.read_safetensors_header(tmp_path / "out" / "model.safetensors")[0]}
    assert "lm_head.weight" in names


def test_scale_convention_confirmed_on_untransformed_module(tmp_path):
    pytest.importorskip("numpy")
    src = make_w8a8_artifact(tmp_path / "art")
    record = export_mod.export(src, tmp_path / "out", reference_dir=tmp_path / "ref")
    conv = record["scale_convention"]
    assert conv["verdict"] == "confirmed"
    assert conv["checked"] == 2
    assert conv["matched"] == 1


@pytest.mark.parametrize(
    ("qcfg", "reason"),
    [
        ({"a_bits": 4, "w_bits": 4, "format": "packed_int4"}, "activation_bits_unsupported"),
        ({"format": "packed_w8a8"}, "activation_scheme_unknown"),
        ({"act_quant": "per_tensor"}, "activation_scheme_unsupported"),
        ({"act_quant": "per_token", "weight_quant": "per_group"}, "weight_scheme_unsupported"),
    ],
)
def test_unsupported_quantization_configs(tmp_path, qcfg, reason):
    src = make_w8a8_artifact(tmp_path / "art", qcfg=qcfg)
    record = export_mod.export(src, tmp_path / "out")
    assert record["status"] == "unsupported"
    assert record["vllm_servable"] is False
    assert record["reason_code"] == reason
    assert not (tmp_path / "out").exists()


def test_pickle_only_artifact_is_unsupported_without_torch_load(tmp_path):
    src = tmp_path / "art"
    src.mkdir()
    (src / "config.json").write_text("{}")
    (src / "packed_w4a4.pt").write_bytes(b"\x80\x04")
    record = export_mod.export(src, tmp_path / "out")
    assert record["reason_code"] == "pickle_weights"


def test_unrecognized_integer_tensors_are_refused(tmp_path):
    src = tmp_path / "art"
    src.mkdir()
    (src / "config.json").write_text("{}")
    (src / "quantization_config.json").write_text(json.dumps({"act_quant": "per_token"}))
    write_safetensors(
        src / "model.safetensors",
        {
            "model.layers.0.mlp.down_proj.qweight": ("U8", [2, 2], [1, 2, 3, 4]),
            "model.layers.0.mlp.down_proj.scales": ("F16", [2], [0.1, 0.2]),
        },
    )
    record = export_mod.export(src, tmp_path / "out")
    assert record["reason_code"] == "unrecognized_quantized_tensors"


def test_scale_layout_mismatch_is_refused(tmp_path):
    src = tmp_path / "art"
    src.mkdir()
    (src / "config.json").write_text("{}")
    (src / "quantization_config.json").write_text(json.dumps({"act_quant": "per_token"}))
    write_safetensors(
        src / "model.safetensors",
        {
            "layer.weight": ("I8", [2, 4], [1] * 8),
            "layer.weight_scales": ("F16", [2, 2], [0.1] * 4),  # group-wise, not per-channel
        },
    )
    record = export_mod.export(src, tmp_path / "out")
    assert record["reason_code"] == "weight_scale_layout_unsupported"


def test_existing_quantization_config_is_left_alone(tmp_path):
    src = make_w8a8_artifact(tmp_path / "art")
    config = json.loads((src / "config.json").read_text())
    config["quantization_config"] = {"quant_method": "compressed-tensors"}
    (src / "config.json").write_text(json.dumps(config))
    record = export_mod.export(src, tmp_path / "out")
    assert record["reason_code"] == "already_quantized_config"


def test_export_refuses_to_overwrite_or_nest(tmp_path):
    src = make_w8a8_artifact(tmp_path / "art")
    with pytest.raises(ValueError):
        export_mod.export(src, src / "vllm")
    export_mod.export(src, tmp_path / "out")
    with pytest.raises(FileExistsError):
        export_mod.export(src, tmp_path / "out")
    assert export_mod.export(src, tmp_path / "out", force=True)["status"] == "exported"


def test_force_never_deletes_a_directory_holding_the_source(tmp_path):
    root = tmp_path / "quantized"
    src = make_w8a8_artifact(root / "slug-a")
    sibling = root / "other-artifact"
    sibling.mkdir()
    (sibling / "model.safetensors").write_bytes(b"keep")
    with pytest.raises(ValueError, match="contain"):
        export_mod.export(src, root, force=True)
    assert (src / "model.safetensors").is_file()
    assert (sibling / "model.safetensors").read_bytes() == b"keep"


@pytest.mark.parametrize("kind", ["foreign_dir", "file"])
def test_force_only_replaces_a_previous_export(tmp_path, kind):
    src = make_w8a8_artifact(tmp_path / "art")
    target = tmp_path / "out"
    if kind == "file":
        target.write_text("keep")
    else:
        target.mkdir()
        (target / "notes.txt").write_text("keep")
    with pytest.raises(ValueError, match="previous vLLM export"):
        export_mod.export(src, target, force=True)
    assert (target.read_text() if kind == "file" else (target / "notes.txt").read_text()) == "keep"


def test_tied_config_with_a_different_lm_head_is_refused(tmp_path):
    src = make_w8a8_artifact(tmp_path / "art", tied=False)
    config = json.loads((src / "config.json").read_text())
    config["tie_word_embeddings"] = True
    (src / "config.json").write_text(json.dumps(config))
    record = export_mod.export(src, tmp_path / "out")
    assert record["reason_code"] == "tied_lm_head_mismatch"
    assert not (tmp_path / "out").exists()


def _add_tensors(src: Path, extra: dict) -> None:
    """Rewrite the fixture's model.safetensors with extra tensors appended."""
    path = src / "model.safetensors"
    tensors, _ = export_mod.read_safetensors_header(path)
    original = {}
    with path.open("rb") as handle:
        for info in tensors:
            handle.seek(info.start)
            raw = handle.read(info.nbytes)
            fmt = _PACK[info.dtype]
            original[info.name] = (info.dtype, info.shape, list(struct.unpack(f"<{len(raw) // struct.calcsize(fmt)}{fmt}", raw)))
    write_safetensors(path, {**original, **extra})


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.0.self_attn.o_proj.weight_zero_point",
        "model.layers.0.self_attn.o_proj.input_scale",
        "model.layers.0.mlp.smooth_scale",
        "model.layers.0.online_hadamard.weight",
        "model.layers.0.self_attn.q_proj.weight_scale",  # second alias next to weight_scales
    ],
)
def test_leftover_float_quantization_tensors_are_refused(tmp_path, name):
    src = make_w8a8_artifact(tmp_path / "art")
    _add_tensors(src, {name: ("F16", [4], [1.0] * 4)})
    record = export_mod.export(src, tmp_path / "out")
    assert record["reason_code"] == "unrecognized_quantized_tensors"
    assert name in record["detail"]


@pytest.mark.parametrize(
    ("extra", "reason"),
    [
        ({"rotate": True}, "runtime_transform_unsupported"),
        ({"online_hadamard": True}, "runtime_transform_unsupported"),
        ({"w_sym": False}, "asymmetric_unsupported"),
        ({"act_quant": "per_token_asym"}, "asymmetric_unsupported"),
    ],
)
def test_runtime_transforms_and_asymmetric_configs_are_refused(tmp_path, extra, reason):
    qcfg = {"format": "packed_w8a8", "act_quant": "per_token", "weight_quant": "per_channel", **extra}
    src = make_w8a8_artifact(tmp_path / "art", qcfg=qcfg)
    assert export_mod.export(src, tmp_path / "out")["reason_code"] == reason


def test_int8_lm_head_is_refused(tmp_path):
    src = make_w8a8_artifact(tmp_path / "art", tied=False)
    _add_tensors(
        src,
        {
            "lm_head.weight": ("I8", [6, 4], [1] * 24),
            "lm_head.weight_scales": ("F16", [6], [0.1] * 6),
        },
    )
    record = export_mod.export(src, tmp_path / "out")
    assert record["reason_code"] == "quantized_embedding_unsupported"


def test_partly_quantized_fused_group_is_refused(tmp_path):
    src = make_w8a8_artifact(tmp_path / "art")
    _add_tensors(
        src,
        {
            "model.layers.0.self_attn.k_proj.weight": ("F16", [4, 4], [0.1] * 16),
            "model.layers.0.self_attn.v_proj.weight": ("I8", [4, 4], [1] * 16),
            "model.layers.0.self_attn.v_proj.weight_scales": ("F16", [4], [0.1] * 4),
        },
    )
    record = export_mod.export(src, tmp_path / "out")
    assert record["reason_code"] == "mixed_fused_group"
    assert "k_proj" in record["detail"]


def test_inference_adapter_is_noted(tmp_path):
    src = make_w8a8_artifact(tmp_path / "art")
    record = export_mod.export(src, tmp_path / "out")
    assert any("quant_agent_inference_adapter.py" in note for note in record["notes"])


def test_sharded_artifact_writes_index(tmp_path):
    src = make_w8a8_artifact(tmp_path / "art")
    write_safetensors(
        src / "model-00002-of-00002.safetensors",
        {
            "model.layers.1.mlp.up_proj.weight": ("I8", [2, 4], [1] * 8),
            "model.layers.1.mlp.up_proj.weight_scale": ("F16", [2, 1], [0.5, 0.5]),
        },
    )
    out = tmp_path / "out"
    record = export_mod.export(src, out)
    assert record["n_quantized_linears"] == 3
    index = json.loads((out / "model.safetensors.index.json").read_text())
    assert index["weight_map"]["model.layers.1.mlp.up_proj.weight_scale"] == "model-00002-of-00002.safetensors"
    assert index["weight_map"]["model.layers.0.self_attn.q_proj.weight_scale"] == "model.safetensors"

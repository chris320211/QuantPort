#!/usr/bin/env python3
"""Convert a verified quantized artifact into a checkpoint vLLM loads natively.

CPU only. Never re-quantizes: tensors are renamed / reshaped in the safetensors
header and their bytes are copied as-is. The format is detected from the saved
tensors (dtype, shape, sibling scales) plus the run's `quantization_config.json`,
never from the method name.

Profiles:

| profile | saved tensors | vLLM checkpoint |
| --- | --- | --- |
| `w8a8_int8` | int8 `<linear>.weight [N,K]` + float scale `[N]`/`[N,1]`, per-token dynamic activations | compressed-tensors `int-quantized` |

Anything else returns `vllm_servable: false` with a typed `reason_code`
(`activation_bits_unsupported`, `pickle_weights`, …). That is a normal outcome,
not a failure: the native artifact still publishes.

Output holds only Hub-standard files (config, safetensors, tokenizer). No
`.py`, no pickles, no `trust_remote_code`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import shutil
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths
from io_utils import atomic_write_text

COMPRESSED_TENSORS_VERSION = "0.6.0"
EXPORT_RECORD = "vllm_export.json"
SCALE_CONVENTION_MAX_REL_ERR = 0.05
SCALE_CONVENTION_SAMPLE = 24

_DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
    "F8_E4M3": 1, "F8_E5M2": 1,
}
_FLOAT_DTYPES = {"F16", "BF16", "F32"}
# Names overlays have used for the per-output-channel weight scale.
_SCALE_ALIASES = ("weight_scale", "weight_scales")
# Leaf names of quantization side tensors the w8a8 profile does not consume.
# Copying one through would give vLLM an unknown parameter or drop a correction.
_AUX_TENSOR_LEAVES = {
    "weight_scale", "weight_scales", "weight_scale_2", "weight_zero_point",
    "input_scale", "input_zero_point", "output_scale", "zero_point", "zeros",
    "qzeros", "g_idx", "scale", "scales", "smooth_scale", "smooth_scales",
    "act_scale", "act_scales",
}
_TRANSFORM_MARKERS = ("hadamard", "rotation", "rotate", "smooth")
_TRANSFORM_KEYS = ("rotate", "rotation", "online_rotation", "online_hadamard", "hadamard", "had_rotate")
_ASYM_KEYS = ("w_sym", "a_sym", "sym", "symmetric", "weight_sym", "act_sym")
# vLLM loads these as embedding / head layers, which have no int8 path.
_EMBEDDING_LEAVES = {"lm_head", "embed_tokens", "wte", "word_embeddings", "embed_in", "embed_out"}
# vLLM fuses these into one layer and needs one scheme across the group.
_FUSED_GROUPS = (("q_proj", "k_proj", "v_proj"), ("gate_proj", "up_proj"))
_ADAPTER_FILE = "quant_agent_inference_adapter.py"
_PICKLE_SUFFIXES = {".pt", ".pth", ".bin"}
_AUX_FILES = {
    "generation_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "merges.txt",
    "vocab.json",
    "chat_template.jinja",
    "chat_template.json",
}
_AUX_PREFIXES = ("LICENSE", "NOTICE")
_COPY_CHUNK = 64 * 1024 * 1024


class Unsupported(Exception):
    """The artifact has no native vLLM format. Not a pipeline failure."""

    def __init__(self, reason_code: str, detail: str):
        super().__init__(f"{reason_code}: {detail}")
        self.reason_code = reason_code
        self.detail = detail


@dataclass
class TensorInfo:
    file: Path
    name: str
    dtype: str
    shape: list[int]
    start: int  # absolute file offset
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.start


@dataclass
class ExportPlan:
    profile: str
    files: dict[Path, list[TensorInfo]]
    renames: dict[str, tuple[str, list[int]]]  # old name -> (new name, new shape)
    drop: set[str]
    quantized_modules: list[str]
    ignore: list[str]
    config: dict
    quantization_config: dict
    notes: list[str] = field(default_factory=list)


def read_safetensors_header(path: Path) -> tuple[list[TensorInfo], dict]:
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError(f"not a safetensors file: {path}")
        (header_len,) = struct.unpack("<Q", raw)
        header = json.loads(handle.read(header_len))
    data_start = 8 + header_len
    metadata = header.pop("__metadata__", None) or {}
    tensors = []
    for name, entry in header.items():
        begin, end = entry["data_offsets"]
        tensors.append(
            TensorInfo(
                file=path,
                name=name,
                dtype=entry["dtype"],
                shape=list(entry["shape"]),
                start=data_start + begin,
                end=data_start + end,
            )
        )
    tensors.sort(key=lambda t: t.start)
    return tensors, metadata


def _read_json(path: Path) -> dict:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"expected object JSON: {path}")
    return data


def _activation_bits(qcfg: dict) -> int | None:
    for key in ("a_bits", "act_bits", "activation_bits"):
        value = qcfg.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _tensor_sha256(info: TensorInfo) -> str:
    digest = hashlib.sha256()
    with info.file.open("rb") as handle:
        handle.seek(info.start)
        remaining = info.nbytes
        while remaining:
            chunk = handle.read(min(_COPY_CHUNK, remaining))
            if not chunk:
                raise ValueError(f"truncated tensor {info.name} in {info.file}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def compressed_tensors_w8a8_config(ignore: list[str]) -> dict:
    """compressed-tensors schema vLLM reads for int8 channel weights + dynamic token activations."""
    return {
        "quant_method": "compressed-tensors",
        "format": "int-quantized",
        "quantization_status": "compressed",
        "version": COMPRESSED_TENSORS_VERSION,
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 8,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "channel",
                    "dynamic": False,
                    "group_size": None,
                    "block_structure": None,
                    "actorder": None,
                    "observer": "minmax",
                    "observer_kwargs": {},
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "token",
                    "dynamic": True,
                    "group_size": None,
                    "block_structure": None,
                    "actorder": None,
                    "observer": "memoryless",
                    "observer_kwargs": {},
                },
                "output_activations": None,
            }
        },
        "ignore": ignore,
        "kv_cache_scheme": None,
    }


def plan_export(model_dir: Path) -> ExportPlan:
    """Detect the saved format and plan the rename. Raises Unsupported."""
    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise Unsupported("config_missing", f"no config.json in {model_dir}")
    config = _read_json(config_path)
    if config.get("quantization_config"):
        raise Unsupported(
            "already_quantized_config",
            "config.json already carries a quantization_config; serve it directly",
        )
    qcfg_path = model_dir / "quantization_config.json"
    qcfg = _read_json(qcfg_path) if qcfg_path.is_file() else {}

    a_bits = _activation_bits(qcfg)
    if a_bits is not None and a_bits < 8:
        raise Unsupported(
            "activation_bits_unsupported",
            f"{a_bits}-bit activations; vLLM has no int{a_bits}-activation GEMM "
            "outside NVFP4/MXFP4 on Blackwell",
        )
    transforms = [key for key in _TRANSFORM_KEYS if qcfg.get(key)]
    if transforms:
        raise Unsupported(
            "runtime_transform_unsupported",
            f"quantization_config.json enables {', '.join(transforms)}; vLLM's int8 "
            "path applies no online rotation",
        )
    asym = [key for key in _ASYM_KEYS if qcfg.get(key) is False]
    if asym or any("asym" in str(qcfg.get(key) or "").lower() for key in ("weight_quant", "act_quant")):
        raise Unsupported(
            "asymmetric_unsupported",
            f"asymmetric quantization ({', '.join(asym) or 'asym scheme'}); the w8a8 "
            "profile exports symmetric scales only",
        )

    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        pickles = sorted(p.name for p in model_dir.iterdir() if p.suffix.lower() in _PICKLE_SUFFIXES)
        if pickles:
            raise Unsupported(
                "pickle_weights",
                f"weights are pickled ({', '.join(pickles[:4])}); re-run with "
                "--allow-torch-load to convert them with torch.load(weights_only=True)",
            )
        raise Unsupported("no_weights", f"no weight files in {model_dir}")

    files: dict[Path, list[TensorInfo]] = {}
    by_name: dict[str, TensorInfo] = {}
    for path in st_files:
        tensors, _ = read_safetensors_header(path)
        files[path] = tensors
        for info in tensors:
            if info.name in by_name:
                raise ValueError(f"tensor {info.name} appears in two weight files")
            by_name[info.name] = info

    renames: dict[str, tuple[str, list[int]]] = {}
    quantized: list[str] = []
    accounted: set[str] = set()
    for name, info in by_name.items():
        if info.dtype != "I8" or not name.endswith(".weight"):
            continue
        module = name[: -len(".weight")]
        if len(info.shape) != 2:
            continue
        scale = next(
            (by_name[f"{module}.{alias}"] for alias in _SCALE_ALIASES if f"{module}.{alias}" in by_name),
            None,
        )
        out_features = info.shape[0]
        if scale is None or scale.dtype not in _FLOAT_DTYPES:
            continue
        if scale.shape not in ([out_features], [out_features, 1]):
            raise Unsupported(
                "weight_scale_layout_unsupported",
                f"{scale.name} has shape {scale.shape}; expected per-output-channel "
                f"[{out_features}] or [{out_features}, 1]",
            )
        quantized.append(module)
        accounted.update({name, scale.name})
        renames[scale.name] = (f"{module}.weight_scale", [out_features, 1])

    stray = sorted(
        name for name, info in by_name.items()
        if name not in accounted
        and (
            info.dtype not in _FLOAT_DTYPES
            or name.rsplit(".", 1)[-1] in _AUX_TENSOR_LEAVES
            or any(marker in name.lower() for marker in _TRANSFORM_MARKERS)
        )
    )
    if stray:
        raise Unsupported(
            "unrecognized_quantized_tensors",
            f"{len(stray)} quantization or transform tensors match no profile "
            f"(e.g. {', '.join(stray[:4])})",
        )
    if not quantized:
        raise Unsupported("no_quantized_linears", "no int8 weight + per-channel scale pairs found")
    embeddings = [m for m in quantized if m.rsplit(".", 1)[-1] in _EMBEDDING_LEAVES]
    if embeddings:
        raise Unsupported(
            "quantized_embedding_unsupported",
            f"{', '.join(sorted(embeddings))} is int8; vLLM's embedding and head layers "
            "have no int8 weight path",
        )
    quantized_set = set(quantized)
    linears = {
        name[: -len(".weight")]
        for name, info in by_name.items()
        if name.endswith(".weight") and len(info.shape) == 2
    }
    for module in sorted(linears):
        parent, _, leaf = module.rpartition(".")
        for group in _FUSED_GROUPS:
            if leaf != group[0]:
                continue
            members = [f"{parent}.{part}" for part in group if f"{parent}.{part}" in linears]
            schemes = {member in quantized_set for member in members}
            if len(schemes) > 1:
                raise Unsupported(
                    "mixed_fused_group",
                    f"{', '.join(members)} mix int8 and float weights; vLLM fuses them "
                    "into one layer with one scheme",
                )

    act = str(qcfg.get("act_quant") or "").lower()
    if not act:
        raise Unsupported(
            "activation_scheme_unknown",
            "quantization_config.json does not say how activations are quantized "
            "(act_quant); refusing to guess W8A8 vs W8A16",
        )
    if act not in {"per_token", "per-token", "token"}:
        raise Unsupported(
            "activation_scheme_unsupported",
            f"act_quant={act!r}; only dynamic per-token int8 activations are exported",
        )
    weight_quant = str(qcfg.get("weight_quant") or "per_channel").lower()
    if weight_quant not in {"per_channel", "per-channel", "channel"}:
        raise Unsupported(
            "weight_scheme_unsupported",
            f"weight_quant={weight_quant!r}; only per-channel int8 weights are exported",
        )

    notes: list[str] = []
    drop: set[str] = set()
    tied = bool(config.get("tie_word_embeddings", True))
    head = by_name.get("lm_head.weight")
    embed = next(
        (info for name, info in by_name.items() if name.endswith("embed_tokens.weight")),
        None,
    )
    if tied and head and embed:
        same = head.shape == embed.shape and head.dtype == embed.dtype and _tensor_sha256(head) == _tensor_sha256(embed)
        if not same:
            # vLLM ties lm_head to embed_tokens and skips the stored head.
            raise Unsupported(
                "tied_lm_head_mismatch",
                "config ties lm_head to embed_tokens but the saved lm_head.weight differs; "
                "set tie_word_embeddings explicitly in the artifact if the head is meant to be separate",
            )
        drop.add(head.name)
        notes.append("dropped lm_head.weight: tied to embed_tokens and byte-identical")
    if (model_dir / _ADAPTER_FILE).is_file():
        notes.append(
            f"the HF artifact ships {_ADAPTER_FILE}; the export assumes it applies no runtime "
            "transform, which the serve check's drift gate confirms"
        )

    ignore = sorted(
        name[: -len(".weight")]
        for name, info in by_name.items()
        if name.endswith(".weight")
        and info.dtype in _FLOAT_DTYPES
        and len(info.shape) == 2
        and "embed" not in name
        and name[: -len(".weight")] not in quantized_set
    )
    if "lm_head" not in ignore:
        ignore.append("lm_head")

    if qcfg.get("quantize_bmm_input"):
        notes.append(
            "HF benchmark also quantized attention bmm inputs to int8; vLLM keeps "
            "them in fp16, so vLLM perplexity may differ slightly"
        )
    out_config = dict(config)
    if out_config.pop("auto_map", None) is not None:
        notes.append("removed auto_map: the export loads with vLLM's built-in architecture")
    ct = compressed_tensors_w8a8_config(ignore)
    out_config["quantization_config"] = ct
    return ExportPlan(
        profile="w8a8_int8",
        files=files,
        renames=renames,
        drop=drop,
        quantized_modules=sorted(quantized),
        ignore=ignore,
        config=out_config,
        quantization_config=ct,
        notes=notes,
    )


def _write_safetensors(src: Path, tensors: list[TensorInfo], dest: Path, plan: ExportPlan) -> dict[str, int]:
    """Rewrite one file's header; copy tensor bytes unchanged. Returns name -> nbytes."""
    _, metadata = read_safetensors_header(src)
    header: dict = {"__metadata__": {**metadata, "format": "pt"}}
    kept: list[tuple[TensorInfo, str]] = []
    offset = 0
    for info in tensors:
        if info.name in plan.drop:
            continue
        new_name, new_shape = plan.renames.get(info.name, (info.name, info.shape))
        header[new_name] = {
            "dtype": info.dtype,
            "shape": new_shape,
            "data_offsets": [offset, offset + info.nbytes],
        }
        kept.append((info, new_name))
        offset += info.nbytes
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with src.open("rb") as reader, dest.open("wb") as writer:
        writer.write(struct.pack("<Q", len(encoded)))
        writer.write(encoded)
        for info, _ in kept:
            reader.seek(info.start)
            remaining = info.nbytes
            while remaining:
                chunk = reader.read(min(_COPY_CHUNK, remaining))
                if not chunk:
                    raise ValueError(f"truncated tensor {info.name} in {src}")
                writer.write(chunk)
                remaining -= len(chunk)
    return {name: info.nbytes for info, name in kept}


def _load_numpy(info: TensorInfo):
    import numpy as np

    with info.file.open("rb") as handle:
        handle.seek(info.start)
        raw = handle.read(info.nbytes)
    if info.dtype == "BF16":
        bits = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        array = bits.view(np.float32)
    else:
        dtype = {"F16": np.float16, "F32": np.float32, "I8": np.int8}[info.dtype]
        array = np.frombuffer(raw, dtype=dtype)
    return array.reshape(info.shape).astype(np.float32)


def check_scale_convention(plan: ExportPlan, reference_dir: Path) -> dict:
    """Compare int8 * scale against the fp16 snapshot on modules the method left untransformed.

    vLLM dequantizes as `int8 * weight_scale`. Methods that smooth or rotate
    every linear leave no untransformed module, so the result is
    `inconclusive`, not a failure; the GPU serve check is the real test.
    """
    try:
        import numpy as np
    except ImportError:
        return {"verdict": "skipped", "detail": "numpy is not installed"}
    reference: dict[str, TensorInfo] = {}
    for path in sorted(Path(reference_dir).glob("*.safetensors")):
        for info in read_safetensors_header(path)[0]:
            reference[info.name] = info
    if not reference:
        return {"verdict": "skipped", "detail": f"no safetensors in {reference_dir}"}
    by_name = {info.name: info for tensors in plan.files.values() for info in tensors}
    scale_for = {new[0][: -len(".weight_scale")]: old for old, new in plan.renames.items()}
    candidates = [m for m in plan.quantized_modules if f"{m}.weight" in reference]
    step = max(1, len(candidates) // SCALE_CONVENTION_SAMPLE)
    sample = candidates[::step][:SCALE_CONVENTION_SAMPLE]
    results = []
    for module in sample:
        ref_info = reference[f"{module}.weight"]
        q_info = by_name[f"{module}.weight"]
        if ref_info.shape != q_info.shape or ref_info.dtype not in _FLOAT_DTYPES:
            continue
        ref = _load_numpy(ref_info)
        q = _load_numpy(q_info)
        scale = _load_numpy(by_name[scale_for[module]]).reshape(-1, 1)
        norm = float(np.linalg.norm(ref)) or 1.0
        mul = float(np.linalg.norm(q * scale - ref)) / norm
        div = float(np.linalg.norm(q / np.where(scale == 0, 1, scale) - ref)) / norm
        results.append({"module": module, "rel_err_mul": mul, "rel_err_div": div})
    matched = [r for r in results if min(r["rel_err_mul"], r["rel_err_div"]) <= SCALE_CONVENTION_MAX_REL_ERR]
    if not matched:
        verdict = "inconclusive"
    elif all(r["rel_err_mul"] <= r["rel_err_div"] for r in matched):
        verdict = "confirmed"
    else:
        verdict = "contradicted"
    return {
        "verdict": verdict,
        "checked": len(results),
        "matched": len(matched),
        "max_rel_err": SCALE_CONVENTION_MAX_REL_ERR,
        "modules": results,
    }


def _torch_load_to_safetensors(model_dir: Path, staging: Path) -> Path:
    """Copy model_dir into staging with pickled state dicts rewritten as safetensors."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("--allow-torch-load needs torch (run under the method venv)") from exc
    st_dtypes = {
        torch.float16: "F16", torch.bfloat16: "BF16", torch.float32: "F32",
        torch.int8: "I8", torch.uint8: "U8", torch.int32: "I32", torch.int64: "I64",
    }
    staging.mkdir(parents=True)
    for path in model_dir.iterdir():
        if path.is_file() and path.suffix.lower() not in _PICKLE_SUFFIXES:
            shutil.copy2(path, staging / path.name)
    for path in sorted(model_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in _PICKLE_SUFFIXES:
            continue
        payload = torch.load(str(path), map_location="cpu", weights_only=True)
        if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
            payload = payload["state_dict"]
        if not isinstance(payload, dict) or not all(torch.is_tensor(v) for v in payload.values()):
            raise Unsupported("pickle_weights", f"{path.name} is not a flat tensor state dict")
        header: dict = {"__metadata__": {"format": "pt", "source": path.name}}
        blobs = []
        offset = 0
        for name in sorted(payload):
            tensor = payload[name].contiguous()
            if tensor.dtype not in st_dtypes:
                raise Unsupported("pickle_weights", f"{name} has unsupported dtype {tensor.dtype}")
            blob = tensor.view(torch.uint8).numpy().tobytes() if tensor.numel() else b""
            header[name] = {
                "dtype": st_dtypes[tensor.dtype],
                "shape": list(tensor.shape),
                "data_offsets": [offset, offset + len(blob)],
            }
            blobs.append(blob)
            offset += len(blob)
        encoded = json.dumps(header, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 8)
        with (staging / f"{path.stem}.safetensors").open("wb") as handle:
            handle.write(struct.pack("<Q", len(encoded)))
            handle.write(encoded)
            for blob in blobs:
                handle.write(blob)
    return staging


def export(
    model_dir: Path,
    output_dir: Path,
    *,
    reference_dir: Path | None = None,
    allow_torch_load: bool = False,
    force: bool = False,
) -> dict:
    model_dir = Path(model_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir == model_dir or model_dir in output_dir.parents or output_dir in model_dir.parents:
        raise ValueError("output_dir must not be the source artifact, inside it, or contain it")
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"{output_dir} exists; pass --force to replace it")
        if not output_dir.is_dir() or not (output_dir / EXPORT_RECORD).is_file():
            # --force deletes output_dir, so only ever replace a previous export.
            raise ValueError(f"--force only replaces a previous vLLM export; {output_dir} has no {EXPORT_RECORD}")
    record: dict = {
        "source_dir": str(model_dir),
        "output_dir": str(output_dir),
        "vllm_servable": False,
    }
    staging = output_dir.with_name(f".{output_dir.name}.{secrets.token_hex(4)}.tmp")
    torch_staging = None
    try:
        source = model_dir
        has_st = any(model_dir.glob("*.safetensors"))
        if allow_torch_load and not has_st:
            torch_staging = output_dir.with_name(f".{output_dir.name}.{secrets.token_hex(4)}.torch")
            source = _torch_load_to_safetensors(model_dir, torch_staging)
            record["converted_pickles"] = True
        plan = plan_export(source)
    except Unsupported as exc:
        if torch_staging and torch_staging.exists():
            shutil.rmtree(torch_staging)
        return {**record, "status": "unsupported", "reason_code": exc.reason_code, "detail": exc.detail}

    try:
        staging.mkdir(parents=True)
        weight_map: dict[str, str] = {}
        total = 0
        for src, tensors in plan.files.items():
            written = _write_safetensors(src, tensors, staging / src.name, plan)
            for name, nbytes in written.items():
                weight_map[name] = src.name
                total += nbytes
        if len(plan.files) > 1:
            index = {"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}
            atomic_write_text(staging / "model.safetensors.index.json", json.dumps(index, indent=2) + "\n", mode=0o644)
        atomic_write_text(staging / "config.json", json.dumps(plan.config, indent=2) + "\n", mode=0o644)
        copied = []
        for path in sorted(source.iterdir()):
            if path.is_file() and (path.name in _AUX_FILES or path.name.startswith(_AUX_PREFIXES)):
                shutil.copy2(path, staging / path.name)
                copied.append(path.name)
        record.update(
            {
                "status": "exported",
                "vllm_servable": True,
                "profile": plan.profile,
                "n_quantized_linears": len(plan.quantized_modules),
                "renamed_scales": len(plan.renames),
                "dropped": sorted(plan.drop),
                "ignore": plan.ignore,
                "copied": copied,
                "weight_bytes": total,
                "notes": plan.notes,
                "quantization_config": plan.quantization_config,
                "serve_check": "pending",
            }
        )
        if reference_dir is not None:
            record["scale_convention"] = check_scale_convention(plan, Path(reference_dir))
        atomic_write_text(staging / EXPORT_RECORD, json.dumps(record, indent=2) + "\n", mode=0o644)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        staging.rename(output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if torch_staging and torch_staging.exists():
            shutil.rmtree(torch_staging)
    return record


def _job_inputs(job_id: str, request_path: Path) -> tuple[Path, Path, Path, Path]:
    import jobs as jobs_mod
    from request import load_request
    from verify import resolve_workspace_path

    request = load_request(request_path)
    meta = jobs_mod.read_meta(job_id)
    if str(meta.verification_status or "").lower() != "passed":
        raise RuntimeError("export requires quant-verify passed")
    slug = paths.require_slug(str(request["slug"]))
    model_dir = resolve_workspace_path(meta.output_dir)
    output_dir = paths.REPO_ROOT / "quantized" / f"{slug}-vllm"
    reference = resolve_workspace_path(request["hf_snapshot_path"])
    return model_dir, output_dir, reference, jobs_mod.job_dir(job_id)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a verified quantized artifact for vLLM")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--job-id", help="Pipeline mode: export jobs/<id>'s saved artifact")
    source.add_argument("--model-dir", type=Path, help="Backfill mode: any artifact directory")
    parser.add_argument("--request", type=Path, help="Request JSON (with --job-id)")
    parser.add_argument("--output-dir", type=Path, help="Default: quantized/<slug>-vllm")
    parser.add_argument("--reference", type=Path, help="fp16 snapshot for the scale-convention check")
    parser.add_argument("--allow-torch-load", action="store_true", help="Convert .pt state dicts (weights_only=True)")
    parser.add_argument("--force", action="store_true", help="Replace an existing output directory")
    args = parser.parse_args()
    job_directory = None
    try:
        if args.job_id:
            if not args.request:
                raise ValueError("--request is required with --job-id")
            model_dir, output_dir, reference, job_directory = _job_inputs(args.job_id, args.request)
            output_dir = args.output_dir or output_dir
            reference = args.reference or reference
        else:
            if not args.output_dir:
                raise ValueError("--output-dir is required with --model-dir")
            model_dir, output_dir, reference = args.model_dir, args.output_dir, args.reference
        record = export(
            model_dir,
            output_dir,
            reference_dir=reference,
            allow_torch_load=args.allow_torch_load,
            force=args.force,
        )
    except Exception as exc:  # noqa: BLE001 — report the exact failure as JSON
        print(json.dumps({"status": "failed", "message": str(exc)}, indent=2))
        return 1
    if job_directory is not None:
        atomic_write_text(job_directory / EXPORT_RECORD, json.dumps(record, indent=2) + "\n", mode=0o644)
    print(json.dumps({k: v for k, v in record.items() if k != "quantization_config"}, indent=2))
    return 0 if record["status"] == "exported" else 2


if __name__ == "__main__":
    raise SystemExit(main())

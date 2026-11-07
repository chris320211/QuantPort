#!/usr/bin/env python
# QUANT_AGENT_OVERLAY_DIR=<overlay>
"""DuQuant W4A4 (+4-bit Q/K/V) for Qwen/Qwen3-8B on integers (plan kernel_triton, format duquant_w4a4_int8_storage).

Same quantization as the verified role_driver_qwen3 job 20261107T043051Z-ce3324 (run.sh recipe: block 128,
256 rotation steps, epochs 0, alpha 0.6 smooth, lac 0.9, swc 0.8, seed 2; PPL 10.93, fp16 fake-quant,
1856 tok/s, 18.5 GB). This wrapper writes the integer artifact:

1. Source = that verified fake-quant artifact. It lives in OUTPUT_DIR after the role_driver job; the first
   run MOVES it (same filesystem rename, nothing deleted) to SOURCE_DIR and converts from there, so it stays
   available. Later runs reuse SOURCE_DIR. If neither holds a matching fake-quant artifact, the role_driver
   pipeline (duquant_port.driver, ~3 min) quantizes again into SOURCE_DIR first. The smoke run always
   re-quantizes the 2-layer slice into a temporary fake-quant directory.
2. duquant_port.int8_runtime.convert_artifact: every decoder projection -> int8 centred W4 codes (q - z,
   integer zero points) + fp32 per-channel scale, checking that every code reproduces its fp16 fake-quant
   weight bit for bit; embeddings / lm_head / norms fp16; duquant_runtime.safetensors (R stacks,
   permutations, GQA o_proj smooth) and tokenizer/config copied.
3. Reload through quant_agent_inference_adapter.py (meta-device build, int8 weights straight to the GPU,
   Triton per-token asym A4 codes + Triton int8 GEMM with an exact zero-point epilogue, Triton Q/K/V
   per-(token, head) A4, SDPA flash attention), compare logits with the fake-quant source on a held-out
   WikiText-2 train window, greedy generate, and write quantization_config.json (format
   duquant_w4a4_int8_storage, quant_method duquant, w/a 4 bit, per_channel / per_token, asymmetric,
   online_transforms) with the conversion and reload-check results.
"""
import gc
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import torch

MODEL_ID = "Qwen/Qwen3-8B"
OUTPUT_DIR = "./quantized/duquant-qwen3-8b-vastrtx3090"
# verified fake-quant artifact of job 20261107T043051Z-ce3324 (moved here from OUTPUT_DIR, never deleted)
SOURCE_DIR = "./quantized/duquant-qwen3-8b-vastrtx3090-fq"
REPO_URL = "https://github.com/Hsu1023/DuQuant"
REPO_COMMIT = "d56cfc6fe97c34c0eb100fec82fe439865905679"
WIKITEXT_CACHE_REL = ".cache/eval/hf"  # cached Salesforce/wikitext parquet; falls back to the Hub

# run.sh: --block_size 128 --max_rotation_step 256 --epochs 0 --wbits 4 --abits 4 --lwc --alpha 0.6
# --smooth --lac 0.9 --swc 0.8 (main.py defaults: nsamples 128, seqlen 2048, seed 2, permutation_times 1).
SETTINGS = {
    "wbits": 4, "abits": 4, "block_size": 128, "max_rotation_step": 256, "permutation_times": 1,
    "epochs": 0, "lwc": True, "swc": 0.8, "lac": 0.9, "alpha": 0.6, "smooth": True,
    "nsamples": 128, "seqlen": 2048, "seed": 2, "calib_dataset": "wikitext2",
}
SOURCE_MATCH = {"quant_method": "duquant", "format": "duquant_fakequant", "w_bits": 4, "a_bits": 4,
                "block_size": 128, "max_rotation_step": 256, "permutation_times": 1, "alpha": 0.6, "lac": 0.9,
                "swc": 0.8, "epochs": 0, "base_model": MODEL_ID, "smoke": False}
PROBE_SEQLEN = 256
# an integer-path probe loss this far above the fake-quant one means a kernel bug, not rounding noise
MAX_PROBE_LOSS_GAP = 0.25


def _env_int(name):
    raw = os.environ.get(name)
    return int(raw) if raw else None


def _data_dirs():
    dirs = [Path.cwd() / WIKITEXT_CACHE_REL]
    overlay = os.environ.get("QUANT_AGENT_OVERLAY_DIR")
    if overlay:
        parents = Path(overlay).resolve().parents  # out/overlays/<slug>/<plan>/<hash> -> workspace
        if len(parents) > 4:
            dirs.append(parents[4] / WIKITEXT_CACHE_REL)
    return [str(d) for d in dirs]


def _read_qcfg(path):
    cfg = Path(path) / "quantization_config.json"
    if not cfg.is_file():
        return None
    try:
        return json.loads(cfg.read_text())
    except ValueError:
        return None


def _is_source(path):
    """A complete fake-quant artifact made with these settings (not a smoke slice)."""
    qcfg = _read_qcfg(path)
    if qcfg is None or not (Path(path) / "duquant_runtime.safetensors").is_file():
        return False
    if not (Path(path) / "model.safetensors.index.json").is_file():
        return False
    diff = {k: qcfg.get(k) for k, v in SOURCE_MATCH.items() if qcfg.get(k) != v}
    if diff:
        print(f"{path}: not a matching fake-quant source ({diff})", flush=True)
    return not diff


def _quantize_fake(model_src, fake_dir, smoke):
    from duquant_port import driver

    settings = dict(SETTINGS)
    nsamples = _env_int("QUANT_AGENT_SMOKE_CALIB_SAMPLES")
    seqlen = _env_int("QUANT_AGENT_SMOKE_SEQ_LEN")
    if nsamples:
        settings["nsamples"] = nsamples
    if seqlen:
        settings["seqlen"] = seqlen
    print(f"quantizing {model_src} -> {fake_dir} (role_driver pipeline) settings={json.dumps(settings)}", flush=True)
    driver.quantize(
        model_src, str(fake_dir), settings, data_dirs=_data_dirs(),
        extra_config={"base_model": MODEL_ID, "repo_url": REPO_URL, "repo_commit": REPO_COMMIT, "smoke": smoke},
    )
    gc.collect()
    torch.cuda.empty_cache()


def _ensure_source():
    """Verified fake-quant artifact at SOURCE_DIR (moved from OUTPUT_DIR, or re-quantized); never deletes one."""
    source, canonical = Path(SOURCE_DIR), Path(OUTPUT_DIR)
    if _is_source(canonical):
        if source.exists():
            # OUTPUT_DIR holds a newer fake-quant run: keep the older SOURCE_DIR under a dated name
            aside = Path(f"{SOURCE_DIR}-{time.strftime('%Y%m%dT%H%M%S', time.gmtime(source.stat().st_mtime))}")
            print(f"keeping previous fake-quant source as {aside}", flush=True)
            os.rename(source, aside)
        source.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(canonical), str(source))  # same filesystem: an atomic rename
        print(f"moved verified fake-quant artifact {canonical} -> {source}", flush=True)
        return source, "moved from OUTPUT_DIR"
    if _is_source(source):
        return source, "reused SOURCE_DIR"
    if source.exists():
        aside = Path(f"{SOURCE_DIR}-incomplete-{int(time.time())}")
        os.rename(source, aside)
        print(f"{source} is not a usable fake-quant source; set aside as {aside}", flush=True)
    tmp = Path(tempfile.mkdtemp(prefix=".duquant-fq-", dir=str(source.parent.resolve())))
    _quantize_fake(MODEL_ID, tmp, smoke=False)
    os.rename(tmp, source)
    return source, "re-quantized (role_driver pipeline)"


def _adapter():
    import importlib.util

    adapter_path = Path(os.environ["QUANT_AGENT_METHOD_REPO"]) / "quant_agent_inference_adapter.py"
    spec = importlib.util.spec_from_file_location("quant_agent_inference_adapter", adapter_path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    return adapter


def _probe_ids(model_dir):
    from transformers import AutoTokenizer

    from duquant_port.data import get_wikitext2_calib

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    seqlen = min(PROBE_SEQLEN, _env_int("QUANT_AGENT_SMOKE_SEQ_LEN") or PROBE_SEQLEN)
    # a different seed than calibration (seed 2): a held-out training window
    return get_wikitext2_calib(tokenizer, 1, seqlen, int(SETTINGS["seed"]) + 1, _data_dirs())


def _forward(model_dir, ids, generate):
    adapter = _adapter()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer = adapter.load_model_and_tokenizer(
        model_path=str(model_dir), model_id=MODEL_ID, dtype="float16", device="cuda", trust_remote_code=False)
    torch.cuda.synchronize()
    out = {"load_peak_vram_gb": torch.cuda.max_memory_allocated() / 2 ** 30}
    ids = ids.to("cuda")
    with torch.inference_mode():
        res = model(input_ids=ids, labels=ids)
        out["loss"] = float(res.loss.float().item())
        out["logits"] = res.logits[0].float().cpu()
        if generate:
            prompt = ids[:, :32]
            gen = model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt), max_new_tokens=12,
                                 do_sample=False, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            if not torch.equal(gen[:, : prompt.shape[1]], prompt):
                raise RuntimeError("generate output does not start with the caller's input_ids")
            out["generated"] = tokenizer.decode(gen[0, prompt.shape[1]:], skip_special_tokens=True)
    out["runtime"] = getattr(model, "quant_agent_runtime", None)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out


def _weight_bytes(path):
    return sum(p.stat().st_size for p in Path(path).glob("model*.safetensors"))


def main():
    method_repo = os.environ.get("QUANT_AGENT_METHOD_REPO")
    if not method_repo:
        raise SystemExit("QUANT_AGENT_METHOD_REPO is not set; run through the launcher")
    print(f"QUANT_AGENT_OVERLAY_DIR={os.environ.get('QUANT_AGENT_OVERLAY_DIR')}", flush=True)
    # the method repo first: its `quantize` package must win over this wrapper's own module name
    sys.path.insert(0, method_repo)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required (Triton int8 GEMM)")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    smoke_model = os.environ.get("QUANT_AGENT_SMOKE_MODEL_DIR")
    output_dir = Path(os.environ.get("QUANT_AGENT_SMOKE_OUTPUT_DIR") or OUTPUT_DIR)
    smoke = os.environ.get("QUANT_AGENT_SMOKE") == "1" or bool(smoke_model)
    started = time.time()

    from duquant_port import int8_runtime

    temp_fake = None
    if smoke:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        temp_fake = Path(tempfile.mkdtemp(prefix=".duquant-fq-smoke-", dir=str(output_dir.parent.resolve())))
        _quantize_fake(smoke_model or MODEL_ID, temp_fake, smoke=True)
        source, origin = temp_fake, "re-quantized smoke slice (temporary)"
    else:
        source, origin = _ensure_source()
    print(f"source fake-quant artifact: {source} ({origin})", flush=True)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-int8-", dir=str(output_dir.parent.resolve())))
    os.chmod(staging, 0o755)  # mkdtemp is 0700; the published artifact dir should not be
    conversion = int8_runtime.convert_artifact(str(source), str(staging))

    probe = _probe_ids(source)
    fake = _forward(source, probe, generate=False)
    real = _forward(staging, probe, generate=True)
    a, b = real.pop("logits"), fake.pop("logits")
    check = {
        "probe_tokens": int(probe.shape[-1]),
        "probe_loss": real["loss"],
        "fake_quant_probe_loss": fake["loss"],
        "finite": bool(torch.isfinite(a).all()) and bool(torch.isfinite(torch.tensor(real["loss"]))),
        "rel_logit_error_vs_fake_quant": ((a - b).norm() / b.norm()).item(),
        "top1_agreement_vs_fake_quant": (a.argmax(-1) == b.argmax(-1)).float().mean().item(),
        "generated": real.get("generated"),
        "load_peak_vram_gb": real["load_peak_vram_gb"],
        "fake_quant_load_peak_vram_gb": fake["load_peak_vram_gb"],
        "runtime": real.get("runtime"),
    }
    print("reload_check " + json.dumps(check), flush=True)
    if not check["finite"]:
        raise SystemExit("integer artifact produced non-finite logits")
    if real["loss"] > fake["loss"] + MAX_PROBE_LOSS_GAP:
        raise SystemExit(f"integer probe loss {real['loss']:.3f} far above fake-quant {fake['loss']:.3f}")

    qcfg = int8_runtime.read_config(staging)
    qcfg["source_artifact"] = str(source) if temp_fake is None else origin
    qcfg["source_origin"] = origin
    qcfg["weight_file_bytes"] = _weight_bytes(staging)
    qcfg["source_weight_file_bytes"] = _weight_bytes(source)
    qcfg["conversion"] = conversion
    qcfg["reload_check"] = check
    qcfg["smoke"] = smoke
    qcfg["artifact"] = ("int8 centred W4 codes + fp32 per-channel scales for every decoder projection, fp16 "
                        "embeddings/lm_head/norms, duquant_runtime.safetensors (R1/R2, permutations, GQA o_proj "
                        "smooth); reload with quant_agent_inference_adapter.py (duquant_port.int8_runtime)")
    with open(staging / "quantization_config.json", "w") as handle:
        json.dump(qcfg, handle, indent=2, default=str)

    if output_dir.exists():
        if _read_qcfg(output_dir) and (_read_qcfg(output_dir) or {}).get("format") == "duquant_fakequant":
            raise SystemExit(f"{output_dir} still holds a fake-quant artifact; refusing to replace it")
        shutil.rmtree(output_dir)  # an earlier integer artifact (or the smoke scratch dir)
    os.rename(staging, output_dir)
    if temp_fake is not None:
        shutil.rmtree(temp_fake, ignore_errors=True)
    print(f"done in {time.time() - started:.1f}s -> {output_dir}; weights {qcfg['weight_file_bytes'] / 1e9:.2f} GB "
          f"(fake-quant {qcfg['source_weight_file_bytes'] / 1e9:.2f} GB); files: {sorted(os.listdir(output_dir))}",
          flush=True)


if __name__ == "__main__":
    main()

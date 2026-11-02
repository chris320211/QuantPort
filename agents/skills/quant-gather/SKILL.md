---
name: quant-gather
description: >-
  Find a quantization method’s paper and GitHub from the method name, clone the
  repo, snapshot the named model’s weights (typically Hub org/name), record GPU
  facts, and install one method venv. Use only as a quant-gather subagent
  launched by the parent quant skill. No method catalog. Does not adapt or
  launch GPU jobs.
disable-model-invocation: true
---

# Quant Gather

You are a **subagent**. Do only gather. Read
[_shared/pipeline_contract.md](../_shared/pipeline_contract.md).

`PY=$([ -x .venv/bin/python ] && echo .venv/bin/python || command -v python || command -v python3)`
`S="$PY agents/skills/_shared/scripts"`

Host-mutating steps need `--allow-unsafe-host-execution` on an isolated box.

Every new shell: if `HF_TOKEN` is unset and `.env` exists,
`source agents/skills/_shared/load_env.sh .env`. Never read, print, or paste
values. Do not ask the parent or user mid-stage; finish or fail with the
request JSON / error.

## Do

1. Web-search `method_name` for the official paper and official GitHub.
   Prefer the authors’ repo. GitHub URL must be `https://github.com/owner/repo`
   (no git@, no other hosts). If two repos look official, do **not** ask: use
   the GitHub URL named in the paper (arXiv / PDF “code is available at”),
   else the repo whose name matches the method, else the paper authors’
   first-party org. Never a third-party reimplementation when that URL is known.
2. Download the paper:

   ```bash
   $S/paper.py --arxiv-id <id>
   $S/refuse.py .cache/papers/<id>.txt
   ```

   Stop if refuse returns `qat_training_only`, `kv_cache_only`, or `bitnet_1_58`.
3. Clone into `.venvs/<slug>/repo`:

   ```bash
   $S/clone.py --slug <slug> --repo-url https://github.com/owner/repo \
     --allow-unsafe-host-execution
   ```

4. Snapshot the named model (uses `HF_TOKEN` from the environment when the
   download is gated or Hub-hosted; never print it):

   ```bash
   $S/snapshot.py --model-id <org/model>
   ```

5. GPU facts:

   ```bash
   $S/gpu.py --gpu-instance <instance>
   ```

6. **One** venv install for this slug (before any port worker):

   ```bash
   $S/install_venv.py --slug <slug> --allow-unsafe-host-execution \
     --step 'pip install -e .'
   ```

   Add extra `--step` values from the repo README (python/pip only; no shell
   operators, and `<`/`>` count as operators, so pin with `==`). Torch pins match this host's **nvcc/toolkit** (not nvidia-smi
   driver CUDA) and are re-applied after `pip install -r` and before local
   `pip install -e .`, so method requirements cannot replace them with a
   CUDA-mismatched wheel. Local editable installs get `--no-build-isolation`
   so `setup.py` can import that torch. PyPI CUDA wheels like flash-attn still
   need an explicit `--no-build-isolation` extra step when the README says so.

7. Write the handoff:

   ```bash
   $S/request.py --write-json '{"method_name":"...","model_id":"...","gpu_instance":"...","arxiv_id":"...","paper_path":"...","repo_url":"...","repo_path":"...","repo_commit":"...","hf_snapshot_path":"...","slug":"..."}'
   ```

   Include optional GPU fields from `gpu.py` when present.
   `request.py` seeds `retry_gpu_jobs_used=0`, `retry_gpu_jobs_max=14`, and
   `tried_overlays=[]` so the parent retry loop works for this slug without
   method-specific JSON.

Slug: lowercase method + short model + instance, e.g. `awq-qwen25-05b-g5xlarge`.

## Do not

Adapt, write overlays, launch jobs, rewrite kernels, or read `.env`.

## Return

The request JSON path only.

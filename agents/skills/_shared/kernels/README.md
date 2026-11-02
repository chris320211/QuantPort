# Shared kernels

Kernels that already made a quantized artifact fast, kept so the next run
reuses them instead of writing new ones. `quant-kernel` checks here first.

Entries are keyed by the **saved format** an artifact uses (its
`quantization_config.json`), never by model or GPU name. A kernel written for
one model serves every model saved in the same format.

```bash
$S/kernels.py list
$S/kernels.py match --job-id <job_id> [--compute-capability 8.6]
$S/kernels.py check
```

`match` reads the job's `quantization_config.json` and returns every entry
whose `match` block it satisfies, with `retune: true` when the GPU's compute
capability is not in the entry's `tuned_on`.

## Layout

One folder per entry, holding a `kernel.json` and, for `kind: module`, the
kernel source:

| key | meaning |
| --- | --- |
| `name` | Unique id |
| `kind` | `module` (self-contained file in this folder) or `port` (pieces inside a stored library port's `overlay.patch`) |
| `module` / `entry_points` | `module` only: the file and the functions an adapter calls |
| `port` | `port` only: the `library/ports/...` folder |
| `match` | `quantization_config.json` key → allowed values; every key must match |
| `requires` | Runtime dependencies |
| `min_compute_capability` | Oldest GPU it runs on |
| `tuned_on` | GPUs whose tile choices it was tuned on (`gpu`, `compute_capability`, `torch`, `triton`) |
| `results` | Benchmarks behind it, with the job and port they came from |
| `notes` | What it replaced and its known limits |

## Rules

- Reuse a `module` entry by copying its file into the overlay's adapter (or
  importing it by path) and calling its entry points. Do not edit it in place
  for one model; a change that helps every user of the format is a change to
  this folder, with tests.
- On a GPU whose compute capability is not in `tuned_on`, retune before
  benchmarking (`tune_config` for `w8a8_int8`) and add the GPU to `tuned_on`
  when the run is published.
- A new kernel joins the library only after its job verified, benchmarked
  faster or smaller than the run it replaced, and was published: add a folder
  with `kernel.json`, then run `$S/kernels.py check`.

# Ports

The patch behind each library row: what it took to run a method on a model it
did not support. One folder per row, named like its contribution or attempt
JSON (`<model>__<gpu>__<method>`); attempts live under `attempts/` so a failed
re-run never replaces the port behind published weights. Linked from the **Port** column of
[LIBRARY.md](../LIBRARY.md) and [ATTEMPTS.md](../ATTEMPTS.md).

| File | What |
| --- | --- |
| `overlay.patch` | Git diff against the method repo at `repo_commit` (the overlay's `base_commit`). Stored byte-for-byte. |
| `manifest.json` | Rationale, target modules, `base_commit`, `patch_sha256`. |
| `quantize.py` | The wrapper the job ran: calibration data, settings, how the artifact is saved. |
| `port.json` | Provenance: job id, method repo and commit, file list. |

For a library row this is the port that produced the published weights. For an
attempt it is the best job's port, which may differ from the last job's.

## Check and replay

```bash
sha256sum overlay.patch                       # matches patch_sha256 in manifest.json
git clone <repo_url> repo && git -C repo checkout <repo_commit>
git -C repo apply --check ../overlay.patch    # the same check quant-port runs
```

`quantize.py` is a reference copy, not a standalone script: host paths are
replaced by `<repo>` and `<overlay>`, and `launch.py` supplies
`QUANT_AGENT_OVERLAY_DIR` and `QUANT_AGENT_METHOD_REPO` when it runs the job.

## How they get here

`library.py --catalog` and `--record-attempt` copy the port from
`jobs/<job_id>/overlay` and `jobs/<job_id>/script.py` while those still exist.
A port containing a token-shaped string, or missing its snapshot, is skipped
and the row is recorded without one (re-recording the same job keeps the port
already stored). The port is written only after the row validates. For a row recorded earlier, run on the
host that still has the job:

```bash
$S/library.py --export-port --job-id <job_id> --request out/requests/<slug>.json
```

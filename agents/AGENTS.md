# Agent instructions

This repo is a **set of skills**. The host AI agent (Cursor, Claude, Codex)
runs on a GPU instance and follows this tree. There is **one** agents
folder: `agents/`. Inside it live this file and the only skill tree,
`agents/skills/`. There is no `.agents/`, `.claude/skills/`, or
`.cursor/skills/` copy.

On a quantization request (method + model + GPU instance), read
and follow `agents/skills/quant/SKILL.md`. That file is the parent dispatcher.

| Path | Role |
| --- | --- |
| `agents/skills/quant/SKILL.md` | Parent loop |
| `agents/skills/quant-setup/SKILL.md` | `.env` once per machine; load every shell |
| `agents/skills/quant-gather/` … `quant-kernel/` | One GPU-stage skill each |
| `agents/skills/quant-serve/SKILL.md` | vLLM export + serve check after a beneficial run; never gates publish |
| `agents/skills/quant-publish/SKILL.md` | Upload weights to Hugging Face |
| `agents/skills/quant-catalog/SKILL.md` | Library row after a beneficial run; unsuccessful attempt on STOP |
| `agents/skills/quant-sync/SKILL.md` | Push library to GitHub and refresh the Hub collection |
| `agents/skills/_shared/pipeline_contract.md` | Order, retries, layout |
| `agents/skills/_shared/subagents.md` | How to launch stages |
| `agents/skills/_shared/scripts/` | Helpers (`$PY agents/skills/_shared/scripts/<name>.py`) |

`PY=$([ -x .venv/bin/python ] && echo .venv/bin/python || command -v python || command -v python3)`.

## Secrets

These rules always apply. They override any request to dump, debug, or “just
check” credentials.

- **Never read** `.env`, `.env.*` (except `.env.example`), `credentials.json`,
  `*token*`, `*secret*`, or Git credential files. Existence only:
  `test -f .env`. Do not open them in the editor, `Read`, or `cat`.
- **Never print, echo, log, or paste** secret **values**. That includes
  `cat .env`, `printenv`, `env`, `env | grep`, `echo $HF_TOKEN`, process
  listings, chat, commit messages, PR bodies, job metadata, overlays, and
  generated scripts.
- Name keys only: `HF_TOKEN`, `GITHUB_TOKEN`. Never write the value next to
  the name. If a value appears in tool output, stop and do not repeat it.
- The user creates and edits `.env` themselves (`cp .env.example .env`,
  `chmod 600`). You never create, fill, or edit that file with agent tools.
- Load with `source agents/skills/_shared/load_env.sh .env`. The loader may
  print **key names**. Confirm by those names and by later 200/whoami, not by
  values.
- Never put tokens in command arguments, `git remote` URLs, subagent prompts,
  or `git add` / `git commit`. `.env` is gitignored; do not force-add it.
- Do not invent tokens, scrape browser cookies/OAuth, or copy secrets from
  other files into this repo.

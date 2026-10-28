#!/usr/bin/env bash
# Strictly parse model-quantization-agent credentials into the current shell.
#
# Usage:
#   source agents/skills/_shared/load_env.sh .env
#
# Behavior:
#   - Refuses a world-readable credential file (mode != 600); auto-tightens to 600.
#   - Parses an explicit key allowlist without evaluating shell syntax; other
#     keys are skipped (named on stderr) and never exported. Works in bash and zsh.
#   - Aliases HUGGINGFACE_HUB_TOKEN -> HF_TOKEN so libraries reading either name
#     see the same value.
#   - Never echoes secret values; only echoes key NAMES that were loaded.
#   - Returns non-zero on missing file so callers can `|| { ...; exit 1; }`.

_quant_load_env() {
  local env_file="${1:-${QUANT_AGENT_ENV:-.env}}"

  if [[ ! -f "$env_file" ]]; then
    echo "[env] $env_file not found." >&2
    echo "[env] Create a mode-600 .env yourself, or export HF_TOKEN in this shell." >&2
    echo "[env] Use the quant-setup skill for guided setup. Never paste tokens in chat." >&2
    return 1
  fi

  # Permission hardening — never source a file other users can read.
  local mode
  mode=$(stat -c %a "$env_file" 2>/dev/null || stat -f %A "$env_file" 2>/dev/null)
  if [[ -z "$mode" ]]; then
    echo "[env] could not stat $env_file" >&2
    return 1
  fi
  if [[ "$mode" != "600" ]]; then
    echo "[env] tightening $env_file from mode $mode to 600" >&2
    chmod 600 "$env_file" || {
      echo "[env] chmod 600 failed; refusing to source" >&2
      return 1
    }
  fi

  # Parse data; never `source` the file. Only the documented keys are exported,
  # so shell syntax and command substitutions remain inert text. Other keys
  # (e.g. credentials for other tools) are skipped by name, never exported.
  # Split with parameter expansion, not BASH_REMATCH, so zsh parses it too.
  # Accepted line shapes: KEY=value, export KEY=value, KEY = value,
  # KEY="value", KEY='value' (CRLF and leading spaces tolerated). Other lines
  # (multi-line values, other tools' formats) are skipped and reported by line
  # number only; they can never be an allowlisted key.
  local loaded_keys="" skipped_keys="" skipped_lines="" line key value lineno=0
  while IFS= read -r line || [[ -n "$line" ]]; do
    lineno=$((lineno + 1))
    line="${line%$'\r'}"
    line="${line#"${line%%[![:space:]]*}"}"
    [[ -z "$line" || "$line" == \#* ]] && continue
    [[ "$line" == "export "* ]] && line="${line#export }"
    if [[ ! "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*[[:space:]]*= ]]; then
      skipped_lines+="$lineno "
      continue
    fi
    key="${line%%=*}"
    key="${key%"${key##*[![:space:]]}"}"
    value="${line#*=}"
    value="${value#"${value%%[![:space:]]*}"}"
    if [[ ${#value} -ge 2 && ( "$value" == \"*\" || "$value" == \'*\' ) ]]; then
      value="${value:1:${#value}-2}"
    fi
    case "$key" in
      GITHUB_TOKEN|HUGGINGFACE_HUB_TOKEN|HF_TOKEN|QUANT_AGENT_WORKSPACE|QUANT_AGENT_TORCH_SPEC|OPENAI_API_KEY)
        export "$key=$value"
        loaded_keys+="$key "
        ;;
      *)
        skipped_keys+="$key "
        ;;
    esac
  done < "$env_file"
  if [[ -n "$skipped_keys" ]]; then
    echo "[env] skipped keys this agent does not use (not exported): $skipped_keys" >&2
  fi
  if [[ -n "$skipped_lines" ]]; then
    echo "[env] skipped lines that are not KEY=value: $skipped_lines" >&2
  fi

  # HuggingFace libraries are split between two env-var names. Normalize.
  if [[ -n "${HUGGINGFACE_HUB_TOKEN:-}" && -z "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN="$HUGGINGFACE_HUB_TOKEN"
  elif [[ -n "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
    export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
  fi

  echo "[env] loaded keys from $env_file: $loaded_keys" >&2
  return 0
}

_quant_load_env "$@"

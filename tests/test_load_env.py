"""The .env loader: allowlisted keys only, inert parsing, bash and zsh. Fake values only."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LOADER = ROOT / "agents" / "skills" / "_shared" / "load_env.sh"
SHELLS = [shell for shell in ("bash", "zsh") if shutil.which(shell)]
REPORT = 'for k in HF_TOKEN HUGGINGFACE_HUB_TOKEN OTHER_TOOL_KEY PWNED; do printf "%s=%s\\n" "$k" "$(printenv $k)"; done'


def _load(shell: str, tmp_path: Path, text: str) -> tuple[int, dict[str, str], str]:
    env_file = tmp_path / ".env"
    env_file.write_text(text)
    env_file.chmod(0o600)
    script = f'source "{LOADER}" "{env_file}"; rc=$?; echo "rc=$rc"; {REPORT}'
    proc = subprocess.run([shell, "-c", script], capture_output=True, text=True, cwd=tmp_path, env={"PATH": "/usr/bin:/bin"})
    values = dict(line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line)
    return int(values.pop("rc")), values, proc.stderr


@pytest.mark.parametrize("shell", SHELLS)
def test_unknown_keys_are_skipped_not_exported(shell, tmp_path):
    rc, values, stderr = _load(shell, tmp_path, "OTHER_TOOL_KEY=fake-other\nHF_TOKEN=fake-hf-value\n")
    assert rc == 0
    assert values["HF_TOKEN"] == "fake-hf-value"
    assert values["HUGGINGFACE_HUB_TOKEN"] == "fake-hf-value"
    assert values["OTHER_TOOL_KEY"] == ""
    assert "OTHER_TOOL_KEY" in stderr and "fake" not in stderr


@pytest.mark.parametrize("shell", SHELLS)
def test_values_stay_inert_text(shell, tmp_path):
    rc, values, _ = _load(shell, tmp_path, 'HF_TOKEN=$(export PWNED=1)x"y"=z\n# comment\n\n')
    assert rc == 0
    assert values["HF_TOKEN"] == '$(export PWNED=1)x"y"=z'
    assert values["PWNED"] == ""


@pytest.mark.parametrize("shell", SHELLS)
def test_unreadable_lines_are_skipped_by_number(shell, tmp_path):
    rc, values, stderr = _load(shell, tmp_path, "OTHER-KEY=fake-x\nHF_TOKEN=fake-hf-value\nfake continuation\n")
    assert rc == 0
    assert values["HF_TOKEN"] == "fake-hf-value"
    assert "skipped lines that are not KEY=value: 1 3" in stderr
    assert "fake" not in stderr


@pytest.mark.parametrize("shell", SHELLS)
@pytest.mark.parametrize(
    "line",
    ["export HF_TOKEN=fake-hf-value", 'HF_TOKEN="fake-hf-value"', "HF_TOKEN='fake-hf-value'",
     "HF_TOKEN=fake-hf-value\r", "   HF_TOKEN=fake-hf-value", "HF_TOKEN = fake-hf-value",
     'export HF_TOKEN = "fake-hf-value"'],
)
def test_common_dotenv_styles_load(shell, tmp_path, line):
    rc, values, _ = _load(shell, tmp_path, f"# header\n{line}\n")
    assert rc == 0
    assert values["HF_TOKEN"] == "fake-hf-value"



"""Contract: installing Rabbit Code for daily coding does not require Torch."""
from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cli_package_has_no_training_dependency():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    required = metadata["project"].get("dependencies", [])
    assert not any("torch" in str(item).lower() for item in required)
    assert any("torch" in str(item).lower()
               for item in metadata["project"]["optional-dependencies"]["training"])
    assert "rabbit" in metadata["project"]["scripts"]
    assert "rabbit-code" in metadata["project"]["scripts"]


def test_standard_library_cli_and_web_ui_import_without_torch():
    """-S excludes site-packages, so no pip-installed torch is available."""
    source = (
        "import sys;"
        "sys.path.insert(0, " + repr(str(ROOT / "src")) + ");"
        "from rabbit_foundry import rabbit_code, helper_cli, web_ui, ui_models;"
        "assert 'torch' not in sys.modules;"
        "assert callable(helper_cli.main);"
        "assert callable(rabbit_code.make_parser)"
    )
    result = subprocess.run(
        [sys.executable, "-S", "-c", source],
        capture_output=True, text=True, timeout=15, check=False,
    )
    assert result.returncode == 0, result.stderr

import json

import pytest

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "run_space_bunny_fingerprint.py"
spec = importlib.util.spec_from_file_location("run_space_bunny_fingerprint", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_load_cases_requires_all_frozen_ids(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps({"id": module.FROZEN_ALPHA_IDS[0], "messages": [{"role": "user", "content": "x"}]}) + "\n")
    with pytest.raises(ValueError, match="missing frozen Alpha cases"):
        module.load_cases(path)


def test_defaults_target_current_opencode_space_bunny():
    assert module.DEFAULT_MODEL == "space-bunny-free"
    assert module.DEFAULT_ENDPOINT == "https://opencode.ai/zen/v1/chat/completions"

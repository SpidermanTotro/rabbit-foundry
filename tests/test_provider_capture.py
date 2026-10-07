import json

from rabbit_foundry.provider_capture import import_provider_exports


def test_import_openrouter_style_trace_and_redact_secret(tmp_path):
    source = tmp_path / "openrouter.jsonl"
    source.write_text(json.dumps({
        "provider": "openrouter",
        "model": "example/model",
        "api_key": "must-not-survive",
        "request": {"messages": [{"role": "user", "content": "fix this"}]},
        "response": {"choices": [{"message": {"role": "assistant", "content": "revised"}}]},
        "axis": "selfcorr",
    }) + "\n")
    out = tmp_path / "captures.jsonl"
    result = import_provider_exports([source], out)
    saved = out.read_text()
    row = json.loads(saved)
    assert result["captures"] == 1
    assert result["providers"] == {"openrouter": 1}
    assert row["model"] == "example/model"
    assert row["messages"][-1] == {"role": "assistant", "content": "revised"}
    assert "must-not-survive" not in saved


def test_import_kilo_style_messages_and_output(tmp_path):
    source = tmp_path / "kilo.json"
    source.write_text(json.dumps({
        "provider": "kilo",
        "model_id": "local-preview",
        "messages": [{"role": "user", "content": "debug"}],
        "output": "candidate repair",
    }))
    out = tmp_path / "captures.jsonl"
    result = import_provider_exports([source], out)
    row = json.loads(out.read_text())
    assert result["providers"] == {"kilo": 1}
    assert row["capture_kind"] == "provider-preview"
    assert row["messages"][-1]["content"] == "candidate repair"

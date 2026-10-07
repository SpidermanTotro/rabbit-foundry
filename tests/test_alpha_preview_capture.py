import json

from rabbit_foundry.alpha_preview_capture import alpha_session_to_capture, convert_alpha_export


def test_alpha_preview_capture_preserves_lineage_and_redacts(monkeypatch):
    monkeypatch.setattr("rabbit_foundry.alpha_preview_capture.detect_environment", lambda: {"platform": {"system": "Linux"}})
    row = alpha_session_to_capture({
        "id": "new-alpha-001",
        "provider": "preview-provider",
        "model": "alpha-preview",
        "client": "kilo",
        "axis": "debug",
        "messages": [
            {"role": "user", "content": "fix this API_KEY=secret-value"},
            {"role": "assistant", "content": "I will inspect the failure"},
            {"role": "tool", "content": "pytest failed"},
            {"role": "assistant", "content": "The test failed because state was lost; fix and test it."},
        ],
    })
    assert row["id"] == "new-alpha-001"
    assert row["source_kind"] == "observable_model_session"
    assert row["weight_access"] == "unavailable"
    assert row["client"] == "kilo"
    assert row["environment"]["platform"]["system"] == "Linux"
    assert "secret-value" not in json.dumps(row)


def test_alpha_export_becomes_course_ready_jsonl(tmp_path, monkeypatch):
    monkeypatch.setattr("rabbit_foundry.alpha_preview_capture.detect_environment", lambda: {"platform": {"system": "Linux"}})
    source = tmp_path / "alpha.jsonl"
    out = tmp_path / "captures.jsonl"
    source.write_text(json.dumps({
        "model": "alpha-preview",
        "messages": [{"role": "user", "content": "debug"}, {"role": "assistant", "content": "fixed"}],
    }) + "\n")
    result = convert_alpha_export(source, out)
    assert result["captures"] == 1
    saved = json.loads(out.read_text())
    assert saved["model"] == "alpha-preview"
    assert saved["source_kind"] == "observable_model_session"

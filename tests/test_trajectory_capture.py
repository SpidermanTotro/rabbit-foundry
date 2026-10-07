import json

from rabbit_foundry.trajectory_capture import import_trajectories, normalize_trajectory


def test_normalize_full_debug_repair_trajectory():
    row = normalize_trajectory({
        "id": "repair-1",
        "provider": "kilo",
        "model": "preview",
        "axis": "debug",
        "events": [
            {"kind": "user", "content": "fix the failing parser"},
            {"kind": "assistant", "content": "inspect parser and tests"},
            {"kind": "tool_call", "content": "pytest -q"},
            {"kind": "tool_result", "content": "1 failed, 9 passed"},
            {"kind": "failure", "content": "test_nested_parser failed"},
            {"kind": "diagnosis", "content": "nested token state is discarded"},
            {"kind": "revision", "content": "preserve nested token state"},
            {"kind": "test", "content": "10 passed"},
            {"kind": "final", "content": "fixed and regression tested"},
        ],
    }, source="preview.jsonl", index=0)

    assert row["capture_kind"] == "provider-trajectory"
    assert row["has_failure"] is True
    assert row["has_diagnosis"] is True
    assert row["has_revision"] is True
    assert row["has_test"] is True
    joined = json.dumps(row["messages"])
    assert "[failure]" in joined
    assert "[diagnosis]" in joined
    assert "[revision]" in joined
    assert "[test]" in joined


def test_import_quarantines_incomplete_trajectory(tmp_path):
    result = import_trajectories([
        {"events": [{"kind": "user", "content": "task"}]},
        {
            "id": "good",
            "events": [
                {"kind": "user", "content": "task"},
                {"kind": "final", "content": "done"},
            ],
        },
    ], tmp_path / "trajectories.jsonl")
    assert result["captures"] == 1
    assert result["rejected"] == 1
    assert "good" in (tmp_path / "trajectories.jsonl").read_text()


def test_redacts_secret_embedded_in_trajectory_event(tmp_path):
    out = tmp_path / "trajectories.jsonl"
    import_trajectories([{
        "events": [
            {"kind": "user", "content": "debug with API_KEY=super-secret"},
            {"kind": "final", "content": "removed Bearer another-secret"},
        ],
    }], out)
    saved = out.read_text()
    assert "super-secret" not in saved
    assert "another-secret" not in saved
    assert "<redacted>" in saved

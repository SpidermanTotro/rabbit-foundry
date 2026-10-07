import json

from rabbit_foundry.session_store import SessionStore


def test_session_store_redacts_secrets_and_defaults_nontrainable(tmp_path):
    store = SessionStore(tmp_path, session_id="test-session")
    store.record("user", {
        "content": "use API_KEY=super-secret",
        "authorization": "Bearer another-secret",
    })
    raw = store.path.read_text()
    assert "super-secret" not in raw
    assert "another-secret" not in raw
    row = json.loads(raw)
    assert row["source"] == "rabbit-code-runtime"
    assert row["training_allowed"] is False
    assert "<redacted>" in raw

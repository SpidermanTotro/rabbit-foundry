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


def test_chat_history_restores_chat_but_not_agent_protocol(tmp_path):
    store = SessionStore(tmp_path, session_id="resume")
    store.record("user", {"content": "first", "mode": "chat"})
    store.record("assistant", {"content": "answer", "mode": "chat"})
    store.record("user", {"content": "agent task", "mode": "agent"})
    store.record("assistant", {
        "content": '{"type":"final","content":"agent"}',
        "mode": "agent",
    })

    resumed = SessionStore(tmp_path, session_id="resume")
    assert resumed.resumed is True
    assert resumed.chat_history() == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
    ]

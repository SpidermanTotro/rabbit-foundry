from rabbit_foundry.agent_loop import AgentLoop, parse_action
from rabbit_foundry.agent_runtime import RabbitCodeRuntime
from rabbit_foundry.model_router import ProviderConfig
from rabbit_foundry.permissions import PermissionPolicy
from rabbit_foundry.session_store import SessionStore
from rabbit_foundry.workspace import Workspace


class SequenceRouter:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []
        self.config = ProviderConfig(
            "fake",
            "http://127.0.0.1:8765/v1",
            "rabbit-code",
        )

    def provider(self, provider_id):
        return self.config

    def complete(self, provider_id, messages, *, max_tokens=1024):
        self.calls.append([dict(message) for message in messages])
        content = self.outputs.pop(0)
        return {
            "choices": [{
                "message": {"role": "assistant", "content": content}
            }]
        }


def runtime(tmp_path, router):
    return RabbitCodeRuntime(
        router,
        "fake",
        Workspace(tmp_path, PermissionPolicy()),
        SessionStore(tmp_path / "sessions", session_id="agent"),
    )


def test_agent_loop_reads_real_file_then_finishes(tmp_path):
    (tmp_path / "note.txt").write_text("rabbit facts")
    router = SequenceRouter([
        '{"type":"tool","tool":"read","args":{"path":"note.txt"}}',
        '{"type":"final","content":"I inspected note.txt."}',
    ])
    result = AgentLoop(runtime(tmp_path, router)).run("inspect the note")
    assert result == "I inspected note.txt."
    assert "rabbit facts" in router.calls[1][-1]["content"]


def test_agent_loop_reports_write_approval_failure_to_model(tmp_path):
    router = SequenceRouter([
        '{"type":"tool","tool":"write","args":{"path":"x.txt","content":"x"}}',
        '{"type":"final","content":"write needs approval"}',
    ])
    result = AgentLoop(runtime(tmp_path, router), approve_write=False).run("write x")
    assert result == "write needs approval"
    assert "ApprovalRequired" in router.calls[1][-1]["content"]
    assert not (tmp_path / "x.txt").exists()


def test_parse_action_accepts_fenced_json_and_plain_text_final():
    fence = chr(96) * 3
    fenced = fence + 'json\n{"type":"final","content":"done"}\n' + fence
    assert parse_action(fenced) == {"type": "final", "content": "done"}
    assert parse_action("normal answer") == {
        "type": "final",
        "content": "normal answer",
    }


def test_parse_action_accepts_common_local_model_tool_shapes():
    assert parse_action(
        '{"tool":"read","args":{"path":"README.md"}}'
    ) == {
        "type": "tool",
        "tool": "read",
        "args": {"path": "README.md"},
    }

    assert parse_action(
        '{"action":"read","path":"README.md"}'
    ) == {
        "type": "tool",
        "tool": "read",
        "args": {"path": "README.md"},
    }

    assert parse_action(
        '{"type":"function","function":{"name":"grep","arguments":"{\\\"needle\\\":\\\"TODO\\\"}"}}'
    ) == {
        "type": "tool",
        "tool": "grep",
        "args": {"needle": "TODO"},
    }


def test_parse_action_accepts_common_final_aliases():
    assert parse_action('{"answer":"done"}') == {
        "type": "final",
        "content": "done",
    }
    assert parse_action('{"action":"finish","message":"done"}') == {
        "type": "final",
        "content": "done",
    }


def test_agent_loop_recovers_from_structured_protocol_error(tmp_path):
    (tmp_path / "note.txt").write_text("rabbit facts")
    router = SequenceRouter([
        '{"unexpected":true}',
        '{"tool":"read","path":"note.txt"}',
        '{"answer":"Recovered and inspected the note."}',
    ])
    result = AgentLoop(
        runtime(tmp_path, router),
        max_steps=4,
    ).run("inspect the note")
    assert result == "Recovered and inspected the note."
    assert "RABBIT_PROTOCOL_ERROR" in router.calls[1][-1]["content"]
    assert "rabbit facts" in router.calls[2][-1]["content"]


def test_agent_mode_rejects_plain_prose_then_uses_tool(tmp_path):
    (tmp_path / "README.md").write_text("Rabbit Code repository")
    router = SequenceRouter([
        "I cannot access local files.",
        '{"tool":"read","path":"README.md"}',
        '{"type":"final","content":"I inspected the repository README."}',
    ])
    result = AgentLoop(
        runtime(tmp_path, router),
        max_steps=4,
    ).run("inspect this repository")
    assert result == "I inspected the repository README."
    assert "RABBIT_PROTOCOL_ERROR" in router.calls[1][-1]["content"]
    assert "Rabbit Code repository" in router.calls[2][-1]["content"]


def test_agent_mode_rejects_final_before_any_tool(tmp_path):
    (tmp_path / "README.md").write_text("Rabbit Code repository")
    router = SequenceRouter([
        '{"type":"final","content":"I cannot access local files."}',
        '{"tool":"read","path":"README.md"}',
        '{"type":"final","content":"Now I inspected it."}',
    ])
    result = AgentLoop(
        runtime(tmp_path, router),
        max_steps=4,
    ).run("inspect this repository")
    assert result == "Now I inspected it."
    assert "RABBIT_TOOL_REQUIRED" in router.calls[1][-1]["content"]
    assert "Rabbit Code repository" in router.calls[2][-1]["content"]


def test_parse_action_can_still_allow_plain_final_outside_agent_loop():
    assert parse_action("plain answer") == {
        "type": "final",
        "content": "plain answer",
    }

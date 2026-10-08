from rabbit_foundry.native_tool_probe import (
    PROBE_NAME,
    PROBE_TOKEN,
    inspect_response,
    local_endpoint,
    probe_payload,
)


def test_probe_is_local_only():
    assert local_endpoint("http://127.0.0.1:8765/v1/chat/completions")
    assert local_endpoint("http://localhost:11434/v1/chat/completions")
    assert local_endpoint("http://[::1]:8765/v1/chat/completions")
    assert not local_endpoint("https://example.com/v1/chat/completions")
    assert not local_endpoint("http://localhost.example.com:8765/")
    assert not local_endpoint("file:///tmp/fake")


def test_probe_payload_contains_no_workspace_information():
    payload = probe_payload("rabbit-code")
    assert payload["stream"] is False
    assert payload["model"] == "rabbit-code"
    assert payload["tools"][0]["function"]["name"] == PROBE_NAME
    assert "read" not in str(payload["tools"]).lower()
    assert PROBE_TOKEN in str(payload["messages"])


def test_probe_verifies_exact_native_call_not_plain_text():
    assert inspect_response({
        "choices": [{
            "message": {
                "content": None,
                "tool_calls": [{
                    "type": "function",
                    "function": {
                        "name": PROBE_NAME,
                        "arguments": '{"token":"rabbit-native-tool-probe"}',
                    },
                }],
            },
        }],
    }) == {"native_tool_calls": True, "reason": "verified_tool_call"}

    assert inspect_response({
        "choices": [{
            "message": {"content": "I would call rabbit_probe but cannot."},
        }],
    }) == {
        "native_tool_calls": False,
        "reason": "model_returned_no_native_calls",
    }


def test_probe_rejects_wrong_args_and_wrong_tool_names():
    wrong_args = {
        "choices": [{
            "message": {
                "tool_calls": [{
                    "function": {
                        "name": PROBE_NAME,
                        "arguments": '{"token":"wrong"}',
                    },
                }],
            },
        }],
    }
    assert inspect_response(wrong_args)["reason"] == "wrong_tool_arguments"
    wrong_name = {
        "choices": [{
            "message": {
                "tool_calls": [{
                    "function": {
                        "name": "write",
                        "arguments": '{"token":"rabbit-native-tool-probe"}',
                    },
                }],
            },
        }],
    }
    assert inspect_response(wrong_name)["reason"] == "wrong_tool_name"

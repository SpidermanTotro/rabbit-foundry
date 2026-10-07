from types import SimpleNamespace

from rabbit_foundry.rabbit_code import execute_command, make_parser


class FakeRuntime:
    def capabilities(self):
        return {"product": "Rabbit Code"}

    def read(self, path):
        return f"READ:{path}"

    def git_status(self):
        return " M README.md\n"

    def ask(self, text, *, max_tokens=1024):
        return f"CHAT:{text}"

    def ask_stream(self, text, *, max_tokens=1024):
        yield "STREAM:"
        yield text


def args():
    return SimpleNamespace(
        agent_steps=8,
        allow_write=False,
        allow_exec=False,
        max_tokens=1024,
        stream=False,
    )


def test_one_shot_parser_accepts_shell_command():
    parsed = make_parser().parse_args([
        "--workspace", ".",
        "read",
        "README.md",
    ])
    assert parsed.command == "read"
    assert parsed.command_args == ["README.md"]


def test_execute_command_accepts_shell_and_repl_forms(capsys):
    runtime = FakeRuntime()

    assert execute_command(runtime, args(), ["capabilities"]) is False
    assert "Rabbit Code" in capsys.readouterr().out

    assert execute_command(runtime, args(), ["/read", "README.md"]) is False
    assert "READ:README.md" in capsys.readouterr().out

    assert execute_command(runtime, args(), ["git-status"]) is False
    assert "README.md" in capsys.readouterr().out


def test_quit_command_requests_exit():
    runtime = FakeRuntime()
    assert execute_command(runtime, args(), ["/quit"]) is True
    assert execute_command(runtime, args(), ["exit"]) is True


def test_chat_command_supports_plain_and_streamed_output(capsys):
    runtime = FakeRuntime()
    plain = args()
    execute_command(runtime, plain, ["chat", "hello"])
    assert "CHAT:hello" in capsys.readouterr().out

    streamed = args()
    streamed.stream = True
    execute_command(runtime, streamed, ["chat", "hello"])
    assert "STREAM:hello" in capsys.readouterr().out

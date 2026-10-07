from types import SimpleNamespace

from rabbit_foundry.rabbit_code import execute_command, make_parser


class FakeRuntime:
    def capabilities(self):
        return {"product": "Rabbit Code"}

    def read(self, path):
        return f"READ:{path}"

    def git_status(self):
        return " M README.md\n"


def args():
    return SimpleNamespace(
        agent_steps=8,
        allow_write=False,
        allow_exec=False,
        max_tokens=1024,
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

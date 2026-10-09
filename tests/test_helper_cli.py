"""Regression tests for the friendly, local-only rabbit helper CLI."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from rabbit_foundry import helper_cli


def test_no_args_displays_help(capsys):
    assert helper_cli.main([]) == 0
    out = capsys.readouterr().out
    for word in ("doctor", "models", "ui", "chat", "agent"):
        assert word in out


def test_version(capsys):
    assert helper_cli.main(["version"]) == 0
    assert "rabbit-code" in capsys.readouterr().out


def test_doctor_never_queries_nonlocal_urls(monkeypatch):
    with pytest.raises(ValueError, match="loopback"):
        helper_cli._loopback_json("https://example.com/api/tags")
    monkeypatch.setattr(helper_cli, "_loopback_json", lambda url: None)
    data = helper_cli.diagnostic_report()
    assert data["ollama_running"] is False
    assert data["rabbit_gateway_running"] is False
    assert "python" in data
    assert "path_local_bin" in data
    assert helper_cli.main(["doctor", "--json"]) == 0


def test_doctor_json_output(capsys, monkeypatch):
    monkeypatch.setattr(helper_cli, "diagnostic_report", lambda: {
        "os": "Linux", "ollama_running": False,
    })
    helper_cli.main(["doctor", "--json"])
    result = json.loads(capsys.readouterr().out)
    assert result == {"os": "Linux", "ollama_running": False}


def test_models_reads_only_installed_local_models(monkeypatch, capsys):
    monkeypatch.setattr(
        "rabbit_foundry.ui_models.installed_ollama_models",
        lambda: ["qwen3:8b", "qwen2.5-coder:7b"],
    )
    assert helper_cli.main(["models"]) == 0
    output = capsys.readouterr().out
    assert "qwen3:8b" in output
    assert "qwen2.5-coder:7b" in output


def test_ui_dispatch_without_network_or_write_approvals(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(helper_cli, "_main_engine", lambda argv: captured.append(argv))
    assert helper_cli.main(["ui", "-w", str(tmp_path), "--port", "8877",
                            "--no-browser"]) == 0
    argv = captured[0]
    assert argv == [
        "--workspace", str(tmp_path), "--ui", "--ui-port", "8877",
    ]
    assert "--allow-network" not in argv
    assert "--allow-write" not in argv
    assert "--allow-exec" not in argv


def test_chat_direct_ollama_routes_only_to_loopback(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(helper_cli, "_main_engine", lambda argv: captured.append(argv))
    monkeypatch.setattr(
        "rabbit_foundry.ui_models.installed_ollama_models",
        lambda: ["qwen3:8b"],
    )
    assert helper_cli.main([
        "chat", "-w", str(tmp_path),
        "--model", "qwen3:8b", "hello", "rabbit",
    ]) == 0
    argv = captured[0]
    assert "http://127.0.0.1:11434/v1" in argv
    assert argv[-2:] == ["chat", "hello rabbit"]


def test_agent_without_model_uses_existing_gateway(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(helper_cli, "_main_engine", lambda argv: captured.append(argv))
    assert helper_cli.main(["agent", "-w", str(tmp_path), "inspect", "README.md"]) == 0
    assert captured[0][-2:] == ["agent", "inspect README.md"]
    assert "--allow-write" not in captured[0]
    assert "--allow-exec" not in captured[0]


def test_uninstalled_ollama_model_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "rabbit_foundry.ui_models.installed_ollama_models",
        lambda: ["qwen3:8b"],
    )
    with pytest.raises(SystemExit) as err:
        helper_cli.main([
            "chat", "--model", "unknown:99b", "hello",
        ])
    assert err.value.code == 2

from pathlib import Path

from rabbit_foundry.sandbox import podman_command


def test_sandbox_disables_network_and_mounts_source_read_only():
    cmd = podman_command("python:3.12", Path("/tmp/repo"), ["python", "-m", "pytest"])
    assert "--network=none" in cmd
    assert "--read-only" in cmd
    assert "--cap-drop=all" in cmd
    assert any(x.endswith(":/workspace:ro,Z") for x in cmd)

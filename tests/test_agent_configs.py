import json

from rabbit_foundry.agent_configs import kilo_config, opencode_config


def test_opencode_points_to_local_openai_compatible_server():
    cfg = json.loads(opencode_config())
    assert cfg["model"] == "rabbit/rabbit-local"
    assert cfg["providers"]["rabbit"]["settings"]["baseURL"].endswith("/v1")


def test_kilo_points_to_same_server():
    cfg = json.loads(kilo_config())
    assert cfg["model"] == "rabbit/rabbit-local"
    assert cfg["provider"]["rabbit"]["options"]["baseURL"].endswith("/v1")

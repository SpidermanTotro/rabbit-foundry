import urllib.error

import pytest

import rabbit_foundry.space_bunny_failover as failover


def test_retryable_failure_uses_second_route(monkeypatch):
    monkeypatch.setenv("OPENCODE_ZEN_API_KEY", "z")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "g")
    seen = []
    def run_one(route, key):
        seen.append(route.name)
        if route.name == "zen":
            raise urllib.error.HTTPError(route.endpoint, 429, "rate", {}, None)
        return "ok"
    result, route = failover.run_with_failover(run_one, preferred="zen")
    assert result == "ok"
    assert route.name == "go"
    assert seen == ["zen", "go"]


def test_auth_failure_does_not_silently_hop_routes(monkeypatch):
    monkeypatch.setenv("OPENCODE_ZEN_API_KEY", "bad")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "g")
    def run_one(route, key):
        raise urllib.error.HTTPError(route.endpoint, 401, "auth", {}, None)
    with pytest.raises(urllib.error.HTTPError) as exc:
        failover.run_with_failover(run_one)
    assert exc.value.code == 401


def test_no_authorized_routes_fails_cleanly(monkeypatch):
    monkeypatch.delenv("OPENCODE_ZEN_API_KEY", raising=False)
    monkeypatch.delenv("OPENCODE_GO_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="No authorized"):
        failover.run_with_failover(lambda route, key: None)

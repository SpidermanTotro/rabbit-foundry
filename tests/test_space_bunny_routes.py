import pytest

from rabbit_foundry.space_bunny_routes import routes_by_preference


def test_zen_then_go():
    routes = routes_by_preference("zen")
    assert [r.name for r in routes] == ["zen", "go"]
    assert routes[0].endpoint.endswith("/zen/v1/chat/completions")
    assert routes[1].endpoint.endswith("/zen/go/v1/chat/completions")


def test_go_then_zen():
    routes = routes_by_preference("go")
    assert [r.name for r in routes] == ["go", "zen"]
    assert all(r.model == "space-bunny-free" for r in routes)


def test_unknown_route_fails_closed():
    with pytest.raises(ValueError):
        routes_by_preference("other")

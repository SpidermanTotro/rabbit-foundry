from rabbit_foundry.router import Route, RouterPolicy


def test_dry_policy_never_silently_falls_back():
    policy = RouterPolicy(fallback_model="qwen2.5-coder:7b", allow_fallback=False)
    result = policy.decide(confidence=0.1)
    assert result.route == Route.DENY


def test_explicit_fallback_can_route_to_qwen():
    policy = RouterPolicy(fallback_model="qwen2.5-coder:7b", allow_fallback=True)
    result = policy.decide(confidence=0.1)
    assert result.route == Route.FALLBACK

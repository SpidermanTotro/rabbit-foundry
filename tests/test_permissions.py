import pytest

from rabbit_foundry.permissions import (
    ApprovalRequired,
    Decision,
    PermissionDenied,
    PermissionPolicy,
)


def test_default_policy_is_readable_but_mutations_require_approval():
    policy = PermissionPolicy()
    policy.require("read")
    policy.require("search")
    with pytest.raises(ApprovalRequired):
        policy.require("write")
    policy.require("write", approved=True)
    with pytest.raises(PermissionDenied):
        policy.require("network")


def test_unknown_permission_fails_closed():
    with pytest.raises(ValueError):
        PermissionPolicy().require("teleport")

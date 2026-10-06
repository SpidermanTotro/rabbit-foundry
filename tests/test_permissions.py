from rabbit_foundry.permissions import Permission, PermissionPolicy


def test_dry_run_cannot_edit_or_shell():
    p = PermissionPolicy.dry_run()
    assert p.read == Permission.ALLOW
    assert p.edit == Permission.DENY
    assert p.shell == Permission.DENY


def test_supervised_asks_before_mutation():
    p = PermissionPolicy.supervised()
    assert p.edit == Permission.ASK
    assert p.shell == Permission.ASK

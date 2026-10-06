from rabbit_foundry import bunny_inventory

def test_inventory_reports_missing_artifact(tmp_path):
    result = bunny_inventory.inspect_artifact("missing", str(tmp_path / "nope"))
    assert result["exists"] is False

def test_inventory_hashes_small_file(tmp_path):
    path = tmp_path / "PROVENANCE.json"
    path.write_text('{"ok": true}\n')
    result = bunny_inventory.inspect_artifact("provenance", str(path))
    assert result["exists"] is True
    assert result["type"] == "file"
    assert len(result["sha256"]) == 64

def test_default_inventory_uses_portable_home_paths():
    assert all(path.startswith("~/") for _, path in bunny_inventory.DEFAULT_ARTIFACTS)

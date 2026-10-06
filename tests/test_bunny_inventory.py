from rabbit_foundry import bunny_inventory


def test_inventory_reports_missing_artifact(tmp_path):
    result = bunny_inventory.inspect_artifact("missing", str(tmp_path / "nope"))
    assert result["exists"] is False


def test_inventory_hashes_and_parses_provenance(tmp_path):
    path = tmp_path / "PROVENANCE.json"
    path.write_text('{"ok": true}\n')
    result = bunny_inventory.inspect_artifact("provenance", str(path))
    assert result["exists"] is True
    assert result["type"] == "file"
    assert len(result["sha256"]) == 64
    assert result["valid_json"] is True


def test_inventory_rejects_invalid_provenance_json(tmp_path):
    path = tmp_path / "PROVENANCE.json"
    path.write_text("{broken")
    result = bunny_inventory.inspect_artifact("provenance", str(path))
    assert result["valid_json"] is False


def test_inventory_checks_gguf_magic_without_large_hash(tmp_path):
    path = tmp_path / "model.gguf"
    path.write_bytes(b"GGUF" + b"\x00" * 32)
    result = bunny_inventory.inspect_artifact("gguf_q4km", str(path))
    assert result["gguf_magic_valid"] is True
    assert len(result["sha256"]) == 64


def test_inventory_summarizes_model_directory(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "a.bin").write_bytes(b"1234")
    nested = model / "nested"
    nested.mkdir()
    (nested / "b.json").write_bytes(b"12")
    result = bunny_inventory.inspect_artifact("merged_model", str(model))
    assert result["type"] == "directory"
    assert result["file_count"] == 2
    assert result["total_bytes"] == 6


def test_default_inventory_uses_portable_home_paths():
    assert all(path.startswith("~/") for _, path in bunny_inventory.DEFAULT_ARTIFACTS)

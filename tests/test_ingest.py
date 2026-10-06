from pathlib import Path

from rabbit_foundry.ingest import build_episode_manifest, ingest_file
from rabbit_foundry.sources import RepositorySource


def test_ingest_pinned_file_without_network(tmp_path):
    source = RepositorySource("owner/repo", "abcdef0123456789", "MIT")
    item = ingest_file(
        source,
        "src/a.py",
        tmp_path,
        fetcher=lambda _: b"print('rabbit')\n" * 20,
    )
    assert Path(item.cache_path).exists()
    assert item.bytes > 0
    payload = build_episode_manifest([item], tmp_path / "episodes.json", window=32)
    assert payload["episodes"]

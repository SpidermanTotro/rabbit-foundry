from pathlib import Path

from rabbit_foundry.ingest import IngestedFile, build_episode_manifest


def ingested(tmp_path: Path, commit: str, text: str, name: str) -> IngestedFile:
    cache = tmp_path / name
    cache.write_text(text)
    return IngestedFile(
        repository="example/repo",
        commit=commit,
        license="MIT",
        path="src/example.py",
        sha256=(commit[0] * 64),
        bytes=cache.stat().st_size,
        cache_path=str(cache),
    )


def test_manifest_builds_hidden_diff_from_adjacent_pinned_revisions(tmp_path: Path):
    before_commit = "a" * 40
    after_commit = "b" * 40
    before = ingested(
        tmp_path, before_commit,
        "value = 1\nprint(value)\n" * 8,
        "before.py",
    )
    after = ingested(
        tmp_path, after_commit,
        "value = 2\nprint(value)\n" * 8,
        "after.py",
    )

    payload = build_episode_manifest(
        [before, after],
        tmp_path / "episodes.json",
        window=16,
        include_repairs=False,
    )

    diffs = [row for row in payload["episodes"] if row["skill"] == "hidden_diff"]
    assert diffs
    assert "hidden_diff" in payload["skills"]
    assert {row["source_commit"] for row in diffs} == {after_commit}
    assert {row["source_before_commit"] for row in diffs} == {before_commit}
    assert all(len(bytes.fromhex(row["prompt_hex"])) == 16 for row in diffs)
    assert all(len(bytes.fromhex(row["target_hex"])) == 16 for row in diffs)


def test_hidden_diff_requires_distinct_changed_revisions(tmp_path: Path):
    text = "same = True\n" * 16
    a = ingested(tmp_path, "c" * 40, text, "a.py")
    b = ingested(tmp_path, "c" * 40, text, "b.py")
    payload = build_episode_manifest(
        [a, b],
        tmp_path / "episodes.json",
        window=8,
        include_repairs=False,
    )
    assert not [row for row in payload["episodes"] if row["skill"] == "hidden_diff"]

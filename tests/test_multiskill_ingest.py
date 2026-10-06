from pathlib import Path

from rabbit_foundry.ingest import IngestedFile, build_episode_manifest


def test_manifest_contains_prediction_and_repair_skills(tmp_path: Path):
    source = tmp_path / "sample.py"
    source.write_text(
        "def add(a, b):\n"
        "    total = a + b\n"
        "    return total\n"
        "\n"
        "print(add(2, 3))\n"
    )
    item = IngestedFile(
        repository="example/repo",
        commit="a" * 40,
        license="MIT",
        path="sample.py",
        sha256="1" * 64,
        bytes=source.stat().st_size,
        cache_path=str(source),
    )
    payload = build_episode_manifest([item], tmp_path / "episodes.json", window=8)
    skills = {row["skill"] for row in payload["episodes"]}
    assert payload["version"] == 2
    assert payload["skills"] == sorted(skills)
    assert "code_prediction" in skills
    assert "code_repair" in skills


def test_repair_generation_can_be_disabled(tmp_path: Path):
    source = tmp_path / "sample.py"
    source.write_text("one = 1\ntwo = 2\nthree = 3\nfour = 4\n")
    item = IngestedFile(
        repository="example/repo",
        commit="b" * 40,
        license="MIT",
        path="sample.py",
        sha256="2" * 64,
        bytes=source.stat().st_size,
        cache_path=str(source),
    )
    payload = build_episode_manifest(
        [item], tmp_path / "episodes.json", window=4, include_repairs=False
    )
    assert payload["skills"] == ["code_prediction"]

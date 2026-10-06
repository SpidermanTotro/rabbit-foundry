from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

from .episodes import next_byte_episodes
from .provenance import ProvenanceLedger, SourceRecord
from .sources import RepositorySource, is_training_path


@dataclass(frozen=True)
class IngestedFile:
    repository: str
    commit: str
    license: str
    path: str
    sha256: str
    bytes: int
    cache_path: str


def raw_url(repository: str, commit: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{repository}/{commit}/{path}"


def fetch_bytes(url: str, timeout: int = 30) -> bytes:
    req = Request(url, headers={"User-Agent": "rabbit-foundry/0.3"})
    with urlopen(req, timeout=timeout) as response:
        return response.read()


def ingest_file(
    source: RepositorySource,
    path: str,
    cache_dir: Path,
    *,
    fetcher=fetch_bytes,
) -> IngestedFile:
    source.validate()
    if not is_training_path(path):
        raise ValueError(f"path is not an allowed training source: {path}")

    data = fetcher(raw_url(source.repository, source.commit, path))
    digest = hashlib.sha256(data).hexdigest()
    target = cache_dir / source.repository.replace("/", "__") / source.commit / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)

    return IngestedFile(
        repository=source.repository,
        commit=source.commit,
        license=source.license,
        path=path,
        sha256=digest,
        bytes=len(data),
        cache_path=str(target),
    )


def build_episode_manifest(files: list[IngestedFile], out: Path, window: int = 128) -> dict:
    rows = []
    for item in files:
        content = Path(item.cache_path).read_bytes()
        for ep in next_byte_episodes(
            item.repository, item.commit, item.path, content, window=window
        ):
            rows.append({
                "episode_id": ep.episode_id,
                "skill": ep.skill,
                "source_repository": ep.source_repository,
                "source_commit": ep.source_commit,
                "source_path": ep.source_path,
                "prompt_hex": ep.prompt.hex(),
                "target_hex": ep.target.hex(),
            })

    payload = {"version": 1, "window": window, "episodes": rows}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def write_provenance(files: list[IngestedFile], out: Path) -> None:
    ledger = ProvenanceLedger()
    for item in files:
        ledger.add(SourceRecord(
            repository=item.repository,
            commit=item.commit,
            license=item.license,
            path=item.path,
            purpose="self-supervised code training",
        ))
    ledger.write(out)

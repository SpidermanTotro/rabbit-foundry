from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class SourceRecord:
    repository: str
    commit: str
    license: str
    path: str
    purpose: str


class ProvenanceLedger:
    def __init__(self):
        self.records: list[SourceRecord] = []

    def add(self, record: SourceRecord) -> None:
        self.records.append(record)

    def write(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps([asdict(r) for r in self.records], indent=2) + "\n"
        )

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


PRESERVE_MODES = (
    "behavior",
    "coding",
    "debugging",
    "tool_use",
    "style",
    "knowledge",
)
EXPORT_FORMATS = ("checkpoint", "gguf")


@dataclass
class ModelRecipe:
    name: str
    preserve: list[str] = field(default_factory=lambda: ["behavior", "coding", "debugging", "tool_use"])
    export: list[str] = field(default_factory=lambda: ["checkpoint"])
    source_model: str | None = None
    target_params: str = "auto"
    local_only: bool = True

    def validate(self) -> None:
        unknown = sorted(set(self.preserve) - set(PRESERVE_MODES))
        if unknown:
            raise ValueError("unknown preserve modes: " + ", ".join(unknown))
        bad_exports = sorted(set(self.export) - set(EXPORT_FORMATS))
        if bad_exports:
            raise ValueError("unknown export formats: " + ", ".join(bad_exports))
        if not self.name.strip():
            raise ValueError("model name cannot be empty")

    def save(self, path: str | Path) -> Path:
        self.validate()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), indent=2) + "\n")
        return target


def recipe_from_answers(
    name: str,
    preserve: list[str],
    *,
    export_gguf: bool = False,
    source_model: str | None = None,
    target_params: str = "auto",
) -> ModelRecipe:
    exports = ["checkpoint"] + (["gguf"] if export_gguf else [])
    recipe = ModelRecipe(
        name=name,
        preserve=preserve,
        export=exports,
        source_model=source_model,
        target_params=target_params,
    )
    recipe.validate()
    return recipe

from __future__ import annotations

import argparse
from pathlib import Path

from rabbit_foundry.model_recipe import PRESERVE_MODES, recipe_from_answers


def ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(prompt + suffix + ": ").strip()
    return value or (default or "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive Rabbit model recipe wizard")
    parser.add_argument("--out", default="configs/generated-model.json")
    args = parser.parse_args()

    print("Rabbit Code - Make a Model")
    name = ask("Model name", "rabbit-native")
    print("\nWhat do you want to preserve?")
    selected = []
    for mode in PRESERVE_MODES:
        default = "y" if mode in {"behavior", "coding", "debugging", "tool_use"} else "n"
        if ask(f"  {mode}? y/n", default).lower().startswith("y"):
            selected.append(mode)
    source = ask("\nSource/teacher model label (optional)")
    size = ask("Target size/parameter class", "auto")
    gguf = ask("Prepare for GGUF export when compatible? y/n", "y").lower().startswith("y")

    recipe = recipe_from_answers(
        name,
        selected,
        export_gguf=gguf,
        source_model=source or None,
        target_params=size,
    )
    path = recipe.save(Path(args.out))
    print(f"\nSaved model recipe: {path}")
    if gguf:
        print("GGUF requested. Export still requires an architecture supported by the configured llama.cpp path.")


if __name__ == "__main__":
    main()

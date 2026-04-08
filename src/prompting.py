from __future__ import annotations

from .constants import BOM_TOKEN, EOM_TOKEN, TEXT2MOL_DEFINITION


def normalize_free_text(text: str) -> str:
    return " ".join(str(text).split())


def normalize_selfies_text(text: str) -> str:
    return "".join(str(text).split())


def build_text2mol_prompt(description: str, definition: str = TEXT2MOL_DEFINITION) -> str:
    normalized_description = normalize_free_text(description)
    return (
        f"{definition}\n\n"
        "Now complete the following example -\n"
        f"Input: {normalized_description}\n"
        "Output: "
    )


def wrap_selfies_target(selfies_text: str) -> str:
    normalized_selfies = normalize_selfies_text(selfies_text)
    return f"{BOM_TOKEN}{normalized_selfies}{EOM_TOKEN}"


def unwrap_selfies_target(text: str) -> str:
    return normalize_selfies_text(text.replace(BOM_TOKEN, "").replace(EOM_TOKEN, ""))

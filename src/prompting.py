from __future__ import annotations

from molecules.selfies import normalize_selfies_text, unwrap_selfies_target, wrap_selfies_target

from .constants import BOM_TOKEN, EOM_TOKEN, TEXT2MOL_DEFINITION


def normalize_free_text(text: str) -> str:
    return " ".join(str(text).split())


def build_text2mol_prompt(description: str, definition: str = TEXT2MOL_DEFINITION) -> str:
    normalized_description = normalize_free_text(description)
    return (
        f"{definition}\n\n"
        "Now complete the following example -\n"
        f"Input: {normalized_description}\n"
        "Output: "
    )

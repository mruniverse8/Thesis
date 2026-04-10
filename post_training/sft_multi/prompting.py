from __future__ import annotations

from src.constants import DIVERSE_TEXT2MOL_DEFINITION
from src.prompting import build_text2mol_prompt


def build_diverse_text2mol_prompt(description: str) -> str:
    return build_text2mol_prompt(description, definition=DIVERSE_TEXT2MOL_DEFINITION)

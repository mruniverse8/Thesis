"""Project-wide constants for BioT5+ text-to-molecule SFT."""

BOM_TOKEN = "<bom>"
EOM_TOKEN = "<eom>"

TEXT2MOL_DEFINITION = (
    "Definition: You are given a molecule description in English. "
    "Your job is to generate the molecule SELFIES that fits the description."
)

DIVERSE_TEXT2MOL_DEFINITION = (
    "Definition: You are given a molecule description in English. "
    "Your job is to generate a diverse set of molecule SELFIES that fit the description."
)

DEFAULT_MODEL_NAME = "QizhiPei/biot5-plus-base"
DEFAULT_BASE_TOKENIZER_NAME = "google/t5-v1_1-base"

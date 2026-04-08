from src.tokenizer_utils import get_existing_additional_special_tokens


class DummyTokenizerNoAttr:
    special_tokens_map = {"additional_special_tokens": ["<bom>", "<eom>"]}
    special_tokens_map_extended = {}


class DummyTokenizerWithDuplicates:
    additional_special_tokens = ["<bom>", "<eom>"]
    special_tokens_map = {"additional_special_tokens": ["<bom>"]}
    special_tokens_map_extended = {"additional_special_tokens": ["<eom>", "<extra>"]}


def test_get_existing_additional_special_tokens_without_attribute():
    tokenizer = DummyTokenizerNoAttr()
    assert get_existing_additional_special_tokens(tokenizer) == ["<bom>", "<eom>"]


def test_get_existing_additional_special_tokens_deduplicates_sources():
    tokenizer = DummyTokenizerWithDuplicates()
    assert get_existing_additional_special_tokens(tokenizer) == ["<bom>", "<eom>", "<extra>"]

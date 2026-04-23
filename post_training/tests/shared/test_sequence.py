import torch

from src.constants import EOM_TOKEN

from post_training.shared.sequence import (
    STAGE_SEPARATOR,
    append_stage_to_prefix,
    build_stage_prefix,
    parse_single_staged_molecule,
    parse_staged_target,
    project_sampled_stage_to_no_h,
    serialize_staged_target,
)


class DummyTokenizer:
    def __init__(self, id_to_token: dict[int, str]) -> None:
        self.id_to_token = dict(id_to_token)
        self.token_to_id = {token: token_id for token_id, token in self.id_to_token.items()}

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_attention_mask: bool = False,
        return_tensors: str | None = None,
    ) -> dict[str, object]:
        del add_special_tokens, return_attention_mask
        remaining = str(text)
        token_ids: list[int] = []
        known_tokens = sorted(self.token_to_id, key=len, reverse=True)
        while remaining:
            matched_token = next(
                (token for token in known_tokens if remaining.startswith(token)),
                None,
            )
            if matched_token is None:
                raise ValueError(f"Unable to tokenize {text!r}; next chunk was {remaining!r}.")
            token_ids.append(int(self.token_to_id[matched_token]))
            remaining = remaining[len(matched_token) :]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([token_ids], dtype=torch.long)}
        return {"input_ids": token_ids}

    def convert_tokens_to_ids(self, token: str) -> int:
        return int(self.token_to_id[token])

    def decode(self, token_ids, skip_special_tokens: bool = False, clean_up_tokenization_spaces: bool = True):
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(self.id_to_token[int(token_id)] for token_id in token_ids)


def test_serialize_and_parse_staged_target_round_trip() -> None:
    sequence = serialize_staged_target(["[C][O]", "[C][C]"])

    assert sequence == "<bom>[C][O]<eom> <bom>[C][C]<eom>"
    assert parse_staged_target(sequence) == ["[C][O]", "[C][C]"]


def test_build_stage_prefix_and_append_stage() -> None:
    prefix = build_stage_prefix(["[C][O]"])
    completed = append_stage_to_prefix(prefix, "[C][C]", EOM_TOKEN)
    continued = append_stage_to_prefix(prefix, "[C][C]", STAGE_SEPARATOR)

    assert prefix == "<bom>[C][O]<eom> "
    assert completed == "<bom>[C][O]<eom> <bom>[C][C]<eom>"
    assert continued == "<bom>[C][O]<eom> <bom>[C][C]<eom> "


def test_parse_single_staged_molecule_requires_wrappers() -> None:
    assert parse_single_staged_molecule("<bom>[C][O]<eom>") == "[C][O]"
    assert parse_single_staged_molecule("[C][O]") is None


def test_parse_staged_target_rejects_wrapped_words_and_unwrapped_tail() -> None:
    wrapped_plain_words = "<bom>alpha beta gamma<eom>"
    wrapped_malformed_selfies = "<bom>[C] alpha [O]<eom>"
    unwrapped_tail = "[N][C]"
    text = f"{wrapped_plain_words} {wrapped_malformed_selfies} {unwrapped_tail}"

    assert parse_single_staged_molecule(wrapped_plain_words) is None
    assert parse_single_staged_molecule(wrapped_malformed_selfies) is None
    assert parse_staged_target(text) == []


def test_parse_staged_target_rejects_valid_stage_with_unwrapped_suffix() -> None:
    text = "<bom>[C][O]<eom> trailing_text"

    assert parse_single_staged_molecule(text) is None
    assert parse_staged_target(text) == []


def test_project_sampled_stage_to_no_h_projects_explicit_hydrogens() -> None:
    tokenizer = DummyTokenizer(
        {
            1: "<bom>",
            2: "[CH",
            3: "4]",
            4: "<eom>",
            5: "[C]",
        }
    )

    projection = project_sampled_stage_to_no_h(tokenizer, "<bom>[CH4]<eom>")

    assert projection.stage_text == "<bom>[C]<eom>"
    assert projection.sampled_selfies == "[C]"
    assert projection.action_token_ids == (1, 5, 4)
    assert projection.metadata["raw_stage_text"] == "<bom>[CH4]<eom>"
    assert projection.metadata["raw_sampled_selfies"] == "[CH4]"
    assert projection.metadata["projection_applied"] is True
    assert projection.metadata["projection_changed"] is True
    assert projection.metadata["projection_failure_reason"] is None


def test_project_sampled_stage_to_no_h_returns_invalid_payload_for_invalid_stage_text() -> None:
    tokenizer = DummyTokenizer(
        {
            1: "<bom>",
            2: "ordinary",
            3: "<eom>",
        }
    )

    projection = project_sampled_stage_to_no_h(tokenizer, "<bom>ordinary<eom>")

    assert projection.stage_text == ""
    assert projection.sampled_selfies is None
    assert projection.action_token_ids == ()
    assert projection.metadata["raw_stage_text"] == "<bom>ordinary<eom>"
    assert projection.metadata["raw_sampled_selfies"] is None
    assert projection.metadata["projection_applied"] is False
    assert projection.metadata["projection_failure_reason"] == "invalid_raw_stage_text"

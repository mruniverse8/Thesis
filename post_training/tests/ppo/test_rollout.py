import pytest
import torch
from types import SimpleNamespace

pytest.importorskip("transformers")

from transformers import T5Config, T5ForConditionalGeneration

from src.constants import EOM_TOKEN

from post_training.ppo.config import RolloutGenerationConfig
from post_training.ppo.rollout import compute_action_stats, sample_stage
from post_training.shared.decoding import (
    StageTokenConstraints,
    mask_logits_to_allowed_token_ids,
)


def _compute_action_stats_naively(
    model: T5ForConditionalGeneration,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    decoder_input_ids: torch.Tensor,
    action_token_ids: list[int],
    stage_token_constraints: StageTokenConstraints | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    running_decoder_input_ids = decoder_input_ids
    total_logprob = torch.zeros((), device=input_ids.device)
    total_entropy = torch.zeros((), device=input_ids.device)

    for token_id in action_token_ids:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=running_decoder_input_ids,
            return_dict=True,
        )
        next_logits = outputs.logits[:, -1, :]
        if stage_token_constraints is not None and stage_token_constraints.enabled:
            allowed_token_ids = stage_token_constraints.allowed_token_ids_for_prefix(
                action_token_ids[: len(running_decoder_input_ids[0]) - len(decoder_input_ids[0])]
            )
            next_logits = mask_logits_to_allowed_token_ids(next_logits, allowed_token_ids)
        log_probs = torch.log_softmax(next_logits, dim=-1)
        probabilities = torch.softmax(next_logits, dim=-1)
        total_logprob = total_logprob + log_probs[0, token_id]
        entropy_terms = torch.where(
            probabilities > 0,
            probabilities * log_probs,
            torch.zeros_like(probabilities),
        )
        total_entropy = total_entropy - entropy_terms.sum(dim=-1).squeeze(0)

        next_token_tensor = torch.tensor([[token_id]], dtype=torch.long, device=input_ids.device)
        running_decoder_input_ids = torch.cat([running_decoder_input_ids, next_token_tensor], dim=1)

    return total_logprob, total_entropy


def test_compute_action_stats_matches_naive_teacher_forcing() -> None:
    torch.manual_seed(0)
    config = T5Config(
        vocab_size=32,
        d_model=16,
        d_ff=32,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        dropout_rate=0.0,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
    )
    model = T5ForConditionalGeneration(config)
    model.eval()

    input_ids = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    decoder_input_ids = torch.tensor([[0, 6, 7]], dtype=torch.long)
    action_token_ids = [8, 9, 10]

    with torch.no_grad():
        optimized = compute_action_stats(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            action_token_ids=action_token_ids,
        )
        naive = _compute_action_stats_naively(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            action_token_ids=action_token_ids,
        )

    assert torch.allclose(optimized[0], naive[0], atol=1.0e-6)
    assert torch.allclose(optimized[1], naive[1], atol=1.0e-6)


def test_compute_action_stats_returns_zeros_for_empty_action_sequence() -> None:
    config = T5Config(
        vocab_size=32,
        d_model=16,
        d_ff=32,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        dropout_rate=0.0,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
    )
    model = T5ForConditionalGeneration(config)

    input_ids = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    decoder_input_ids = torch.tensor([[0, 6, 7]], dtype=torch.long)

    logprob_sum, entropy_sum = compute_action_stats(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        action_token_ids=[],
    )

    assert logprob_sum.item() == 0.0
    assert entropy_sum.item() == 0.0


def test_compute_action_stats_matches_naive_teacher_forcing_with_constraints() -> None:
    torch.manual_seed(0)
    config = T5Config(
        vocab_size=32,
        d_model=16,
        d_ff=32,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        dropout_rate=0.0,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
    )
    model = T5ForConditionalGeneration(config)
    model.eval()

    input_ids = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    decoder_input_ids = torch.tensor([[0]], dtype=torch.long)
    action_token_ids = [6, 8, 7]
    token_constraints = StageTokenConstraints(
        bom_token_id=6,
        eom_token_id=7,
        content_token_ids=(8,),
    )

    with torch.no_grad():
        optimized = compute_action_stats(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            action_token_ids=action_token_ids,
            stage_token_constraints=token_constraints,
        )
        naive = _compute_action_stats_naively(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            action_token_ids=action_token_ids,
            stage_token_constraints=token_constraints,
        )

    assert torch.allclose(optimized[0], naive[0], atol=1.0e-6)
    assert torch.allclose(optimized[1], naive[1], atol=1.0e-6)


class DummyTokenizer:
    def __init__(self, id_to_token: dict[int, str], token_to_id: dict[str, int]) -> None:
        self.id_to_token = dict(id_to_token)
        self.token_to_id = dict(token_to_id)

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True):
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(self.id_to_token[int(token_id)] for token_id in token_ids)

    def convert_tokens_to_ids(self, token: str) -> int:
        return int(self.token_to_id[token])


def test_sample_stage_enforces_bom_and_masks_language_tokens() -> None:
    tokenizer = DummyTokenizer(
        {
            1: "<bom>",
            2: "[C]",
            3: "<eom>",
            4: "ordinary",
        },
        {EOM_TOKEN: 3},
    )
    token_constraints = StageTokenConstraints(
        bom_token_id=1,
        eom_token_id=3,
        content_token_ids=(2,),
    )

    class DummyPolicyModel:
        def __init__(self) -> None:
            self.policy_model = self
            self._stage_token_constraints = token_constraints

        def get_stage_token_constraints(self):
            return self._stage_token_constraints

        def __call__(
            self,
            *,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            decoder_input_ids: torch.Tensor,
            return_dict: bool,
        ) -> SimpleNamespace:
            del input_ids, attention_mask, return_dict
            logits = torch.full((1, decoder_input_ids.size(1), 8), -20.0)
            current_length = decoder_input_ids.size(1)
            if current_length == 1:
                logits[:, -1, 4] = 10.0
                logits[:, -1, 1] = 0.0
            elif current_length == 2:
                logits[:, -1, 4] = 10.0
                logits[:, -1, 2] = 0.0
            else:
                logits[:, -1, 4] = 10.0
                logits[:, -1, 3] = 6.0
                logits[:, -1, 2] = 0.0
            return SimpleNamespace(logits=logits)

    stage_sample = sample_stage(
        DummyPolicyModel(),
        tokenizer,
        input_ids=torch.tensor([[1, 2]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1]], dtype=torch.long),
        decoder_prefix_ids=torch.tensor([[0]], dtype=torch.long),
        generation_config=RolloutGenerationConfig(max_stage_new_tokens=4),
    )

    assert stage_sample["action_token_ids"] == [1, 2, 3]
    assert stage_sample["stop_token"] == EOM_TOKEN
    assert stage_sample["termination_reason"] == "stop_token"
    assert stage_sample["stage_text"] == "<bom>[C]<eom>"
    assert stage_sample["sampled_selfies"] == "[C]"

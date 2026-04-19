import pytest
import torch

pytest.importorskip("transformers")

from transformers import T5Config, T5ForConditionalGeneration

from post_training.ppo.rollout import compute_action_stats


def _compute_action_stats_naively(
    model: T5ForConditionalGeneration,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    decoder_input_ids: torch.Tensor,
    action_token_ids: list[int],
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
        log_probs = torch.log_softmax(next_logits, dim=-1)
        probabilities = torch.softmax(next_logits, dim=-1)
        total_logprob = total_logprob + log_probs[0, token_id]
        total_entropy = total_entropy - (probabilities * log_probs).sum(dim=-1).squeeze(0)

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

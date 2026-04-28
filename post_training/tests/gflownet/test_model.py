import pytest
import torch

pytest.importorskip("transformers")

from transformers import T5Config, T5ForConditionalGeneration

from post_training.gflownet.model import GFlowNetModel
from post_training.shared.decoding import StageTokenConstraints


def _build_tiny_gflownet_model(*, vocab_size: int = 32) -> GFlowNetModel:
    config = T5Config(
        vocab_size=vocab_size,
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
    return GFlowNetModel(T5ForConditionalGeneration(config))


def _assert_score_rows_close(
    left: tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]],
    right: tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]],
) -> None:
    assert [len(group) for group in left] == [len(group) for group in right]
    for left_group, right_group in zip(left, right):
        for left_value, right_value in zip(left_group, right_group):
            assert torch.allclose(left_value, right_value, atol=1.0e-6)


def test_score_action_sequences_matches_repeated_single_scoring_with_padding() -> None:
    torch.manual_seed(1)
    model = _build_tiny_gflownet_model()
    model.eval()

    input_ids = torch.tensor(
        [
            [2, 3, 4, 0],
            [2, 5, 6, 7],
            [2, 8, 0, 0],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 0, 0],
        ],
        dtype=torch.long,
    )
    decoder_prefix_ids = (
        torch.tensor([[0]], dtype=torch.long),
        torch.tensor([[0, 9]], dtype=torch.long),
        torch.tensor([[0]], dtype=torch.long),
    )
    action_token_ids = ((6, 8), (10,), (6, 8, 10, 11))

    with torch.no_grad():
        batched_scores = model.score_action_sequences(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_prefix_ids=decoder_prefix_ids,
            action_token_ids=action_token_ids,
            stop_token_id=7,
        )
        single_scores = [
            model.score_action_sequence(
                input_ids=input_ids[index : index + 1],
                attention_mask=attention_mask[index : index + 1],
                decoder_prefix_ids=decoder_prefix_ids[index],
                action_token_ids=action_token_ids[index],
                stop_token_id=7,
            )
            for index in range(len(action_token_ids))
        ]

    assert len(batched_scores) == 3
    assert [len(score[0]) for score in batched_scores] == [2, 1, 4]
    assert [len(score[1]) for score in batched_scores] == [3, 2, 5]
    assert [len(score[2]) for score in batched_scores] == [3, 2, 5]
    for batched_score, single_score in zip(batched_scores, single_scores):
        _assert_score_rows_close(batched_score, single_score)


def test_score_action_sequences_backpropagates_through_policy_and_flow_head() -> None:
    torch.manual_seed(2)
    model = _build_tiny_gflownet_model()

    scores = model.score_action_sequences(
        input_ids=torch.tensor([[2, 3, 4], [2, 5, 6]], dtype=torch.long),
        attention_mask=torch.ones((2, 3), dtype=torch.long),
        decoder_prefix_ids=(
            torch.tensor([[0]], dtype=torch.long),
            torch.tensor([[0]], dtype=torch.long),
        ),
        action_token_ids=((6, 8), (6, 8, 10)),
        stop_token_id=7,
    )
    loss_terms = [value for score in scores for group in score for value in group]
    torch.stack(loss_terms).sum().backward()

    policy_grads = [
        parameter.grad
        for parameter in model.policy_model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    flow_head_grads = [
        parameter.grad
        for parameter in model.flow_head.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert any(torch.count_nonzero(grad).item() > 0 for grad in policy_grads)
    assert any(torch.count_nonzero(grad).item() > 0 for grad in flow_head_grads)


def test_score_action_sequence_applies_stage_token_constraints_to_stop_logprobs() -> None:
    torch.manual_seed(0)
    model = _build_tiny_gflownet_model()
    model.eval()

    input_ids = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    decoder_prefix_ids = torch.tensor([[0]], dtype=torch.long)
    action_token_ids = (6, 8)
    token_constraints = StageTokenConstraints(
        bom_token_id=6,
        eom_token_id=7,
        content_token_ids=(8,),
    )
    model.set_stage_token_constraints(token_constraints)

    with torch.no_grad():
        log_pf_tokens, log_stop, log_state_flows = model.score_action_sequence(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_prefix_ids=decoder_prefix_ids,
            action_token_ids=action_token_ids,
            stop_token_id=7,
        )

    assert len(log_pf_tokens) == 2
    assert len(log_stop) == 3
    assert len(log_state_flows) == 3
    assert torch.isneginf(log_stop[0])
    assert torch.isneginf(log_stop[1])
    assert torch.isfinite(log_stop[2])

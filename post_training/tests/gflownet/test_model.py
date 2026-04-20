import pytest
import torch

pytest.importorskip("transformers")

from transformers import T5Config, T5ForConditionalGeneration

from post_training.gflownet.model import GFlowNetModel
from post_training.shared.decoding import StageTokenConstraints


def test_score_action_sequence_applies_stage_token_constraints_to_stop_logprobs() -> None:
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
    model = GFlowNetModel(T5ForConditionalGeneration(config))
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

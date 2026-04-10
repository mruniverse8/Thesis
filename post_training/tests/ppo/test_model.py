import pytest
import torch

pytest.importorskip("transformers")

from transformers import T5Config, T5ForConditionalGeneration

from post_training.ppo.model import PolicyValueModel, gather_last_token_hidden_state


def test_gather_last_token_hidden_state_uses_attention_mask() -> None:
    hidden_state = torch.tensor(
        [
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]],
        ]
    )
    attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    gathered = gather_last_token_hidden_state(hidden_state, attention_mask)

    assert torch.equal(gathered, torch.tensor([[2.0, 2.0], [6.0, 6.0]]))


def test_policy_value_model_computes_scalar_values() -> None:
    config = T5Config(
        vocab_size=32,
        d_model=16,
        d_ff=32,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
    )
    wrapper = PolicyValueModel(T5ForConditionalGeneration(config))

    input_ids = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    decoder_input_ids = torch.tensor([[0, 6, 7]], dtype=torch.long)
    values = wrapper.compute_values(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
    )

    assert values.shape == (1,)

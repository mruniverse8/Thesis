import pytest
import torch

pytest.importorskip("transformers")

from transformers import T5Config, T5ForConditionalGeneration

from post_training.ppo.model import (
    PolicyValueModel,
    gather_last_token_hidden_state,
    load_reference_model,
)


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


def test_policy_value_model_from_pretrained_does_not_resize_embeddings(monkeypatch) -> None:
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
    model = T5ForConditionalGeneration(config)

    def fail_resize(*args, **kwargs):
        raise AssertionError("resize_token_embeddings should not be called")

    model.resize_token_embeddings = fail_resize
    monkeypatch.setattr(
        "post_training.ppo.model.T5ForConditionalGeneration.from_pretrained",
        lambda *args, **kwargs: model,
    )

    wrapper = PolicyValueModel.from_pretrained("demo-checkpoint", use_lora=False)

    assert wrapper.policy_model is model


def test_load_reference_model_does_not_resize_embeddings(monkeypatch) -> None:
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
    model = T5ForConditionalGeneration(config)

    def fail_resize(*args, **kwargs):
        raise AssertionError("resize_token_embeddings should not be called")

    model.resize_token_embeddings = fail_resize
    monkeypatch.setattr(
        "post_training.ppo.model.T5ForConditionalGeneration.from_pretrained",
        lambda *args, **kwargs: model,
    )

    loaded = load_reference_model("demo-checkpoint")

    assert loaded is model
    assert all(not parameter.requires_grad for parameter in loaded.parameters())

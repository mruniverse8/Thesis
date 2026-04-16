import json
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")

from transformers import T5Config, T5ForConditionalGeneration

from post_training.ppo.model import (
    PolicyValueModel,
    gather_last_token_hidden_state,
    load_reference_model,
)


class DummyTokenizer:
    def save_pretrained(self, output_dir: str | Path) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer.json").write_text("ok\n", encoding="utf-8")
        (path / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")


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

def test_policy_value_model_from_pretrained_loads_value_head_state(monkeypatch, tmp_path: Path) -> None:
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
    monkeypatch.setattr(
        "post_training.ppo.model.T5ForConditionalGeneration.from_pretrained",
        lambda *args, **kwargs: model,
    )

    expected = PolicyValueModel(T5ForConditionalGeneration(config))
    expected.value_head.projection.weight.data.fill_(0.5)
    expected.value_head.projection.bias.data.fill_(0.25)
    torch.save(expected.value_head.state_dict(), tmp_path / "value_head.pt")

    loaded = PolicyValueModel.from_pretrained(tmp_path, use_lora=False)

    assert torch.equal(loaded.value_head.projection.weight, expected.value_head.projection.weight)
    assert torch.equal(loaded.value_head.projection.bias, expected.value_head.projection.bias)


def test_policy_value_model_save_checkpoint_preserves_hf_config_and_writes_archive(tmp_path: Path) -> None:
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
    checkpoint_dir = tmp_path / "best"

    archive_path = wrapper.save_checkpoint(
        checkpoint_dir,
        tokenizer=DummyTokenizer(),
        config={"seed": 42},
        metrics={"loss": 1.0},
        create_archive=True,
    )

    hf_config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    training_config = json.loads((checkpoint_dir / "training_config.json").read_text(encoding="utf-8"))

    assert hf_config["vocab_size"] == 32
    assert training_config == {"seed": 42}
    assert (checkpoint_dir / "value_head.pt").exists()
    assert archive_path is not None and archive_path.exists()


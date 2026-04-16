from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from transformers import T5ForConditionalGeneration

from src.io_utils import ensure_dir, write_json
from src.tokenizer_utils import assert_tokenizer_matches_model_vocab


def freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def load_reference_model(
    checkpoint_path: str | Path,
) -> T5ForConditionalGeneration:
    model = T5ForConditionalGeneration.from_pretrained(checkpoint_path)
    freeze_module(model)
    model.eval()
    return model


def _apply_lora_adapters(
    model: T5ForConditionalGeneration,
    *,
    rank: int,
    alpha: int,
    dropout: float,
    target_modules: Sequence[str],
) -> T5ForConditionalGeneration:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError(
            "LoRA adapters require `peft`. Install it before running PPO training."
        ) from exc

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(target_modules),
    )
    return get_peft_model(model, lora_config)


def gather_last_token_hidden_state(
    decoder_hidden_state: torch.Tensor,
    decoder_attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if decoder_attention_mask is None:
        return decoder_hidden_state[:, -1, :]

    last_indices = decoder_attention_mask.long().sum(dim=1).clamp(min=1) - 1
    batch_indices = torch.arange(
        decoder_hidden_state.size(0),
        device=decoder_hidden_state.device,
    )
    return decoder_hidden_state[batch_indices, last_indices]


class ScalarValueHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, 1)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.projection(hidden_state).squeeze(-1)


class PolicyValueModel(nn.Module):
    def __init__(
        self,
        policy_model: T5ForConditionalGeneration,
        value_head: ScalarValueHead | None = None,
    ) -> None:
        super().__init__()
        self.policy_model = policy_model
        self.value_head = value_head or ScalarValueHead(policy_model.config.d_model)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | Path,
        *,
        use_lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Sequence[str] = ("q", "v"),
        freeze_base_model_without_lora: bool = False,
    ) -> "PolicyValueModel":
        model = T5ForConditionalGeneration.from_pretrained(checkpoint_path)

        if use_lora:
            model = _apply_lora_adapters(
                model,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                target_modules=target_modules,
            )
        elif freeze_base_model_without_lora:
            freeze_module(model)

        return cls(policy_model=model)

    def forward(self, **kwargs: Any):
        return self.policy_model(**kwargs)

    def generate(self, *args: Any, **kwargs: Any):
        return self.policy_model.generate(*args, **kwargs)

    def compute_values(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.policy_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        final_decoder_state = outputs.decoder_hidden_states[-1]
        pooled_state = gather_last_token_hidden_state(
            final_decoder_state,
            decoder_attention_mask=decoder_attention_mask,
        )
        return self.value_head(pooled_state)

    def save_checkpoint(
        self,
        output_dir: str | Path,
        *,
        tokenizer: Any | None = None,
        config: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        checkpoint_dir = ensure_dir(output_dir)
        self.policy_model.save_pretrained(checkpoint_dir)
        torch.save(self.value_head.state_dict(), checkpoint_dir / "value_head.pt")
        if tokenizer is not None:
            tokenizer.save_pretrained(checkpoint_dir)
        if config is not None:
            write_json(checkpoint_dir / "config.json", config)
        if metrics is not None:
            write_json(checkpoint_dir / "metrics.json", metrics)


def assert_checkpoint_tokenizer_matches_model(
    tokenizer: Any,
    model: T5ForConditionalGeneration,
    *,
    context: str,
) -> None:
    assert_tokenizer_matches_model_vocab(tokenizer, model, context=context)

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from transformers import T5ForConditionalGeneration

from src.checkpoint_bootstrap import archive_checkpoint_directory
from src.io_utils import ensure_dir, write_json
from src.tokenizer_utils import assert_tokenizer_matches_model_vocab


def freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


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
            "LoRA adapters require `peft`. Install it before running GFlowNet training."
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


class ScalarFlowHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, 1)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.projection(hidden_state).squeeze(-1)


class GFlowNetModel(nn.Module):
    def __init__(
        self,
        policy_model: T5ForConditionalGeneration,
        flow_head: ScalarFlowHead | None = None,
    ) -> None:
        super().__init__()
        self.policy_model = policy_model
        self.flow_head = flow_head or ScalarFlowHead(policy_model.config.d_model)

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
    ) -> "GFlowNetModel":
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

        flow_head = ScalarFlowHead(model.config.d_model)
        flow_head_path = Path(checkpoint_path) / "flow_head.pt"
        if flow_head_path.exists():
            flow_head.load_state_dict(torch.load(flow_head_path, map_location="cpu"))

        return cls(policy_model=model, flow_head=flow_head)

    def forward(self, **kwargs: Any):
        return self.policy_model(**kwargs)

    def generate(self, *args: Any, **kwargs: Any):
        return self.policy_model.generate(*args, **kwargs)

    def score_action_sequence(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_prefix_ids: torch.Tensor,
        action_token_ids: Sequence[int],
        stop_token_id: int,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        if decoder_prefix_ids.ndim != 2 or decoder_prefix_ids.size(0) != 1:
            raise ValueError("decoder_prefix_ids must have shape [1, prefix_length].")

        if action_token_ids:
            action_tokens = torch.tensor(
                [[int(token_id) for token_id in action_token_ids]],
                dtype=torch.long,
                device=input_ids.device,
            )
            decoder_input_ids = torch.cat([decoder_prefix_ids, action_tokens], dim=1)
        else:
            action_tokens = torch.empty((1, 0), dtype=torch.long, device=input_ids.device)
            decoder_input_ids = decoder_prefix_ids

        outputs = self.policy_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True,
            return_dict=True,
        )

        log_probs = torch.log_softmax(outputs.logits[0], dim=-1)
        action_start_index = decoder_prefix_ids.size(1) - 1
        stop_slice = log_probs[
            action_start_index : action_start_index + len(action_token_ids) + 1,
            stop_token_id,
        ]
        stop_log_probs = tuple(stop_slice.unbind())

        action_log_probs: tuple[torch.Tensor, ...]
        if action_token_ids:
            action_token_tensor = torch.tensor(
                [int(token_id) for token_id in action_token_ids],
                dtype=torch.long,
                device=input_ids.device,
            )
            action_slice = log_probs[action_start_index : action_start_index + len(action_token_ids)]
            selected = action_slice.gather(-1, action_token_tensor.unsqueeze(-1)).squeeze(-1)
            action_log_probs = tuple(selected.unbind())
        else:
            action_log_probs = ()

        final_decoder_state = outputs.decoder_hidden_states[-1][0][
            action_start_index : action_start_index + len(action_token_ids) + 1
        ]
        prefix_log_flows = tuple(self.flow_head(final_decoder_state).unbind())
        return action_log_probs, stop_log_probs, prefix_log_flows

    def save_checkpoint(
        self,
        output_dir: str | Path,
        *,
        tokenizer: Any | None = None,
        config: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        create_archive: bool = False,
    ) -> Path | None:
        checkpoint_dir = ensure_dir(output_dir)
        self.policy_model.save_pretrained(checkpoint_dir)
        torch.save(self.flow_head.state_dict(), checkpoint_dir / "flow_head.pt")
        if tokenizer is not None:
            tokenizer.save_pretrained(checkpoint_dir)
        if config is not None:
            write_json(checkpoint_dir / "training_config.json", config)
        if metrics is not None:
            write_json(checkpoint_dir / "metrics.json", metrics)
        if create_archive:
            return archive_checkpoint_directory(checkpoint_dir)
        return None


def assert_checkpoint_tokenizer_matches_model(
    tokenizer: Any,
    model: T5ForConditionalGeneration,
    *,
    context: str,
) -> None:
    assert_tokenizer_matches_model_vocab(tokenizer, model, context=context)

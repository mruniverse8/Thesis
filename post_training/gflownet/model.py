from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from transformers import T5ForConditionalGeneration

from post_training.shared.decoding import (
    StageTokenConstraints,
    mask_logits_to_allowed_token_ids,
)
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
        self._stage_token_constraints: StageTokenConstraints | None = None

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

    def set_stage_token_constraints(
        self,
        stage_token_constraints: StageTokenConstraints | None,
    ) -> None:
        self._stage_token_constraints = stage_token_constraints

    def get_stage_token_constraints(self) -> StageTokenConstraints | None:
        return self._stage_token_constraints

    def _decoder_pad_token_id(self) -> int:
        pad_token_id = getattr(self.policy_model.config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.policy_model.config, "decoder_start_token_id", 0)
        return int(pad_token_id)

    def score_action_sequences(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_prefix_ids: Sequence[torch.Tensor],
        action_token_ids: Sequence[Sequence[int]],
        stop_token_id: int,
        stage_token_constraints: StageTokenConstraints | None = None,
    ) -> list[
        tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]
    ]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch_size, source_length].")
        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape [batch_size, source_length].")
        batch_size = int(input_ids.size(0))
        if int(attention_mask.size(0)) != batch_size:
            raise ValueError("attention_mask batch size must match input_ids.")
        if len(decoder_prefix_ids) != batch_size:
            raise ValueError("decoder_prefix_ids length must match input_ids batch size.")
        if len(action_token_ids) != batch_size:
            raise ValueError("action_token_ids length must match input_ids batch size.")
        if batch_size == 0:
            return []

        resolved_constraints = (
            stage_token_constraints
            if stage_token_constraints is not None
            else self.get_stage_token_constraints()
        )
        device = input_ids.device
        decoder_sequences: list[torch.Tensor] = []
        decoder_prefix_lengths: list[int] = []
        normalized_action_token_ids: list[tuple[int, ...]] = []

        for row_index, prefix_ids in enumerate(decoder_prefix_ids):
            prefix_tensor = prefix_ids.to(device=device, dtype=torch.long)
            if prefix_tensor.ndim == 2:
                if prefix_tensor.size(0) != 1:
                    raise ValueError(
                        "Each decoder prefix tensor must have shape [1, prefix_length]."
                    )
                prefix_tensor = prefix_tensor.squeeze(0)
            elif prefix_tensor.ndim != 1:
                raise ValueError(
                    "Each decoder prefix tensor must have shape [prefix_length] "
                    "or [1, prefix_length]."
                )
            if prefix_tensor.numel() < 1:
                raise ValueError("Each decoder prefix must include a decoder start token.")

            row_action_token_ids = tuple(
                int(token_id) for token_id in action_token_ids[row_index]
            )
            action_tensor = torch.tensor(
                row_action_token_ids,
                dtype=torch.long,
                device=device,
            )
            decoder_sequences.append(torch.cat([prefix_tensor, action_tensor], dim=0))
            decoder_prefix_lengths.append(int(prefix_tensor.numel()))
            normalized_action_token_ids.append(row_action_token_ids)

        pad_token_id = self._decoder_pad_token_id()
        max_decoder_length = max(int(sequence.numel()) for sequence in decoder_sequences)
        decoder_input_ids = torch.full(
            (batch_size, max_decoder_length),
            pad_token_id,
            dtype=torch.long,
            device=device,
        )
        decoder_attention_mask = torch.zeros_like(decoder_input_ids)
        for row_index, sequence in enumerate(decoder_sequences):
            sequence_length = int(sequence.numel())
            decoder_input_ids[row_index, :sequence_length] = sequence
            decoder_attention_mask[row_index, :sequence_length] = 1

        outputs = self.policy_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        final_decoder_hidden_states = outputs.decoder_hidden_states[-1]
        scored_rows: list[
            tuple[
                tuple[torch.Tensor, ...],
                tuple[torch.Tensor, ...],
                tuple[torch.Tensor, ...],
            ]
        ] = []
        for row_index, row_action_token_ids in enumerate(normalized_action_token_ids):
            action_start_index = decoder_prefix_lengths[row_index] - 1
            num_positions = len(row_action_token_ids) + 1
            position_logits = outputs.logits[row_index][
                action_start_index : action_start_index + num_positions
            ]
            stop_log_probs: list[torch.Tensor] = []
            action_log_probs: list[torch.Tensor] = []
            for position, raw_logits in enumerate(position_logits):
                step_logits = raw_logits
                if resolved_constraints is not None and resolved_constraints.enabled:
                    allowed_token_ids = resolved_constraints.allowed_token_ids_for_prefix(
                        row_action_token_ids[:position]
                    )
                    step_logits = mask_logits_to_allowed_token_ids(
                        step_logits.unsqueeze(0),
                        allowed_token_ids,
                    ).squeeze(0)
                step_log_probs = torch.log_softmax(step_logits, dim=-1)
                stop_log_probs.append(step_log_probs[int(stop_token_id)])
                if position < len(row_action_token_ids):
                    action_log_probs.append(
                        step_log_probs[int(row_action_token_ids[position])]
                    )

            decoder_states = final_decoder_hidden_states[row_index][
                action_start_index : action_start_index + num_positions
            ]
            prefix_log_flows = tuple(self.flow_head(decoder_states).unbind())
            scored_rows.append(
                (tuple(action_log_probs), tuple(stop_log_probs), prefix_log_flows)
            )
        return scored_rows

    def score_action_sequence(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_prefix_ids: torch.Tensor,
        action_token_ids: Sequence[int],
        stop_token_id: int,
        stage_token_constraints: StageTokenConstraints | None = None,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        scored_rows = self.score_action_sequences(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_prefix_ids=(decoder_prefix_ids,),
            action_token_ids=(action_token_ids,),
            stop_token_id=stop_token_id,
            stage_token_constraints=stage_token_constraints,
        )
        return scored_rows[0]

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

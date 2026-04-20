from __future__ import annotations

from typing import Any, Sequence

import torch
from transformers import PreTrainedTokenizerBase

from reward_utils.defaults import RewardConfig
from src.constants import EOM_TOKEN

from post_training.shared.decoding import (
    StageTokenConstraints,
    mask_logits_to_allowed_token_ids,
    resolve_stage_token_constraints,
)
from post_training.shared.sequence import build_stage_prefix, parse_single_staged_molecule

from .config import GFlowNetRolloutConfig
from .model import GFlowNetModel
from .rewarding import score_stage_terminal_reward
from .trajectory import SampledStageTrajectory


def encode_prompt(
    tokenizer: PreTrainedTokenizerBase,
    prompt_text: str,
    *,
    max_source_length: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_source_length,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in encoded.items()}


def encode_decoder_prefix(
    tokenizer: PreTrainedTokenizerBase,
    prefix_text: str,
    *,
    decoder_start_token_id: int,
    device: torch.device,
) -> torch.Tensor:
    if prefix_text:
        prefix_ids = tokenizer(
            prefix_text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(device)
    else:
        prefix_ids = torch.empty((1, 0), dtype=torch.long, device=device)

    start_tensor = torch.tensor([[decoder_start_token_id]], device=device)
    return torch.cat([start_tensor, prefix_ids], dim=1)


def _top_p_sample(probabilities: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return torch.multinomial(probabilities, num_samples=1)

    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    cutoff_mask = cumulative_probs > top_p
    cutoff_mask[..., 1:] = cutoff_mask[..., :-1].clone()
    cutoff_mask[..., 0] = False
    sorted_probs = sorted_probs.masked_fill(cutoff_mask, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    sampled_in_sorted = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_indices.gather(-1, sampled_in_sorted)


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    action_token_ids: Sequence[int] = (),
    stage_token_constraints: StageTokenConstraints | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    adjusted_logits = logits / max(temperature, 1.0e-6)
    if stage_token_constraints is not None and stage_token_constraints.enabled:
        allowed_token_ids = stage_token_constraints.allowed_token_ids_for_prefix(action_token_ids)
        adjusted_logits = mask_logits_to_allowed_token_ids(adjusted_logits, allowed_token_ids)
    probabilities = torch.softmax(adjusted_logits, dim=-1)
    next_token = _top_p_sample(probabilities, top_p=top_p)
    log_probs = torch.log_softmax(adjusted_logits, dim=-1)
    next_log_prob = log_probs.gather(-1, next_token).squeeze(-1)
    entropy_terms = torch.where(
        probabilities > 0,
        probabilities * log_probs,
        torch.zeros_like(probabilities),
    )
    entropy = -entropy_terms.sum(dim=-1)
    return next_token, next_log_prob, entropy


def build_sampled_stage_trajectory_from_generation(
    *,
    example: dict[str, Any],
    rollout_id: str,
    stage_index: int,
    decoder_prefix_text: str,
    previous_valid_selfies: Sequence[str],
    tokenizer: PreTrainedTokenizerBase,
    action_token_ids: Sequence[int],
    stop_token: str | None,
    termination_reason: str,
    reward_config: RewardConfig | None,
    invalid_terminal_reward: float,
) -> SampledStageTrajectory:
    prompt_text = str(example["prompt"])
    description = str(example["description"])
    target_selfies_list = tuple(str(item) for item in example["target_selfies_list"])

    stage_token_ids = list(int(token_id) for token_id in action_token_ids)
    if stop_token == EOM_TOKEN:
        eom_token_id = tokenizer.convert_tokens_to_ids(EOM_TOKEN)
        stage_token_ids.append(int(eom_token_id))

    stage_text = (
        tokenizer.decode(
            stage_token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True,
        ).strip()
        if stage_token_ids
        else ""
    )
    sampled_selfies = parse_single_staged_molecule(stage_text)
    reward_summary = score_stage_terminal_reward(
        sampled_selfies,
        targets=target_selfies_list,
        previous_candidates=tuple(str(item) for item in previous_valid_selfies),
        num_prefix_states=len(action_token_ids) + 1,
        reward_config=reward_config,
        invalid_terminal_reward=invalid_terminal_reward,
    )
    return SampledStageTrajectory(
        rollout_id=rollout_id,
        example_id=str(example["id"]),
        prompt_text=prompt_text,
        description=description,
        target_selfies_list=target_selfies_list,
        stage_index=stage_index,
        decoder_prefix_text=decoder_prefix_text,
        previous_valid_selfies=tuple(str(item) for item in previous_valid_selfies),
        stage_text=stage_text,
        sampled_selfies=sampled_selfies,
        action_token_ids=tuple(int(token_id) for token_id in action_token_ids),
        reward_breakdown=reward_summary.reward_breakdown,
        prefix_rewards=reward_summary.prefix_rewards,
        terminal_reward=reward_summary.terminal_reward,
        stop_token=stop_token,
        termination_reason=termination_reason,
        is_valid=reward_summary.is_valid_terminal,
        is_duplicate=reward_summary.is_duplicate_terminal,
        metadata={
            "num_actions": len(action_token_ids),
            "stop_action_token": EOM_TOKEN,
        },
    )


def sample_stage(
    model: GFlowNetModel,
    tokenizer: PreTrainedTokenizerBase,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    decoder_prefix_ids: torch.Tensor,
    generation_config: GFlowNetRolloutConfig,
    stage_token_constraints: StageTokenConstraints | None = None,
) -> dict[str, Any]:
    device = input_ids.device
    eom_token_id = tokenizer.convert_tokens_to_ids(EOM_TOKEN)
    eos_token_id = model.policy_model.config.eos_token_id
    resolved_constraints = resolve_stage_token_constraints(model, stage_token_constraints)

    current_decoder_input_ids = decoder_prefix_ids.clone()
    action_token_ids: list[int] = []
    stop_token: str | None = None
    termination_reason = "max_stage_new_tokens"

    with torch.no_grad():
        for _ in range(generation_config.max_stage_new_tokens):
            total_length = input_ids.size(1) + current_decoder_input_ids.size(1)
            if total_length >= generation_config.max_sequence_length:
                termination_reason = "max_sequence_length"
                break

            outputs = model.policy_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=current_decoder_input_ids,
                return_dict=True,
            )
            next_logits = outputs.logits[:, -1, :]
            next_token, _, _ = sample_next_token(
                next_logits,
                temperature=generation_config.temperature,
                top_p=generation_config.top_p,
                action_token_ids=tuple(action_token_ids),
                stage_token_constraints=resolved_constraints,
            )
            next_token_id = int(next_token.item())

            if next_token_id == int(eom_token_id):
                stop_token = EOM_TOKEN
                termination_reason = "stop_token"
                break

            if eos_token_id is not None and next_token_id == int(eos_token_id):
                stop_token = tokenizer.eos_token
                termination_reason = "eos_token"
                break

            action_token_ids.append(next_token_id)
            current_decoder_input_ids = torch.cat([current_decoder_input_ids, next_token], dim=1)

    stage_token_ids = [*action_token_ids]
    if stop_token == EOM_TOKEN:
        stage_token_ids.append(int(eom_token_id))
    stage_text = (
        tokenizer.decode(
            stage_token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True,
        ).strip()
        if stage_token_ids
        else ""
    )

    return {
        "stage_text": stage_text,
        "sampled_selfies": parse_single_staged_molecule(stage_text),
        "action_token_ids": tuple(action_token_ids),
        "stop_token": stop_token,
        "termination_reason": termination_reason,
    }


def sample_stage_trajectories_for_example(
    model: GFlowNetModel,
    tokenizer: PreTrainedTokenizerBase,
    example: dict[str, Any],
    *,
    rollout_id: str,
    generation_config: GFlowNetRolloutConfig,
    reward_config: RewardConfig | None = None,
    invalid_terminal_reward: float = 1.0e-4,
    device: torch.device,
    stage_token_constraints: StageTokenConstraints | None = None,
) -> list[SampledStageTrajectory]:
    prompt_inputs = encode_prompt(
        tokenizer,
        str(example["prompt"]),
        max_source_length=generation_config.max_source_length,
        device=device,
    )
    decoder_start_token_id = int(model.policy_model.config.decoder_start_token_id)
    target_selfies_list = tuple(str(item) for item in example["target_selfies_list"])
    planned_stage_count = max(
        1,
        min(generation_config.max_molecules_per_sequence, len(target_selfies_list)),
    )

    previous_valid_selfies: list[str] = []
    trajectories: list[SampledStageTrajectory] = []
    for stage_index in range(1, planned_stage_count + 1):
        prefix_text = build_stage_prefix(
            previous_valid_selfies,
            separator_token=generation_config.stage_separator,
        )
        decoder_prefix_ids = encode_decoder_prefix(
            tokenizer,
            prefix_text,
            decoder_start_token_id=decoder_start_token_id,
            device=device,
        )
        stage_sample = sample_stage(
            model,
            tokenizer,
            input_ids=prompt_inputs["input_ids"],
            attention_mask=prompt_inputs["attention_mask"],
            decoder_prefix_ids=decoder_prefix_ids,
            generation_config=generation_config,
            stage_token_constraints=stage_token_constraints,
        )
        trajectory = build_sampled_stage_trajectory_from_generation(
            example=example,
            rollout_id=rollout_id,
            stage_index=stage_index,
            decoder_prefix_text=prefix_text,
            previous_valid_selfies=tuple(previous_valid_selfies),
            tokenizer=tokenizer,
            action_token_ids=stage_sample["action_token_ids"],
            stop_token=stage_sample["stop_token"],
            termination_reason=stage_sample["termination_reason"],
            reward_config=reward_config,
            invalid_terminal_reward=invalid_terminal_reward,
        )
        trajectories.append(trajectory)

        if trajectory.sampled_selfies and trajectory.is_valid:
            previous_valid_selfies.append(trajectory.sampled_selfies)

        if trajectory.termination_reason != "stop_token":
            break
        if generation_config.terminate_on_invalid_stage and not trajectory.is_valid:
            break

    return trajectories

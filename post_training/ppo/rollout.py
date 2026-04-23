from __future__ import annotations

from typing import Any, Sequence

import torch
from transformers import PreTrainedTokenizerBase, T5ForConditionalGeneration

from reward_utils.defaults import RewardConfig
from src.constants import EOM_TOKEN

from post_training.shared.decoding import (
    StageTokenConstraints,
    mask_logits_to_allowed_token_ids,
    resolve_stage_token_constraints,
)
from post_training.shared.sequence import build_stage_prefix, project_sampled_stage_to_no_h

from .config import RolloutGenerationConfig, StageTrajectory
from .model import PolicyValueModel
from .rewarding import score_stage_reward


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


def compute_action_stats(
    model: T5ForConditionalGeneration | PolicyValueModel,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    decoder_input_ids: torch.Tensor,
    action_token_ids: Sequence[int],
    stage_token_constraints: StageTokenConstraints | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not action_token_ids:
        zero = torch.zeros((), device=input_ids.device)
        return zero, zero
    resolved_constraints = resolve_stage_token_constraints(model, stage_token_constraints)

    # Shape actions as a single decoder continuation: [a_1, a_2, ..., a_T].
    action_tokens = torch.tensor(
        [[int(token_id) for token_id in action_token_ids]],
        dtype=torch.long,
        device=input_ids.device,
    )

    # Teacher forcing lets us score the whole sampled action in one forward pass.
    # To predict token a_t, the decoder should see the original prefix plus
    # all previously chosen action tokens [a_1, ..., a_{t-1}].
    teacher_forced_decoder_input_ids = decoder_input_ids
    if action_tokens.size(1) > 1:
        teacher_forced_decoder_input_ids = torch.cat(
            [decoder_input_ids, action_tokens[:, :-1]],
            dim=1,
        )

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=teacher_forced_decoder_input_ids,
        return_dict=True,
    )

    # The logits returned for the decoder positions align with:
    #   prefix_last_position -> predicts a_1
    #   a_1 position         -> predicts a_2
    #   ...
    # so we slice from the last prefix position onward to get exactly T action logits.
    action_start_index = decoder_input_ids.size(1) - 1
    action_logits = outputs.logits[:, action_start_index:, :]
    selected_log_probs: list[torch.Tensor] = []
    token_entropies: list[torch.Tensor] = []
    for position, token_id in enumerate(action_token_ids):
        step_logits = action_logits[:, position, :]
        if resolved_constraints is not None and resolved_constraints.enabled:
            allowed_token_ids = resolved_constraints.allowed_token_ids_for_prefix(
                action_token_ids[:position]
            )
            step_logits = mask_logits_to_allowed_token_ids(step_logits, allowed_token_ids)
        step_log_probs = torch.log_softmax(step_logits, dim=-1)
        step_probabilities = torch.softmax(step_logits, dim=-1)
        selected_log_probs.append(step_log_probs[:, int(token_id)])
        entropy_terms = torch.where(
            step_probabilities > 0,
            step_probabilities * step_log_probs,
            torch.zeros_like(step_probabilities),
        )
        token_entropies.append(-entropy_terms.sum(dim=-1))
    return torch.stack(selected_log_probs).sum(), torch.stack(token_entropies).sum()


def sample_stage(
    policy_model: PolicyValueModel,
    tokenizer: PreTrainedTokenizerBase,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    decoder_prefix_ids: torch.Tensor,
    generation_config: RolloutGenerationConfig,
    stage_token_constraints: StageTokenConstraints | None = None,
) -> dict[str, Any]:
    eom_token_id = tokenizer.convert_tokens_to_ids(EOM_TOKEN)
    resolved_constraints = resolve_stage_token_constraints(policy_model, stage_token_constraints)

    current_decoder_input_ids = decoder_prefix_ids.clone()
    raw_action_token_ids: list[int] = []
    raw_total_logprob = 0.0
    raw_total_entropy = 0.0
    stop_token: str | None = None
    termination_reason = "max_stage_new_tokens"

    with torch.no_grad():
        for _ in range(generation_config.max_stage_new_tokens):
            total_length = input_ids.size(1) + current_decoder_input_ids.size(1)
            if total_length >= generation_config.max_sequence_length:
                termination_reason = "max_sequence_length"
                break

            outputs = policy_model.policy_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=current_decoder_input_ids,
                return_dict=True,
            )
            next_logits = outputs.logits[:, -1, :]
            next_token, next_log_prob, entropy = sample_next_token(
                next_logits,
                temperature=generation_config.temperature,
                top_p=generation_config.top_p,
                action_token_ids=tuple(raw_action_token_ids),
                stage_token_constraints=resolved_constraints,
            )
            next_token_id = int(next_token.item())
            raw_action_token_ids.append(next_token_id)
            raw_total_logprob += float(next_log_prob.item())
            raw_total_entropy += float(entropy.item())

            current_decoder_input_ids = torch.cat([current_decoder_input_ids, next_token], dim=1)

            if next_token_id == eom_token_id:
                stop_token = EOM_TOKEN
                termination_reason = "stop_token"
                break

    raw_stage_text = ""
    if raw_action_token_ids:
        raw_stage_text = tokenizer.decode(
            raw_action_token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True,
        ).strip()

    projection = project_sampled_stage_to_no_h(tokenizer, raw_stage_text)
    with torch.no_grad():
        projected_logprob_sum, projected_entropy_sum = compute_action_stats(
            policy_model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_prefix_ids,
            action_token_ids=projection.action_token_ids,
            stage_token_constraints=resolved_constraints,
        )

    metadata = dict(projection.metadata)
    metadata.update(
        {
            "raw_action_token_ids": tuple(raw_action_token_ids),
            "raw_action_logprob_sum": raw_total_logprob,
            "raw_entropy_sum": raw_total_entropy,
        }
    )
    return {
        "stage_text": projection.stage_text,
        "sampled_selfies": projection.sampled_selfies,
        "action_token_ids": list(projection.action_token_ids),
        "action_logprob_sum": float(projected_logprob_sum.item()),
        "entropy_sum": float(projected_entropy_sum.item()),
        "stop_token": stop_token,
        "termination_reason": termination_reason,
        "metadata": metadata,
    }


def sample_rollout_for_example(
    policy_model: PolicyValueModel,
    reference_model: T5ForConditionalGeneration,
    tokenizer: PreTrainedTokenizerBase,
    example: dict[str, Any],
    *,
    rollout_id: str,
    generation_config: RolloutGenerationConfig,
    reward_config: RewardConfig | None = None,
    device: torch.device,
    stage_token_constraints: StageTokenConstraints | None = None,
) -> list[StageTrajectory]:
    prompt_text = str(example["prompt"])
    description = str(example["description"])
    target_selfies_list = tuple(str(item) for item in example["target_selfies_list"])
    example_id = str(example["id"])

    prompt_inputs = encode_prompt(
        tokenizer,
        prompt_text,
        max_source_length=generation_config.max_source_length,
        device=device,
    )
    decoder_start_token_id = int(policy_model.policy_model.config.decoder_start_token_id)
    planned_stage_count = max(1, min(generation_config.max_molecules_per_sequence, len(target_selfies_list)))
    previous_selfies: list[str] = []
    trajectories: list[StageTrajectory] = []
    resolved_constraints = resolve_stage_token_constraints(policy_model, stage_token_constraints)

    for stage_index in range(1, planned_stage_count + 1):
        prefix_text = build_stage_prefix(
            previous_selfies,
            separator_token=generation_config.stage_separator,
        )
        decoder_prefix_ids = encode_decoder_prefix(
            tokenizer,
            prefix_text,
            decoder_start_token_id=decoder_start_token_id,
            device=device,
        )

        with torch.no_grad():
            value_old = float(
                policy_model.compute_values(
                    input_ids=prompt_inputs["input_ids"],
                    attention_mask=prompt_inputs["attention_mask"],
                    decoder_input_ids=decoder_prefix_ids,
                ).item()
            )

        stage_sample = sample_stage(
            policy_model,
            tokenizer,
            input_ids=prompt_inputs["input_ids"],
            attention_mask=prompt_inputs["attention_mask"],
            decoder_prefix_ids=decoder_prefix_ids,
            generation_config=generation_config,
            stage_token_constraints=resolved_constraints,
        )

        reference_logprob_sum, _ = compute_action_stats(
            reference_model,
            input_ids=prompt_inputs["input_ids"],
            attention_mask=prompt_inputs["attention_mask"],
            decoder_input_ids=decoder_prefix_ids,
            action_token_ids=stage_sample["action_token_ids"],
            stage_token_constraints=resolved_constraints,
        )

        reward_breakdown = score_stage_reward(
            stage_sample["sampled_selfies"] or "",
            targets=target_selfies_list,
            previous_candidates=previous_selfies,
            config=reward_config,
        )

        trajectory = StageTrajectory(
            rollout_id=rollout_id,
            example_id=example_id,
            prompt_text=prompt_text,
            description=description,
            target_selfies_list=target_selfies_list,
            stage_index=stage_index,
            decoder_prefix_text=prefix_text,
            stage_text=str(stage_sample["stage_text"]),
            sampled_selfies=stage_sample["sampled_selfies"],
            stop_token=stage_sample["stop_token"],
            termination_reason=stage_sample["termination_reason"],
            action_token_ids=tuple(stage_sample["action_token_ids"]),
            action_logprob_sum_old=float(stage_sample["action_logprob_sum"]),
            reference_logprob_sum=float(reference_logprob_sum.item()),
            value_old=value_old,
            reward_breakdown=reward_breakdown,
            reward=reward_breakdown.amplified_reward,
            entropy_sum_old=float(stage_sample["entropy_sum"]),
            is_valid=reward_breakdown.candidate.is_valid,
            is_duplicate=reward_breakdown.is_duplicate,
            metadata=dict(stage_sample.get("metadata", {})),
        )
        trajectories.append(trajectory)

        if trajectory.sampled_selfies and trajectory.is_valid:
            previous_selfies.append(trajectory.sampled_selfies)

        if trajectory.termination_reason != "stop_token":
            break
        if generation_config.terminate_on_invalid_stage and not trajectory.is_valid:
            break

    return trajectories

from __future__ import annotations

from dataclasses import dataclass
import random
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
from post_training.shared.sequence import (
    build_stage_prefix,
    project_sampled_stage_to_no_h,
    serialize_staged_molecule,
)
from molecules.selfies import decode_biot5_selfies

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


def encode_prompts(
    tokenizer: PreTrainedTokenizerBase,
    prompt_texts: Sequence[str],
    *,
    max_source_length: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        list(prompt_texts),
        padding=True,
        truncation=True,
        max_length=max_source_length,
        return_tensors="pt",
    )
    prompt_inputs = {key: value.to(device) for key, value in encoded.items()}
    if "attention_mask" not in prompt_inputs:
        prompt_inputs["attention_mask"] = torch.ones_like(prompt_inputs["input_ids"])
    return prompt_inputs


@dataclass(frozen=True)
class EncoderCache:
    last_hidden_state: torch.Tensor
    attention_mask: torch.Tensor

    def as_encoder_outputs(self) -> tuple[torch.Tensor]:
        return (self.last_hidden_state,)


def encode_prompt_cached(
    model: GFlowNetModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt_text: str,
    *,
    max_source_length: int,
    device: torch.device,
) -> EncoderCache:
    prompt_inputs = encode_prompt(
        tokenizer,
        prompt_text,
        max_source_length=max_source_length,
        device=device,
    )
    with torch.no_grad():
        encoder_outputs = model.policy_model.get_encoder()(
            input_ids=prompt_inputs["input_ids"],
            attention_mask=prompt_inputs["attention_mask"],
            return_dict=True,
        )
    return EncoderCache(
        last_hidden_state=encoder_outputs.last_hidden_state,
        attention_mask=prompt_inputs["attention_mask"],
    )


def encode_prompts_cached(
    model: GFlowNetModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt_texts: Sequence[str],
    *,
    max_source_length: int,
    device: torch.device,
) -> EncoderCache:
    prompt_inputs = encode_prompts(
        tokenizer,
        prompt_texts,
        max_source_length=max_source_length,
        device=device,
    )
    with torch.no_grad():
        encoder_outputs = model.policy_model.get_encoder()(
            input_ids=prompt_inputs["input_ids"],
            attention_mask=prompt_inputs["attention_mask"],
            return_dict=True,
        )
    return EncoderCache(
        last_hidden_state=encoder_outputs.last_hidden_state,
        attention_mask=prompt_inputs["attention_mask"],
    )


def _policy_model_prompt_kwargs(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    encoder_cache: EncoderCache | None,
) -> dict[str, object]:
    if encoder_cache is None:
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    return {
        "encoder_outputs": encoder_cache.as_encoder_outputs(),
        "attention_mask": encoder_cache.attention_mask,
    }


def _encode_prompt_for_sampling(
    model: GFlowNetModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt_text: str,
    *,
    max_source_length: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], EncoderCache | None]:
    get_encoder = getattr(model.policy_model, "get_encoder", None)
    if callable(get_encoder):
        encoder_cache = encode_prompt_cached(
            model,
            tokenizer,
            prompt_text,
            max_source_length=max_source_length,
            device=device,
        )
        prompt_inputs = {
            "input_ids": torch.empty_like(encoder_cache.attention_mask),
            "attention_mask": encoder_cache.attention_mask,
        }
        return prompt_inputs, encoder_cache

    prompt_inputs = encode_prompt(
        tokenizer,
        prompt_text,
        max_source_length=max_source_length,
        device=device,
    )
    return prompt_inputs, None


def _encode_prompts_for_sampling(
    model: GFlowNetModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt_texts: Sequence[str],
    *,
    max_source_length: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], EncoderCache | None]:
    get_encoder = getattr(model.policy_model, "get_encoder", None)
    if callable(get_encoder):
        encoder_cache = encode_prompts_cached(
            model,
            tokenizer,
            prompt_texts,
            max_source_length=max_source_length,
            device=device,
        )
        prompt_inputs = {
            "input_ids": torch.empty_like(encoder_cache.attention_mask),
            "attention_mask": encoder_cache.attention_mask,
        }
        return prompt_inputs, encoder_cache

    prompt_inputs = encode_prompts(
        tokenizer,
        prompt_texts,
        max_source_length=max_source_length,
        device=device,
    )
    return prompt_inputs, None


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


def _normalize_decoder_prefix_ids(
    decoder_prefix_ids: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    prefix_tensor = decoder_prefix_ids.to(device=device, dtype=torch.long)
    if prefix_tensor.ndim == 2:
        if prefix_tensor.size(0) != 1:
            raise ValueError("decoder_prefix_ids must have shape [1, prefix_length].")
        prefix_tensor = prefix_tensor.squeeze(0)
    elif prefix_tensor.ndim != 1:
        raise ValueError("decoder_prefix_ids must have shape [prefix_length] or [1, prefix_length].")
    if prefix_tensor.numel() < 1:
        raise ValueError("decoder_prefix_ids must include a decoder start token.")
    return prefix_tensor


def _slice_encoder_cache(
    encoder_cache: EncoderCache | None,
    row_indices: Sequence[int],
) -> EncoderCache | None:
    if encoder_cache is None:
        return None
    return EncoderCache(
        last_hidden_state=encoder_cache.last_hidden_state[row_indices],
        attention_mask=encoder_cache.attention_mask[row_indices],
    )


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


def _nonempty_metadata_text(metadata: dict[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _cleanup_selected_selfies(raw_stage_text: str | None) -> str | None:
    if raw_stage_text is None or not str(raw_stage_text).strip():
        return None
    try:
        selected_selfies = decode_biot5_selfies(str(raw_stage_text)).get("selected_selfies")
    except Exception:
        return None
    if selected_selfies is None:
        return None
    text = str(selected_selfies).strip()
    return text or None


def _resolve_invalid_candidate_text(
    *,
    sampled_selfies: str | None,
    stage_text: str,
    metadata: dict[str, Any],
) -> tuple[str | None, str | None]:
    if sampled_selfies is not None:
        return None, None

    raw_sampled_selfies = _nonempty_metadata_text(metadata, "raw_sampled_selfies")
    if raw_sampled_selfies is not None:
        return raw_sampled_selfies, "raw_sampled_selfies"

    raw_stage_text = _nonempty_metadata_text(metadata, "raw_stage_text")
    cleanup_selected_selfies = _cleanup_selected_selfies(raw_stage_text)
    if cleanup_selected_selfies is not None:
        return cleanup_selected_selfies, "cleanup_selected_selfies"

    if raw_stage_text is not None:
        return raw_stage_text, "raw_stage_text"

    fallback_stage_text = str(stage_text).strip()
    if fallback_stage_text:
        return fallback_stage_text, "stage_text"

    return None, None


def build_sampled_stage_trajectory_from_generation(
    *,
    example: dict[str, Any],
    rollout_id: str,
    stage_index: int,
    decoder_prefix_text: str,
    previous_sampled_selfies: Sequence[str],
    stage_text: str,
    sampled_selfies: str | None,
    action_token_ids: Sequence[int],
    metadata: dict[str, Any] | None,
    stop_token: str | None,
    termination_reason: str,
    reward_config: RewardConfig | None,
    invalid_terminal_reward: float,
) -> SampledStageTrajectory:
    prompt_text = str(example["prompt"])
    description = str(example["description"])
    target_selfies_list = tuple(str(item) for item in example["target_selfies_list"])
    trajectory_metadata = dict(metadata or {})
    invalid_candidate_text, invalid_candidate_text_source = _resolve_invalid_candidate_text(
        sampled_selfies=sampled_selfies,
        stage_text=stage_text,
        metadata=trajectory_metadata,
    )

    reward_summary = score_stage_terminal_reward(
        sampled_selfies,
        targets=target_selfies_list,
        previous_candidates=tuple(str(item) for item in previous_sampled_selfies),
        num_prefix_states=len(action_token_ids) + 1,
        reward_config=reward_config,
        invalid_terminal_reward=invalid_terminal_reward,
        invalid_candidate_text=invalid_candidate_text,
    )
    return SampledStageTrajectory(
        rollout_id=rollout_id,
        example_id=str(example["id"]),
        prompt_text=prompt_text,
        description=description,
        target_selfies_list=target_selfies_list,
        stage_index=stage_index,
        decoder_prefix_text=decoder_prefix_text,
        previous_sampled_selfies=tuple(str(item) for item in previous_sampled_selfies),
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
            **trajectory_metadata,
            "invalid_candidate_text": invalid_candidate_text,
            "invalid_candidate_text_source": invalid_candidate_text_source,
            "num_actions": len(action_token_ids),
            "stop_action_token": EOM_TOKEN,
        },
    )


def _target_selfies_list_for_guidance(
    example: dict[str, Any],
    *,
    max_molecules_per_sequence: int,
    rng: random.Random | None = None,
    shuffle_target_selfies_list: bool = False,
) -> tuple[str, ...]:
    target_selfies_list = tuple(str(item) for item in example["target_selfies_list"])
    max_stage_count = max(1, int(max_molecules_per_sequence))
    target_selfies_list = target_selfies_list[:max_stage_count]
    if not target_selfies_list:
        raise ValueError("Target-guided rollout requires at least one target SELFIES.")
    if shuffle_target_selfies_list and len(target_selfies_list) > 1:
        shuffled_target_selfies_list = list(target_selfies_list)
        generator = rng or random
        generator.shuffle(shuffled_target_selfies_list)
        target_selfies_list = tuple(shuffled_target_selfies_list)
    return target_selfies_list


def _example_with_target_selfies_list(
    example: dict[str, Any],
    target_selfies_list: Sequence[str],
) -> dict[str, Any]:
    guided_example = dict(example)
    guided_example["target_selfies_list"] = tuple(str(item) for item in target_selfies_list)
    return guided_example


def _select_target_guided_stage_index(
    target_selfies_list: Sequence[str],
    *,
    rng: random.Random | None,
    stage_strategy: str,
) -> int:
    normalized_strategy = str(stage_strategy).strip().lower()
    if normalized_strategy != "random":
        raise ValueError("Target-guided stage strategy must be: random.")
    generator = rng or random
    return int(generator.randrange(len(target_selfies_list))) + 1


def _encode_staged_molecule_action_ids(
    tokenizer: PreTrainedTokenizerBase,
    stage_text: str,
) -> tuple[int, ...]:
    encoded = tokenizer(
        stage_text,
        add_special_tokens=False,
        return_attention_mask=False,
    )
    input_ids = encoded["input_ids"]
    if isinstance(input_ids, torch.Tensor):
        if input_ids.ndim == 2:
            token_ids = [int(token_id) for token_id in input_ids[0].tolist()]
        else:
            token_ids = [int(token_id) for token_id in input_ids.reshape(-1).tolist()]
    elif input_ids and isinstance(input_ids[0], list):
        token_ids = [int(token_id) for token_id in input_ids[0]]
    else:
        token_ids = [int(token_id) for token_id in input_ids]

    eom_token_id = int(tokenizer.convert_tokens_to_ids(EOM_TOKEN))
    if not token_ids or int(token_ids[-1]) != eom_token_id:
        raise ValueError("Teacher-forced stage text did not tokenize with terminal <eom>.")
    return tuple(token_ids[:-1])


def sample_target_prefix_stage_trajectory_for_example(
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
    rng: random.Random | None = None,
    stage_strategy: str = "random",
    shuffle_target_selfies_list: bool = False,
) -> SampledStageTrajectory:
    target_selfies_list = _target_selfies_list_for_guidance(
        example,
        max_molecules_per_sequence=generation_config.max_molecules_per_sequence,
        rng=rng,
        shuffle_target_selfies_list=shuffle_target_selfies_list,
    )
    guided_example = _example_with_target_selfies_list(example, target_selfies_list)
    stage_index = _select_target_guided_stage_index(
        target_selfies_list,
        rng=rng,
        stage_strategy=stage_strategy,
    )
    previous_sampled_selfies = target_selfies_list[: stage_index - 1]
    prefix_text = build_stage_prefix(
        previous_sampled_selfies,
        separator_token=generation_config.stage_separator,
    )
    prompt_inputs, encoder_cache = _encode_prompt_for_sampling(
        model,
        tokenizer,
        str(example["prompt"]),
        max_source_length=generation_config.max_source_length,
        device=device,
    )
    decoder_start_token_id = int(model.policy_model.config.decoder_start_token_id)
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
        encoder_cache=encoder_cache,
        decoder_prefix_ids=decoder_prefix_ids,
        generation_config=generation_config,
        stage_token_constraints=stage_token_constraints,
    )
    metadata = {
        **dict(stage_sample.get("metadata", {})),
        "trajectory_source": "target_prefix_rollout",
        "target_guided_stage_index": stage_index,
    }
    return build_sampled_stage_trajectory_from_generation(
        example=guided_example,
        rollout_id=rollout_id,
        stage_index=stage_index,
        decoder_prefix_text=prefix_text,
        previous_sampled_selfies=previous_sampled_selfies,
        stage_text=str(stage_sample["stage_text"]),
        sampled_selfies=stage_sample["sampled_selfies"],
        action_token_ids=stage_sample["action_token_ids"],
        metadata=metadata,
        stop_token=stage_sample["stop_token"],
        termination_reason=stage_sample["termination_reason"],
        reward_config=reward_config,
        invalid_terminal_reward=invalid_terminal_reward,
    )


def build_target_teacher_stage_trajectory_for_example(
    tokenizer: PreTrainedTokenizerBase,
    example: dict[str, Any],
    *,
    rollout_id: str,
    generation_config: GFlowNetRolloutConfig,
    reward_config: RewardConfig | None = None,
    invalid_terminal_reward: float = 1.0e-4,
    rng: random.Random | None = None,
    stage_strategy: str = "random",
    shuffle_target_selfies_list: bool = False,
) -> SampledStageTrajectory:
    target_selfies_list = _target_selfies_list_for_guidance(
        example,
        max_molecules_per_sequence=generation_config.max_molecules_per_sequence,
        rng=rng,
        shuffle_target_selfies_list=shuffle_target_selfies_list,
    )
    guided_example = _example_with_target_selfies_list(example, target_selfies_list)
    stage_index = _select_target_guided_stage_index(
        target_selfies_list,
        rng=rng,
        stage_strategy=stage_strategy,
    )
    previous_sampled_selfies = target_selfies_list[: stage_index - 1]
    target_selfies = target_selfies_list[stage_index - 1]
    prefix_text = build_stage_prefix(
        previous_sampled_selfies,
        separator_token=generation_config.stage_separator,
    )
    stage_text = serialize_staged_molecule(target_selfies)
    action_token_ids = _encode_staged_molecule_action_ids(tokenizer, stage_text)
    return build_sampled_stage_trajectory_from_generation(
        example=guided_example,
        rollout_id=rollout_id,
        stage_index=stage_index,
        decoder_prefix_text=prefix_text,
        previous_sampled_selfies=previous_sampled_selfies,
        stage_text=stage_text,
        sampled_selfies=target_selfies,
        action_token_ids=action_token_ids,
        metadata={
            "raw_stage_text": stage_text,
            "raw_sampled_selfies": target_selfies,
            "trajectory_source": "target_teacher",
            "target_guided_stage_index": stage_index,
        },
        stop_token=EOM_TOKEN,
        termination_reason="stop_token",
        reward_config=reward_config,
        invalid_terminal_reward=invalid_terminal_reward,
    )


@dataclass(frozen=True)
class _BeamState:
    decoder_input_ids: torch.Tensor
    raw_action_token_ids: tuple[int, ...]
    score: float = 0.0
    stop_token: str | None = None
    termination_reason: str = "max_stage_new_tokens"
    generated_token_count: int = 0

    @property
    def is_finished(self) -> bool:
        return self.stop_token is not None or self.termination_reason in {
            "eos_token",
            "max_sequence_length",
        }


def _normalized_beam_score(beam: _BeamState, *, length_penalty: float) -> float:
    token_count = max(1, int(beam.generated_token_count))
    return float(beam.score) / float(token_count**max(0.0, length_penalty))


def _beam_rank_key(beam: _BeamState, *, length_penalty: float) -> tuple[float, bool, float]:
    return (
        _normalized_beam_score(beam, length_penalty=length_penalty),
        beam.stop_token == EOM_TOKEN,
        float(beam.score),
    )


def _select_top_beams(
    beams: Sequence[_BeamState],
    *,
    num_beams: int,
    length_penalty: float,
) -> list[_BeamState]:
    return sorted(
        beams,
        key=lambda beam: _beam_rank_key(beam, length_penalty=length_penalty),
        reverse=True,
    )[: max(1, int(num_beams))]


def _build_stage_sample_from_generation(
    tokenizer: PreTrainedTokenizerBase,
    *,
    raw_action_token_ids: Sequence[int],
    stop_token: str | None,
    termination_reason: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    eom_token_id = tokenizer.convert_tokens_to_ids(EOM_TOKEN)
    raw_stage_token_ids = [int(token_id) for token_id in raw_action_token_ids]
    if stop_token == EOM_TOKEN:
        raw_stage_token_ids.append(int(eom_token_id))
    raw_stage_text = (
        tokenizer.decode(
            raw_stage_token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True,
        ).strip()
        if raw_stage_token_ids
        else ""
    )
    projection = project_sampled_stage_to_no_h(
        tokenizer,
        raw_stage_text,
        drop_terminal_eom_from_action_ids=True,
    )
    action_token_ids = tuple(projection.action_token_ids)
    used_raw_action_ids_for_invalid_projection = False
    if projection.sampled_selfies is None and not action_token_ids:
        action_token_ids = tuple(int(token_id) for token_id in raw_action_token_ids)
        used_raw_action_ids_for_invalid_projection = bool(action_token_ids)

    return {
        "stage_text": projection.stage_text,
        "sampled_selfies": projection.sampled_selfies,
        "action_token_ids": action_token_ids,
        "stop_token": stop_token,
        "termination_reason": termination_reason,
        "metadata": {
            **projection.metadata,
            **dict(metadata or {}),
            "raw_action_token_ids": tuple(int(token_id) for token_id in raw_action_token_ids),
            "used_raw_action_ids_for_invalid_projection": used_raw_action_ids_for_invalid_projection,
        },
    }


def beam_search_stage(
    model: GFlowNetModel,
    tokenizer: PreTrainedTokenizerBase,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    encoder_cache: EncoderCache | None = None,
    decoder_prefix_ids: torch.Tensor,
    generation_config: GFlowNetRolloutConfig,
    stage_token_constraints: StageTokenConstraints | None = None,
) -> dict[str, Any]:
    eom_token_id = int(tokenizer.convert_tokens_to_ids(EOM_TOKEN))
    eos_token_id = model.policy_model.config.eos_token_id
    resolved_constraints = resolve_stage_token_constraints(model, stage_token_constraints)
    num_beams = max(1, int(generation_config.num_beams))
    length_penalty = max(0.0, float(generation_config.length_penalty))

    beams: list[_BeamState] = [
        _BeamState(
            decoder_input_ids=decoder_prefix_ids.clone(),
            raw_action_token_ids=(),
        )
    ]

    with torch.no_grad():
        for _ in range(generation_config.max_stage_new_tokens):
            candidates: list[_BeamState] = []
            for beam in beams:
                if beam.is_finished:
                    candidates.append(beam)
                    continue

                total_length = input_ids.size(1) + beam.decoder_input_ids.size(1)
                if total_length >= generation_config.max_sequence_length:
                    candidates.append(
                        _BeamState(
                            decoder_input_ids=beam.decoder_input_ids,
                            raw_action_token_ids=beam.raw_action_token_ids,
                            score=beam.score,
                            stop_token=None,
                            termination_reason="max_sequence_length",
                            generated_token_count=beam.generated_token_count,
                        )
                    )
                    continue

                outputs = model.policy_model(
                    **_policy_model_prompt_kwargs(input_ids, attention_mask, encoder_cache),
                    decoder_input_ids=beam.decoder_input_ids,
                    return_dict=True,
                )
                next_logits = outputs.logits[:, -1, :] / max(
                    float(generation_config.temperature),
                    1.0e-6,
                )
                if resolved_constraints is not None and resolved_constraints.enabled:
                    allowed_token_ids = resolved_constraints.allowed_token_ids_for_prefix(
                        beam.raw_action_token_ids
                    )
                    next_logits = mask_logits_to_allowed_token_ids(
                        next_logits,
                        allowed_token_ids,
                    )
                    top_k = min(num_beams, len(allowed_token_ids))
                else:
                    top_k = min(num_beams, int(next_logits.size(-1)))

                log_probs = torch.log_softmax(next_logits, dim=-1).squeeze(0)
                top_log_probs, top_token_ids = torch.topk(log_probs, k=max(1, top_k))
                for token_log_prob, token_id_tensor in zip(top_log_probs, top_token_ids):
                    if not torch.isfinite(token_log_prob):
                        continue

                    token_id = int(token_id_tensor.item())
                    next_score = float(beam.score) + float(token_log_prob.item())
                    generated_token_count = int(beam.generated_token_count) + 1
                    if token_id == eom_token_id:
                        candidates.append(
                            _BeamState(
                                decoder_input_ids=beam.decoder_input_ids,
                                raw_action_token_ids=beam.raw_action_token_ids,
                                score=next_score,
                                stop_token=EOM_TOKEN,
                                termination_reason="stop_token",
                                generated_token_count=generated_token_count,
                            )
                        )
                        continue
                    if eos_token_id is not None and token_id == int(eos_token_id):
                        candidates.append(
                            _BeamState(
                                decoder_input_ids=beam.decoder_input_ids,
                                raw_action_token_ids=beam.raw_action_token_ids,
                                score=next_score,
                                stop_token=tokenizer.eos_token,
                                termination_reason="eos_token",
                                generated_token_count=generated_token_count,
                            )
                        )
                        continue

                    next_token = torch.tensor(
                        [[token_id]],
                        dtype=torch.long,
                        device=beam.decoder_input_ids.device,
                    )
                    candidates.append(
                        _BeamState(
                            decoder_input_ids=torch.cat(
                                [beam.decoder_input_ids, next_token],
                                dim=1,
                            ),
                            raw_action_token_ids=(*beam.raw_action_token_ids, token_id),
                            score=next_score,
                            stop_token=None,
                            termination_reason="max_stage_new_tokens",
                            generated_token_count=generated_token_count,
                        )
                    )

            if not candidates:
                break

            beams = _select_top_beams(
                candidates,
                num_beams=num_beams,
                length_penalty=length_penalty,
            )
            if generation_config.early_stopping and all(beam.is_finished for beam in beams):
                break

    ranked_beams = _select_top_beams(
        beams,
        num_beams=len(beams),
        length_penalty=length_penalty,
    )
    best_beam = ranked_beams[0]
    return _build_stage_sample_from_generation(
        tokenizer,
        raw_action_token_ids=best_beam.raw_action_token_ids,
        stop_token=best_beam.stop_token,
        termination_reason=best_beam.termination_reason,
        metadata={
            "decoding_strategy": "beam",
            "num_beams": num_beams,
            "beam_rank": 0,
            "beam_score": float(best_beam.score),
            "beam_normalized_score": _normalized_beam_score(
                best_beam,
                length_penalty=length_penalty,
            ),
        },
    )


def sample_stage(
    model: GFlowNetModel,
    tokenizer: PreTrainedTokenizerBase,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    encoder_cache: EncoderCache | None = None,
    decoder_prefix_ids: torch.Tensor,
    generation_config: GFlowNetRolloutConfig,
    stage_token_constraints: StageTokenConstraints | None = None,
) -> dict[str, Any]:
    decoding_strategy = str(generation_config.decoding_strategy).strip().lower()
    if decoding_strategy == "beam":
        return beam_search_stage(
            model,
            tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_cache=encoder_cache,
            decoder_prefix_ids=decoder_prefix_ids,
            generation_config=generation_config,
            stage_token_constraints=stage_token_constraints,
        )

    eom_token_id = tokenizer.convert_tokens_to_ids(EOM_TOKEN)
    eos_token_id = model.policy_model.config.eos_token_id
    resolved_constraints = resolve_stage_token_constraints(model, stage_token_constraints)

    current_decoder_input_ids = decoder_prefix_ids.clone()
    past_key_values = None
    raw_action_token_ids: list[int] = []
    stop_token: str | None = None
    termination_reason = "max_stage_new_tokens"
    source_length = attention_mask.size(1)
    decoder_prefix_length = decoder_prefix_ids.size(1)

    with torch.no_grad():
        for _ in range(generation_config.max_stage_new_tokens):
            total_length = source_length + decoder_prefix_length + len(raw_action_token_ids)
            if total_length >= generation_config.max_sequence_length:
                termination_reason = "max_sequence_length"
                break

            outputs = model.policy_model(
                **_policy_model_prompt_kwargs(input_ids, attention_mask, encoder_cache),
                decoder_input_ids=current_decoder_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]
            next_token, _, _ = sample_next_token(
                next_logits,
                temperature=generation_config.temperature,
                top_p=generation_config.top_p,
                action_token_ids=tuple(raw_action_token_ids),
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

            raw_action_token_ids.append(next_token_id)
            current_decoder_input_ids = next_token

    return _build_stage_sample_from_generation(
        tokenizer,
        raw_action_token_ids=raw_action_token_ids,
        stop_token=stop_token,
        termination_reason=termination_reason,
    )


def sample_stage_batch(
    model: GFlowNetModel,
    tokenizer: PreTrainedTokenizerBase,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    encoder_cache: EncoderCache | None = None,
    decoder_prefix_ids: Sequence[torch.Tensor],
    generation_config: GFlowNetRolloutConfig,
    stage_token_constraints: StageTokenConstraints | None = None,
) -> list[dict[str, Any]]:
    batch_size = len(decoder_prefix_ids)
    if batch_size == 0:
        return []
    if input_ids.ndim != 2 or attention_mask.ndim != 2:
        raise ValueError("input_ids and attention_mask must have shape [batch_size, sequence_length].")
    if int(input_ids.size(0)) != batch_size or int(attention_mask.size(0)) != batch_size:
        raise ValueError("decoder_prefix_ids length must match the prompt batch size.")

    decoding_strategy = str(generation_config.decoding_strategy).strip().lower()
    if decoding_strategy == "beam":
        outputs: list[dict[str, Any]] = []
        for row_index, prefix_ids in enumerate(decoder_prefix_ids):
            outputs.append(
                beam_search_stage(
                    model,
                    tokenizer,
                    input_ids=input_ids[row_index : row_index + 1],
                    attention_mask=attention_mask[row_index : row_index + 1],
                    encoder_cache=_slice_encoder_cache(encoder_cache, [row_index]),
                    decoder_prefix_ids=_normalize_decoder_prefix_ids(
                        prefix_ids,
                        device=attention_mask.device,
                    ).unsqueeze(0),
                    generation_config=generation_config,
                    stage_token_constraints=stage_token_constraints,
                )
            )
        return outputs

    eom_token_id = int(tokenizer.convert_tokens_to_ids(EOM_TOKEN))
    eos_token_id = model.policy_model.config.eos_token_id
    resolved_constraints = resolve_stage_token_constraints(model, stage_token_constraints)
    device = attention_mask.device
    source_lengths = [int(value) for value in attention_mask.sum(dim=1).tolist()]
    current_decoder_sequences = [
        _normalize_decoder_prefix_ids(prefix_ids, device=device).clone()
        for prefix_ids in decoder_prefix_ids
    ]
    decoder_prefix_lengths = [int(sequence.numel()) for sequence in current_decoder_sequences]
    raw_action_token_ids: list[list[int]] = [[] for _ in range(batch_size)]
    stop_tokens: list[str | None] = [None] * batch_size
    termination_reasons: list[str] = ["max_stage_new_tokens"] * batch_size
    finished = [False] * batch_size
    decoder_pad_token_id = int(getattr(model.policy_model.config, "decoder_start_token_id", 0))

    with torch.no_grad():
        for _ in range(generation_config.max_stage_new_tokens):
            active_rows: list[int] = []
            for row_index in range(batch_size):
                if finished[row_index]:
                    continue
                total_length = (
                    source_lengths[row_index]
                    + decoder_prefix_lengths[row_index]
                    + len(raw_action_token_ids[row_index])
                )
                if total_length >= generation_config.max_sequence_length:
                    termination_reasons[row_index] = "max_sequence_length"
                    finished[row_index] = True
                    continue
                active_rows.append(row_index)

            if not active_rows:
                break

            max_decoder_length = max(
                int(current_decoder_sequences[row_index].numel()) for row_index in active_rows
            )
            decoder_input_ids = torch.full(
                (len(active_rows), max_decoder_length),
                decoder_pad_token_id,
                dtype=torch.long,
                device=device,
            )
            decoder_attention_mask = torch.zeros_like(decoder_input_ids)
            for batch_row, row_index in enumerate(active_rows):
                sequence = current_decoder_sequences[row_index]
                decoder_input_ids[batch_row, : sequence.numel()] = sequence
                decoder_attention_mask[batch_row, : sequence.numel()] = 1

            active_input_ids = input_ids[active_rows]
            active_attention_mask = attention_mask[active_rows]
            active_encoder_cache = _slice_encoder_cache(encoder_cache, active_rows)
            outputs = model.policy_model(
                **_policy_model_prompt_kwargs(
                    active_input_ids,
                    active_attention_mask,
                    active_encoder_cache,
                ),
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                return_dict=True,
            )
            next_logits = outputs.logits[:, -1, :]

            for batch_row, row_index in enumerate(active_rows):
                next_token, _, _ = sample_next_token(
                    next_logits[batch_row : batch_row + 1, :],
                    temperature=generation_config.temperature,
                    top_p=generation_config.top_p,
                    action_token_ids=tuple(raw_action_token_ids[row_index]),
                    stage_token_constraints=resolved_constraints,
                )
                next_token_id = int(next_token.item())
                if next_token_id == eom_token_id:
                    stop_tokens[row_index] = EOM_TOKEN
                    termination_reasons[row_index] = "stop_token"
                    finished[row_index] = True
                    continue
                if eos_token_id is not None and next_token_id == int(eos_token_id):
                    stop_tokens[row_index] = tokenizer.eos_token
                    termination_reasons[row_index] = "eos_token"
                    finished[row_index] = True
                    continue

                raw_action_token_ids[row_index].append(next_token_id)
                current_decoder_sequences[row_index] = torch.cat(
                    [
                        current_decoder_sequences[row_index],
                        next_token.reshape(-1).to(device=device, dtype=torch.long),
                    ]
                )

    return [
        _build_stage_sample_from_generation(
            tokenizer,
            raw_action_token_ids=raw_action_token_ids[row_index],
            stop_token=stop_tokens[row_index],
            termination_reason=termination_reasons[row_index],
        )
        for row_index in range(batch_size)
    ]


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
    rng: random.Random | None = None,
    return_last_valid_trajectory_only: bool = False,
) -> list[SampledStageTrajectory]:
    prompt_inputs, encoder_cache = _encode_prompt_for_sampling(
        model,
        tokenizer,
        str(example["prompt"]),
        max_source_length=generation_config.max_source_length,
        device=device,
    )
    decoder_start_token_id = int(model.policy_model.config.decoder_start_token_id)
    planned_stage_count = max(1, int(generation_config.max_molecules_per_sequence))

    previous_sampled_selfies: list[str] = []
    trajectories: list[SampledStageTrajectory] = []
    last_valid_trajectory: SampledStageTrajectory | None = None
    generator = rng or random
    stage_index = 1
    while stage_index <= planned_stage_count:
        prefix_text = build_stage_prefix(
            previous_sampled_selfies,
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
            encoder_cache=encoder_cache,
            decoder_prefix_ids=decoder_prefix_ids,
            generation_config=generation_config,
            stage_token_constraints=stage_token_constraints,
        )
        trajectory = build_sampled_stage_trajectory_from_generation(
            example=example,
            rollout_id=rollout_id,
            stage_index=stage_index,
            decoder_prefix_text=prefix_text,
            previous_sampled_selfies=tuple(previous_sampled_selfies),
            stage_text=str(stage_sample["stage_text"]),
            sampled_selfies=stage_sample["sampled_selfies"],
            action_token_ids=stage_sample["action_token_ids"],
            metadata=dict(stage_sample.get("metadata", {})),
            stop_token=stage_sample["stop_token"],
            termination_reason=stage_sample["termination_reason"],
            reward_config=reward_config,
            invalid_terminal_reward=invalid_terminal_reward,
        )
        if trajectory.sampled_selfies and trajectory.is_valid:
            last_valid_trajectory = trajectory
            previous_sampled_selfies.append(trajectory.sampled_selfies)

        trajectory_appended = False
        if not return_last_valid_trajectory_only:
            if trajectory.is_valid:
                should_append = stage_index == 1 or (
                    float(generator.random()) < generation_config.append_probability
                )
            else:
                should_append = (
                    float(generator.random()) < generation_config.invalid_append_probability
                )
            if should_append:
                trajectories.append(trajectory)
                trajectory_appended = True

        if trajectory.termination_reason != "stop_token":
            if (
                trajectory.is_valid
                and not return_last_valid_trajectory_only
                and not trajectory_appended
            ):
                trajectories.append(trajectory)
            break
        if generation_config.terminate_on_invalid_stage and not trajectory.is_valid:
            stage_index += 1
            continue
        stage_index += 1

    if return_last_valid_trajectory_only:
        return [last_valid_trajectory] if last_valid_trajectory is not None else []
    return trajectories


def sample_stage_trajectories_for_examples(
    model: GFlowNetModel,
    tokenizer: PreTrainedTokenizerBase,
    examples: Sequence[dict[str, Any]],
    *,
    rollout_ids: Sequence[str],
    generation_config: GFlowNetRolloutConfig,
    reward_config: RewardConfig | None = None,
    invalid_terminal_reward: float = 1.0e-4,
    device: torch.device,
    stage_token_constraints: StageTokenConstraints | None = None,
    rng: random.Random | None = None,
    return_last_valid_trajectory_only: bool = False,
) -> list[list[SampledStageTrajectory]]:
    if len(examples) != len(rollout_ids):
        raise ValueError("examples and rollout_ids must have the same length.")
    if not examples:
        return []
    if str(generation_config.decoding_strategy).strip().lower() == "beam":
        return [
            sample_stage_trajectories_for_example(
                model,
                tokenizer,
                example,
                rollout_id=rollout_id,
                generation_config=generation_config,
                reward_config=reward_config,
                invalid_terminal_reward=invalid_terminal_reward,
                device=device,
                stage_token_constraints=stage_token_constraints,
                rng=rng,
                return_last_valid_trajectory_only=return_last_valid_trajectory_only,
            )
            for example, rollout_id in zip(examples, rollout_ids)
        ]

    prompt_inputs, encoder_cache = _encode_prompts_for_sampling(
        model,
        tokenizer,
        [str(example["prompt"]) for example in examples],
        max_source_length=generation_config.max_source_length,
        device=device,
    )
    decoder_start_token_id = int(model.policy_model.config.decoder_start_token_id)
    planned_stage_count = max(1, int(generation_config.max_molecules_per_sequence))
    generator = rng or random
    previous_sampled_selfies: list[list[str]] = [[] for _ in examples]
    trajectory_records: list[list[SampledStageTrajectory]] = [[] for _ in examples]
    last_valid_trajectories: list[SampledStageTrajectory | None] = [None for _ in examples]
    stage_indices = [1 for _ in examples]
    finished = [False for _ in examples]

    while True:
        active_rows = [
            row_index
            for row_index in range(len(examples))
            if not finished[row_index] and stage_indices[row_index] <= planned_stage_count
        ]
        if not active_rows:
            break

        prefix_texts = [
            build_stage_prefix(
                previous_sampled_selfies[row_index],
                separator_token=generation_config.stage_separator,
            )
            for row_index in active_rows
        ]
        stage_samples = sample_stage_batch(
            model,
            tokenizer,
            input_ids=prompt_inputs["input_ids"][active_rows],
            attention_mask=prompt_inputs["attention_mask"][active_rows],
            encoder_cache=_slice_encoder_cache(encoder_cache, active_rows),
            decoder_prefix_ids=[
                encode_decoder_prefix(
                    tokenizer,
                    prefix_text,
                    decoder_start_token_id=decoder_start_token_id,
                    device=device,
                )
                for prefix_text in prefix_texts
            ],
            generation_config=generation_config,
            stage_token_constraints=stage_token_constraints,
        )

        for local_row, row_index in enumerate(active_rows):
            trajectory = build_sampled_stage_trajectory_from_generation(
                example=examples[row_index],
                rollout_id=str(rollout_ids[row_index]),
                stage_index=stage_indices[row_index],
                decoder_prefix_text=prefix_texts[local_row],
                previous_sampled_selfies=tuple(previous_sampled_selfies[row_index]),
                stage_text=str(stage_samples[local_row]["stage_text"]),
                sampled_selfies=stage_samples[local_row]["sampled_selfies"],
                action_token_ids=stage_samples[local_row]["action_token_ids"],
                metadata=dict(stage_samples[local_row].get("metadata", {})),
                stop_token=stage_samples[local_row]["stop_token"],
                termination_reason=stage_samples[local_row]["termination_reason"],
                reward_config=reward_config,
                invalid_terminal_reward=invalid_terminal_reward,
            )
            if trajectory.sampled_selfies and trajectory.is_valid:
                last_valid_trajectories[row_index] = trajectory
                previous_sampled_selfies[row_index].append(trajectory.sampled_selfies)

            trajectory_appended = False
            if not return_last_valid_trajectory_only:
                if trajectory.is_valid:
                    should_append = stage_indices[row_index] == 1 or (
                        float(generator.random()) < generation_config.append_probability
                    )
                else:
                    should_append = (
                        float(generator.random()) < generation_config.invalid_append_probability
                    )
                if should_append:
                    trajectory_records[row_index].append(trajectory)
                    trajectory_appended = True

            if trajectory.termination_reason != "stop_token":
                if (
                    trajectory.is_valid
                    and not return_last_valid_trajectory_only
                    and not trajectory_appended
                ):
                    trajectory_records[row_index].append(trajectory)
                finished[row_index] = True
                continue

            stage_indices[row_index] += 1
            if (
                generation_config.terminate_on_invalid_stage
                and not trajectory.is_valid
                and stage_indices[row_index] > planned_stage_count
            ):
                finished[row_index] = True
            elif stage_indices[row_index] > planned_stage_count:
                finished[row_index] = True

    if return_last_valid_trajectory_only:
        return [
            [trajectory] if trajectory is not None else []
            for trajectory in last_valid_trajectories
        ]
    return trajectory_records

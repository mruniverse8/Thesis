from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from molecules.defaults import (
    DEFAULT_DIVERSITY_BETA,
    DEFAULT_DIVERSITY_WEIGHT,
    DEFAULT_FINGERPRINT_NUM_BITS,
    DEFAULT_FINGERPRINT_RADIUS,
    DEFAULT_MATCH_ALPHA,
    DEFAULT_MATCH_WEIGHT,
    DEFAULT_PLUS_VALID,
    DEFAULT_REWARD_AMPLIFICATION,
    DEFAULT_REWARD_VARIANT,
    RewardConfig,
)
from molecules.fingerprints import ensure_molecule_record
from molecules.parsing import is_duplicate_candidate
from molecules.representations import MoleculeRecord, MoleculeRepresentation
from molecules.similarity import compute_dice_similarity, compute_tanimoto_similarity


@dataclass(frozen=True)
class RewardComponent:
    reward: float
    max_similarity: float
    exponent: float
    best_index: int | None
    best_reference: str | None
    num_references: int


@dataclass(frozen=True)
class RewardBreakdown:
    candidate: MoleculeRecord
    match: RewardComponent
    diversity: RewardComponent
    total_reward: float
    amplified_reward: float
    match_weight: float
    diversity_weight: float
    is_duplicate: bool


def _materialize_records(
    molecules: Sequence[MoleculeRecord | object | str],
    *,
    representation: MoleculeRepresentation,
) -> list[MoleculeRecord]:
    return [ensure_molecule_record(molecule, representation=representation) for molecule in molecules]


def _compute_component(
    candidate: MoleculeRecord,
    references: Sequence[MoleculeRecord],
    *,
    similarity_fn,
    exponent: float,
) -> RewardComponent:
    best_similarity = 0.0
    best_index: int | None = None
    best_reference: str | None = None

    for index, reference in enumerate(references):
        #if not reference.is_valid:
        #    continue
        similarity = similarity_fn(candidate, reference)
        if similarity > best_similarity:
            best_similarity = similarity
            best_index = index
            best_reference = reference.canonical_smiles

    reward = best_similarity**exponent if best_index is not None else 0.0
    return RewardComponent(
        reward=reward,
        max_similarity=best_similarity,
        exponent=exponent,
        best_index=best_index,
        best_reference=best_reference,
        num_references=len(references),
    )


def _zero_reward_component(*, exponent: float, num_references: int) -> RewardComponent:
    return RewardComponent(
        reward=0.0,
        max_similarity=0.0,
        exponent=exponent,
        best_index=None,
        best_reference=None,
        num_references=num_references,
    )


def _apply_duplicate_penalty(
    total_reward: float,
    *,
    is_duplicate: bool,
    duplicate_penalty_factor: float,
) -> float:
    if not is_duplicate:
        return total_reward
    return total_reward * duplicate_penalty_factor


def compute_rmatch(
    candidate: MoleculeRecord | object | str,
    targets: Sequence[MoleculeRecord | object | str],
    *,
    alpha: float = DEFAULT_MATCH_ALPHA,
    candidate_representation: MoleculeRepresentation = "auto",
    target_representation: MoleculeRepresentation = "auto",
    radius: int = DEFAULT_FINGERPRINT_RADIUS,
    n_bits: int = DEFAULT_FINGERPRINT_NUM_BITS,
) -> RewardComponent:
    candidate_record = ensure_molecule_record(candidate, representation=candidate_representation)
    target_records = _materialize_records(targets, representation=target_representation)
    return _compute_component(
        candidate_record,
        target_records,
        similarity_fn=lambda a, b: compute_dice_similarity(
            a,
            b,
            radius=radius,
            n_bits=n_bits,
        ),
        exponent=alpha,
    )


def compute_rdiv(
    candidate: MoleculeRecord | object | str,
    previous_candidates: Sequence[MoleculeRecord | object | str],
    *,
    beta: float = DEFAULT_DIVERSITY_BETA,
    candidate_representation: MoleculeRepresentation = "auto",
    previous_representation: MoleculeRepresentation = "auto",
    radius: int = DEFAULT_FINGERPRINT_RADIUS,
    n_bits: int = DEFAULT_FINGERPRINT_NUM_BITS,
) -> RewardComponent:
    candidate_record = ensure_molecule_record(candidate, representation=candidate_representation)
    previous_records = _materialize_records(previous_candidates, representation=previous_representation)

    if not previous_records:
        return RewardComponent(
            reward=0.0,
            max_similarity=0.0,
            exponent=beta,
            best_index=None,
            best_reference=None,
            num_references=0,
        )

    base_component = _compute_component(
        candidate_record,
        previous_records,
        similarity_fn=lambda a, b: compute_tanimoto_similarity(
            a,
            b,
            radius=radius,
            n_bits=n_bits,
        ),
        exponent=beta,
    )
    if not candidate_record.is_valid or base_component.best_index is None:
        return base_component

    return RewardComponent(
        reward=1.0 - base_component.reward,
        max_similarity=base_component.max_similarity,
        exponent=beta,
        best_index=base_component.best_index,
        best_reference=base_component.best_reference,
        num_references=base_component.num_references,
    )


def _compute_total_reward_var1(
    candidate_record: MoleculeRecord,
    target_records: Sequence[MoleculeRecord],
    previous_records: Sequence[MoleculeRecord],
    *,
    is_duplicate: bool,
    reward_config: RewardConfig,
    match_weight: float,
    diversity_weight: float,
    reward_amplification: float,
) -> RewardBreakdown:
    match_component = compute_rmatch(
        candidate_record,
        target_records,
        alpha=reward_config.match_alpha,
        radius=reward_config.fingerprint_radius,
        n_bits=reward_config.fingerprint_num_bits,
    )
    diversity_component = compute_rdiv(
        candidate_record,
        previous_records,
        beta=reward_config.diversity_beta,
        radius=reward_config.fingerprint_radius,
        n_bits=reward_config.fingerprint_num_bits,
    )
    total_reward = match_weight * match_component.reward + diversity_weight * diversity_component.reward
    return RewardBreakdown(
        candidate=candidate_record,
        match=match_component,
        diversity=diversity_component,
        total_reward=total_reward,
        amplified_reward=reward_amplification * total_reward,
        match_weight=match_weight,
        diversity_weight=diversity_weight,
        is_duplicate=is_duplicate,
    )


def _compute_total_reward_var2(
    candidate_record: MoleculeRecord,
    target_records: Sequence[MoleculeRecord],
    previous_records: Sequence[MoleculeRecord],
    *,
    is_duplicate: bool,
    reward_config: RewardConfig,
    plus_valid: float,
    duplicate_penalty_factor: float,
    reward_amplification: float,
) -> RewardBreakdown:
    match_component = compute_rmatch(
        candidate_record,
        target_records,
        alpha=reward_config.match_alpha,
        radius=reward_config.fingerprint_radius,
        n_bits=reward_config.fingerprint_num_bits,
    )
    diversity_component = _zero_reward_component(
        exponent=reward_config.diversity_beta,
        num_references=len(previous_records),
    )
    total_reward = match_component.reward + (plus_valid if candidate_record.is_valid else 0.0)
    total_reward = _apply_duplicate_penalty(
        total_reward,
        is_duplicate=is_duplicate,
        duplicate_penalty_factor=duplicate_penalty_factor,
    )
    return RewardBreakdown(
        candidate=candidate_record,
        match=match_component,
        diversity=diversity_component,
        total_reward=total_reward,
        amplified_reward=reward_amplification * total_reward,
        match_weight=1.0,
        diversity_weight=0.0,
        is_duplicate=is_duplicate,
    )


def _compute_total_reward_var3(
    candidate_record: MoleculeRecord,
    target_records: Sequence[MoleculeRecord],
    previous_records: Sequence[MoleculeRecord],
    *,
    is_duplicate: bool,
    reward_config: RewardConfig,
    plus_valid: float,
    match_weight: float,
    diversity_weight: float,
    reward_amplification: float,
) -> RewardBreakdown:
    match_component = compute_rmatch(
        candidate_record,
        target_records,
        alpha=reward_config.match_alpha,
        radius=reward_config.fingerprint_radius,
        n_bits=reward_config.fingerprint_num_bits,
    )
    diversity_component = compute_rdiv(
        candidate_record,
        previous_records,
        beta=reward_config.diversity_beta,
        radius=reward_config.fingerprint_radius,
        n_bits=reward_config.fingerprint_num_bits,
    )
    total_reward = (
        match_weight * match_component.reward
        + diversity_weight * diversity_component.reward
        + (plus_valid if candidate_record.is_valid else 0.0)
    )
    return RewardBreakdown(
        candidate=candidate_record,
        match=match_component,
        diversity=diversity_component,
        total_reward=total_reward,
        amplified_reward=reward_amplification * total_reward,
        match_weight=match_weight,
        diversity_weight=diversity_weight,
        is_duplicate=is_duplicate,
    )


def compute_total_reward(
    candidate: MoleculeRecord | object | str,
    targets: Sequence[MoleculeRecord | object | str],
    previous_candidates: Sequence[MoleculeRecord | object | str],
    *,
    config: RewardConfig | None = None,
    candidate_representation: MoleculeRepresentation = "auto",
    target_representation: MoleculeRepresentation = "auto",
    previous_representation: MoleculeRepresentation = "auto",
    match_weight: float = DEFAULT_MATCH_WEIGHT,
    diversity_weight: float = DEFAULT_DIVERSITY_WEIGHT,
    reward_amplification: float = DEFAULT_REWARD_AMPLIFICATION,
) -> RewardBreakdown:
    reward_config = config or RewardConfig()
    candidate_record = ensure_molecule_record(candidate, representation=candidate_representation)
    target_records = _materialize_records(targets, representation=target_representation)
    previous_records = _materialize_records(previous_candidates, representation=previous_representation)

    effective_match_weight = reward_config.match_weight if config else match_weight
    effective_diversity_weight = reward_config.diversity_weight if config else diversity_weight
    effective_amplification = reward_config.reward_amplification if config else reward_amplification
    effective_plus_valid = reward_config.plus_valid if config else DEFAULT_PLUS_VALID
    effective_variant = reward_config.reward_variant if config else DEFAULT_REWARD_VARIANT
    effective_duplicate_penalty_factor = reward_config.duplicate_penalty_factor
    is_duplicate = is_duplicate_candidate(candidate_record, list(previous_records))

    if effective_variant == "reward_var1":
        return _compute_total_reward_var1(
            candidate_record,
            target_records,
            previous_records,
            is_duplicate=is_duplicate,
            reward_config=reward_config,
            match_weight=effective_match_weight,
            diversity_weight=effective_diversity_weight,
            reward_amplification=effective_amplification,
        )
    if effective_variant == "reward_var2":
        return _compute_total_reward_var2(
            candidate_record,
            target_records,
            previous_records,
            is_duplicate=is_duplicate,
            reward_config=reward_config,
            plus_valid=effective_plus_valid,
            duplicate_penalty_factor=effective_duplicate_penalty_factor,
            reward_amplification=effective_amplification,
        )
    if effective_variant == "reward_var3":
        return _compute_total_reward_var3(
            candidate_record,
            target_records,
            previous_records,
            is_duplicate=is_duplicate,
            reward_config=reward_config,
            plus_valid=effective_plus_valid,
            match_weight=effective_match_weight,
            diversity_weight=effective_diversity_weight,
            reward_amplification=effective_amplification,
        )
    raise ValueError(f"Unsupported reward variant: {effective_variant}")


def score_candidate_sequence(
    candidates: Sequence[MoleculeRecord | object | str],
    targets: Sequence[MoleculeRecord | object | str],
    *,
    config: RewardConfig | None = None,
    candidate_representation: MoleculeRepresentation = "auto",
    target_representation: MoleculeRepresentation = "auto",
) -> list[RewardBreakdown]:
    reward_config = config or RewardConfig()
    target_records = _materialize_records(targets, representation=target_representation)
    scored_candidates: list[RewardBreakdown] = []
    previous_records: list[MoleculeRecord] = []

    for candidate in candidates:
        breakdown = compute_total_reward(
            candidate,
            target_records,
            previous_records,
            config=reward_config,
            candidate_representation=candidate_representation,
            target_representation=target_representation,
            previous_representation="auto",
        )
        scored_candidates.append(breakdown)
        previous_records.append(breakdown.candidate)

    return scored_candidates


__all__ = [
    "RewardBreakdown",
    "RewardComponent",
    "compute_rdiv",
    "compute_rmatch",
    "compute_total_reward",
    "score_candidate_sequence",
]

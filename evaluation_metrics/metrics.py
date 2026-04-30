from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from rdkit import DataStructs

from molecules.defaults import (
    DEFAULT_ACCEPTANCE_DICE_THRESHOLD,
    DEFAULT_ACCEPTANCE_TANIMOTO_THRESHOLD,
    DEFAULT_FINGERPRINT_NUM_BITS,
    DEFAULT_FINGERPRINT_RADIUS,
)
from molecules.fingerprints import build_morgan_fingerprint
from molecules.parsing import parse_molecule_text
from molecules.representations import MoleculeRecord, MoleculeRepresentation

from ._inputs import molecule_input_parts_from_value


_METRIC_MAPPING_TEXT_KEYS = ("text", "molecule_text", "selfies", "smiles")
_METRIC_REPRESENTATION_KEY_GROUPS = (
    ("selfies", ("selfies",)),
    ("smiles", ("smiles",)),
)


@dataclass(frozen=True)
class EvaluationMetricConfig:
    acceptance_dice_threshold: float = DEFAULT_ACCEPTANCE_DICE_THRESHOLD
    # NCircles is disabled by default because exact clique search is exponential.
    compute_n_circles: bool = False
    n_circles_tanimoto_threshold: float = DEFAULT_ACCEPTANCE_TANIMOTO_THRESHOLD
    fingerprint_radius: int = DEFAULT_FINGERPRINT_RADIUS
    fingerprint_num_bits: int = DEFAULT_FINGERPRINT_NUM_BITS
    n_circles_exact_max_molecules: int = 64

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.acceptance_dice_threshold) <= 1.0:
            raise ValueError("acceptance_dice_threshold must be within [0.0, 1.0].")
        if not 0.0 <= float(self.n_circles_tanimoto_threshold) <= 1.0:
            raise ValueError("n_circles_tanimoto_threshold must be within [0.0, 1.0].")
        if int(self.fingerprint_radius) < 0:
            raise ValueError("fingerprint_radius must be non-negative.")
        if int(self.fingerprint_num_bits) <= 0:
            raise ValueError("fingerprint_num_bits must be positive.")
        if int(self.n_circles_exact_max_molecules) < 1:
            raise ValueError("n_circles_exact_max_molecules must be positive.")


@dataclass(frozen=True)
class MoleculeInput:
    text: str
    representation: MoleculeRepresentation = "auto"
    molecule_id: str | None = None


@dataclass(frozen=True)
class GenerationGroup:
    group_id: str
    candidates: tuple[MoleculeInput, ...]
    targets: tuple[MoleculeInput, ...]


@dataclass(frozen=True)
class PreparedMolecule:
    molecule_input: MoleculeInput
    record: MoleculeRecord
    fingerprint: object


@dataclass(frozen=True)
class CandidateAssessment:
    group_id: str
    candidate_id: str | None
    input_text: str
    input_representation: str
    canonical_smiles: str | None
    is_valid: bool
    is_duplicate_valid: bool
    is_accepted: bool
    max_dice_similarity: float
    best_target_smiles: str | None
    rejection_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "candidate_id": self.candidate_id,
            "input_text": self.input_text,
            "input_representation": self.input_representation,
            "canonical_smiles": self.canonical_smiles,
            "is_valid": self.is_valid,
            "is_duplicate_valid": self.is_duplicate_valid,
            "is_accepted": self.is_accepted,
            "max_dice_similarity": self.max_dice_similarity,
            "best_target_smiles": self.best_target_smiles,
            "rejection_reason": self.rejection_reason,
        }


@dataclass(frozen=True)
class EvaluationMetricsResult:
    config: EvaluationMetricConfig
    num_groups: int
    num_candidates: int
    num_valid_candidates: int
    num_unique_valid_molecules: int
    num_accepted: int
    accepted_unique_count: int
    n_circles: int
    n_circles_exact: bool
    valid_fraction: float
    internal_diversity: float
    mean_max_dice_similarity: float
    prefix_valid_fraction: float
    prefix_duplicate_fraction: float
    prefix_duplicate_valid_fraction: float
    prefix_average_max_dice_similarity: float
    prefix_accepted_unique_internal_diversity: float
    prefix_valid_internal_diversity: float
    accepted_unique_smiles: tuple[str, ...] = field(default_factory=tuple)
    novelty_count: int = 0
    novelty_fraction: float = 0.0
    novel_accepted_unique_smiles: tuple[str, ...] = field(default_factory=tuple)
    candidate_assessments: tuple[CandidateAssessment, ...] = field(default_factory=tuple)

    def to_dict(self, *, include_assessments: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "num_groups": self.num_groups,
            "num_candidates": self.num_candidates,
            "num_valid_candidates": self.num_valid_candidates,
            "num_unique_valid_molecules": self.num_unique_valid_molecules,
            "num_accepted": self.num_accepted,
            "accepted_unique_count": self.accepted_unique_count,
            "n_circles": self.n_circles,
            "n_circles_exact": self.n_circles_exact,
            "valid_fraction": self.valid_fraction,
            "internal_diversity": self.internal_diversity,
            "mean_max_dice_similarity": self.mean_max_dice_similarity,
            "prefix_valid_fraction": self.prefix_valid_fraction,
            "prefix_duplicate_fraction": self.prefix_duplicate_fraction,
            "prefix_duplicate_valid_fraction": self.prefix_duplicate_valid_fraction,
            "prefix_average_max_dice_similarity": self.prefix_average_max_dice_similarity,
            "prefix_accepted_unique_internal_diversity": (
                self.prefix_accepted_unique_internal_diversity
            ),
            "prefix_valid_internal_diversity": self.prefix_valid_internal_diversity,
            "accepted_unique_smiles": list(self.accepted_unique_smiles),
            "novelty_count": self.novelty_count,
            "novelty_fraction": self.novelty_fraction,
            "novel_accepted_unique_smiles": list(self.novel_accepted_unique_smiles),
            "acceptance_dice_threshold": self.config.acceptance_dice_threshold,
            "compute_n_circles": self.config.compute_n_circles,
            "n_circles_tanimoto_threshold": self.config.n_circles_tanimoto_threshold,
            "fingerprint_radius": self.config.fingerprint_radius,
            "fingerprint_num_bits": self.config.fingerprint_num_bits,
        }
        if include_assessments:
            payload["candidate_assessments"] = [
                assessment.to_dict() for assessment in self.candidate_assessments
            ]
        return payload


def _as_molecule_input(
    value: MoleculeInput | Mapping[str, Any] | str,
    *,
    default_representation: MoleculeRepresentation,
    fallback_id: str | None = None,
) -> MoleculeInput:
    if isinstance(value, MoleculeInput):
        return value
    parts = molecule_input_parts_from_value(
        value,
        default_representation=default_representation,
        fallback_id=fallback_id,
        mapping_text_keys=_METRIC_MAPPING_TEXT_KEYS,
        representation_key_groups=_METRIC_REPRESENTATION_KEY_GROUPS,
        strip_scalar_text=False,
        drop_empty_text=False,
    )
    if parts is None:
        return MoleculeInput(
            text="",
            representation=default_representation,
            molecule_id=fallback_id,
        )
    return MoleculeInput(
        text=parts.text,
        representation=parts.representation,  # type: ignore[arg-type]
        molecule_id=parts.molecule_id,
    )


def _parse_molecule_with_fingerprint(
    molecule: MoleculeInput,
    *,
    config: EvaluationMetricConfig,
) -> tuple[MoleculeRecord, object | None]:
    record = parse_molecule_text(molecule.text, representation=molecule.representation)
    fingerprint = build_morgan_fingerprint(
        record,
        radius=config.fingerprint_radius,
        n_bits=config.fingerprint_num_bits,
    )
    return record, fingerprint


def _prepare_molecule(
    molecule: MoleculeInput,
    *,
    config: EvaluationMetricConfig,
) -> PreparedMolecule | None:
    record, fingerprint = _parse_molecule_with_fingerprint(molecule, config=config)
    if not record.is_valid or record.canonical_smiles is None or fingerprint is None:
        return None
    return PreparedMolecule(
        molecule_input=molecule,
        record=record,
        fingerprint=fingerprint,
    )


def _prepare_targets(
    targets: Sequence[MoleculeInput],
    *,
    config: EvaluationMetricConfig,
) -> tuple[PreparedMolecule, ...]:
    prepared_by_smiles: dict[str, PreparedMolecule] = {}
    for target in targets:
        prepared = _prepare_molecule(target, config=config)
        if prepared is None:
            continue
        prepared_by_smiles.setdefault(prepared.record.canonical_smiles or "", prepared)
    return tuple(
        prepared
        for smiles, prepared in prepared_by_smiles.items()
        if smiles
    )


def _candidate_assessment(
    *,
    group_id: str,
    candidate: MoleculeInput,
    canonical_smiles: str | None,
    is_valid: bool,
    is_duplicate_valid: bool,
    is_accepted: bool,
    max_dice_similarity: float,
    best_target_smiles: str | None,
    rejection_reason: str | None,
) -> CandidateAssessment:
    return CandidateAssessment(
        group_id=group_id,
        candidate_id=candidate.molecule_id,
        input_text=candidate.text,
        input_representation=candidate.representation,
        canonical_smiles=canonical_smiles,
        is_valid=is_valid,
        is_duplicate_valid=is_duplicate_valid,
        is_accepted=is_accepted,
        max_dice_similarity=max_dice_similarity,
        best_target_smiles=best_target_smiles,
        rejection_reason=rejection_reason,
    )


def _best_dice_similarity(
    fingerprint: object,
    targets: Sequence[PreparedMolecule],
) -> tuple[float, str | None]:
    best_similarity = 0.0
    best_target_smiles: str | None = None
    for target in targets:
        similarity = float(DataStructs.DiceSimilarity(fingerprint, target.fingerprint))
        if similarity > best_similarity:
            best_similarity = similarity
            best_target_smiles = target.record.canonical_smiles
    return best_similarity, best_target_smiles


def _pairwise_tanimoto_sum(fingerprints: Sequence[object]) -> tuple[float, int]:
    total = 0.0
    pair_count = 0
    for index, fingerprint in enumerate(fingerprints):
        remaining = fingerprints[index + 1 :]
        if not remaining:
            continue
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprint, list(remaining))
        total += sum(float(value) for value in similarities)
        pair_count += len(similarities)
    return total, pair_count


def internal_diversity(fingerprints: Sequence[object]) -> float:
    if len(fingerprints) < 2:
        return 0.0
    tanimoto_sum, pair_count = _pairwise_tanimoto_sum(fingerprints)
    if pair_count == 0:
        return 0.0
    return 1.0 - (tanimoto_sum / pair_count)


def _compatible_adjacency_bits(
    fingerprints: Sequence[object],
    *,
    tanimoto_threshold: float,
) -> list[int]:
    adjacency = [0 for _ in fingerprints]
    for index, fingerprint in enumerate(fingerprints):
        remaining = fingerprints[index + 1 :]
        if not remaining:
            continue
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprint, list(remaining))
        for offset, similarity in enumerate(similarities, start=index + 1):
            if float(similarity) < tanimoto_threshold:
                adjacency[index] |= 1 << offset
                adjacency[offset] |= 1 << index
    return adjacency


def _maximum_clique_size_exact(adjacency: Sequence[int]) -> int:
    best = 0

    def expand(candidates: int, size: int) -> None:
        nonlocal best
        if size + candidates.bit_count() <= best:
            return
        if candidates == 0:
            best = max(best, size)
            return

        while candidates:
            if size + candidates.bit_count() <= best:
                return
            vertex_bit = candidates & -candidates
            vertex = vertex_bit.bit_length() - 1
            expand(candidates & adjacency[vertex], size + 1)
            candidates &= ~vertex_bit
        best = max(best, size)

    expand((1 << len(adjacency)) - 1, 0)
    return best


def _maximum_clique_size_greedy(adjacency: Sequence[int]) -> int:
    if not adjacency:
        return 0
    degrees = [bits.bit_count() for bits in adjacency]
    orderings = [
        sorted(range(len(adjacency)), key=lambda index: (-degrees[index], index)),
        sorted(range(len(adjacency)), key=lambda index: (degrees[index], index)),
    ]
    best = 0
    for ordering in orderings:
        chosen_bits = 0
        size = 0
        for vertex in ordering:
            if chosen_bits & ~(adjacency[vertex]) == 0:
                chosen_bits |= 1 << vertex
                size += 1
        best = max(best, size)
    return best


def n_circles(
    fingerprints: Sequence[object],
    *,
    tanimoto_threshold: float,
    exact_max_molecules: int,
) -> tuple[int, bool]:
    if len(fingerprints) < 2:
        return len(fingerprints), True
    adjacency = _compatible_adjacency_bits(
        fingerprints,
        tanimoto_threshold=tanimoto_threshold,
    )
    if len(fingerprints) <= exact_max_molecules:
        return _maximum_clique_size_exact(adjacency), True
    return _maximum_clique_size_greedy(adjacency), False


def evaluate_generation_groups(
    groups: Sequence[GenerationGroup],
    *,
    config: EvaluationMetricConfig | None = None,
) -> EvaluationMetricsResult:
    metric_config = config or EvaluationMetricConfig()
    assessments: list[CandidateAssessment] = []
    valid_seen_smiles: set[str] = set()
    accepted_by_smiles: dict[str, PreparedMolecule] = {}
    target_seen_smiles: set[str] = set()
    valid_candidate_fingerprints: list[object] = []
    max_dice_sum = 0.0
    duplicate_valid_candidate_count = 0

    for group in groups:
        target_references = _prepare_targets(group.targets, config=metric_config)
        target_seen_smiles.update(
            target.record.canonical_smiles
            for target in target_references
            if target.record.canonical_smiles is not None
        )
        for candidate in group.candidates:
            candidate_record, candidate_fp = _parse_molecule_with_fingerprint(
                candidate,
                config=metric_config,
            )
            canonical_smiles = candidate_record.canonical_smiles
            is_valid = (
                candidate_record.is_valid
                and canonical_smiles is not None
                and candidate_fp is not None
            )
            is_duplicate_valid = bool(is_valid and canonical_smiles in valid_seen_smiles)
            if is_valid and canonical_smiles is not None:
                assert candidate_fp is not None
                duplicate_valid_candidate_count += int(is_duplicate_valid)
                valid_seen_smiles.add(canonical_smiles)
                valid_candidate_fingerprints.append(candidate_fp)

            if not is_valid:
                assessments.append(
                    _candidate_assessment(
                        group_id=group.group_id,
                        candidate=candidate,
                        canonical_smiles=None,
                        is_valid=False,
                        is_duplicate_valid=False,
                        is_accepted=False,
                        max_dice_similarity=0.0,
                        best_target_smiles=None,
                        rejection_reason="invalid_molecule",
                    )
                )
                continue

            if not target_references:
                assessments.append(
                    _candidate_assessment(
                        group_id=group.group_id,
                        candidate=candidate,
                        canonical_smiles=canonical_smiles,
                        is_valid=True,
                        is_duplicate_valid=is_duplicate_valid,
                        is_accepted=False,
                        max_dice_similarity=0.0,
                        best_target_smiles=None,
                        rejection_reason="no_valid_targets",
                    )
                )
                continue

            assert canonical_smiles is not None
            assert candidate_fp is not None
            max_dice, best_target_smiles = _best_dice_similarity(
                candidate_fp,
                target_references,
            )
            max_dice_sum += max_dice
            is_accepted = max_dice > metric_config.acceptance_dice_threshold
            rejection_reason = None if is_accepted else "below_dice_threshold"
            if is_accepted:
                prepared_candidate = PreparedMolecule(
                    molecule_input=candidate,
                    record=candidate_record,
                    fingerprint=candidate_fp,
                )
                accepted_by_smiles.setdefault(canonical_smiles, prepared_candidate)

            assessments.append(
                _candidate_assessment(
                    group_id=group.group_id,
                    candidate=candidate,
                    canonical_smiles=canonical_smiles,
                    is_valid=True,
                    is_duplicate_valid=is_duplicate_valid,
                    is_accepted=is_accepted,
                    max_dice_similarity=max_dice,
                    best_target_smiles=best_target_smiles,
                    rejection_reason=rejection_reason,
                )
            )

    accepted_unique = tuple(accepted_by_smiles.values())
    accepted_fingerprints = tuple(molecule.fingerprint for molecule in accepted_unique)
    n_circles_value, n_circles_exact_value = (
        n_circles(
            accepted_fingerprints,
            tanimoto_threshold=metric_config.n_circles_tanimoto_threshold,
            exact_max_molecules=metric_config.n_circles_exact_max_molecules,
        )
        if metric_config.compute_n_circles
        else (0, False)
    )
    num_candidates = len(assessments)
    num_valid_candidates = sum(int(assessment.is_valid) for assessment in assessments)
    num_accepted = sum(int(assessment.is_accepted) for assessment in assessments)
    # Prefix-average metric: cumulative over all generated samples seen so far.
    prefix_valid_fraction = num_valid_candidates / max(num_candidates, 1)
    prefix_duplicate_fraction = duplicate_valid_candidate_count / max(num_candidates, 1)
    prefix_duplicate_valid_fraction = (
        duplicate_valid_candidate_count / max(num_valid_candidates, 1)
    )
    prefix_average_max_dice_similarity = max_dice_sum / max(num_candidates, 1)
    prefix_accepted_unique_internal_diversity = internal_diversity(accepted_fingerprints)
    prefix_valid_internal_diversity = internal_diversity(valid_candidate_fingerprints)
    novel_accepted_unique_smiles = tuple(
        smiles
        for smiles in sorted(accepted_by_smiles)
        if smiles not in target_seen_smiles
    )
    novelty_count = len(novel_accepted_unique_smiles)
    novelty_fraction = novelty_count / max(len(accepted_unique), 1)
    return EvaluationMetricsResult(
        config=metric_config,
        num_groups=len(groups),
        num_candidates=num_candidates,
        num_valid_candidates=num_valid_candidates,
        num_unique_valid_molecules=len(valid_seen_smiles),
        num_accepted=num_accepted,
        accepted_unique_count=len(accepted_unique),
        n_circles=n_circles_value,
        n_circles_exact=n_circles_exact_value,
        valid_fraction=prefix_valid_fraction,
        internal_diversity=prefix_accepted_unique_internal_diversity,
        mean_max_dice_similarity=prefix_average_max_dice_similarity,
        prefix_valid_fraction=prefix_valid_fraction,
        prefix_duplicate_fraction=prefix_duplicate_fraction,
        prefix_duplicate_valid_fraction=prefix_duplicate_valid_fraction,
        prefix_average_max_dice_similarity=prefix_average_max_dice_similarity,
        prefix_accepted_unique_internal_diversity=prefix_accepted_unique_internal_diversity,
        prefix_valid_internal_diversity=prefix_valid_internal_diversity,
        accepted_unique_smiles=tuple(sorted(accepted_by_smiles)),
        novelty_count=novelty_count,
        novelty_fraction=novelty_fraction,
        novel_accepted_unique_smiles=novel_accepted_unique_smiles,
        candidate_assessments=tuple(assessments),
    )


def evaluate_generated_molecules(
    candidates: Sequence[MoleculeInput | Mapping[str, Any] | str],
    targets: Sequence[MoleculeInput | Mapping[str, Any] | str],
    *,
    candidate_representation: MoleculeRepresentation = "selfies",
    target_representation: MoleculeRepresentation = "selfies",
    config: EvaluationMetricConfig | None = None,
) -> EvaluationMetricsResult:
    group = GenerationGroup(
        group_id="global",
        candidates=tuple(
            _as_molecule_input(
                candidate,
                default_representation=candidate_representation,
                fallback_id=f"candidate-{index:06d}",
            )
            for index, candidate in enumerate(candidates)
        ),
        targets=tuple(
            _as_molecule_input(
                target,
                default_representation=target_representation,
                fallback_id=f"target-{index:06d}",
            )
            for index, target in enumerate(targets)
        ),
    )
    return evaluate_generation_groups((group,), config=config)


__all__ = [
    "CandidateAssessment",
    "EvaluationMetricConfig",
    "EvaluationMetricsResult",
    "GenerationGroup",
    "MoleculeInput",
    "evaluate_generated_molecules",
    "evaluate_generation_groups",
    "internal_diversity",
    "n_circles",
]

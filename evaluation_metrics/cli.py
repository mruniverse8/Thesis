from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping

from src.io_utils import read_jsonl, write_json

from ._inputs import MoleculeInputParts, collect_molecule_input_parts, row_id
from .metrics import (
    EvaluationMetricConfig,
    GenerationGroup,
    MoleculeInput,
    evaluate_generation_groups,
)


_CLI_MAPPING_TEXT_KEYS = (
    "text",
    "molecule_text",
    "sampled_selfies",
    "generated_selfies",
    "selected_selfies",
    "selfies",
    "canonical_smiles",
    "smiles",
)
_CLI_REPRESENTATION_KEY_GROUPS = (("smiles", ("canonical_smiles", "smiles")),)


def _molecule_input_from_parts(parts: MoleculeInputParts) -> MoleculeInput:
    return MoleculeInput(
        text=parts.text,
        representation=parts.representation,  # type: ignore[arg-type]
        molecule_id=parts.molecule_id,
    )


def _collect_molecule_inputs(
    row: Mapping[str, object],
    *,
    keys: Iterable[str],
    default_representation: str,
) -> tuple[MoleculeInput, ...]:
    return tuple(
        _molecule_input_from_parts(parts)
        for parts in collect_molecule_input_parts(
            row,
            keys=keys,
            default_representation=default_representation,
            mapping_text_keys=_CLI_MAPPING_TEXT_KEYS,
            representation_key_groups=_CLI_REPRESENTATION_KEY_GROUPS,
        )
    )


def _candidate_inputs(row: Mapping[str, object]) -> tuple[MoleculeInput, ...]:
    selfies_candidates = _collect_molecule_inputs(
        row,
        keys=(
            "generated_selfies_list",
            "sampled_selfies_list",
            "candidate_selfies_list",
            "predicted_selfies_list",
            "candidates",
            "molecules",
            "generated_selfies",
            "sampled_selfies",
            "selected_selfies",
        ),
        default_representation="selfies",
    )
    smiles_candidates = _collect_molecule_inputs(
        row,
        keys=("canonical_smiles", "smiles"),
        default_representation="smiles",
    )
    return selfies_candidates + smiles_candidates


def _target_inputs(row: Mapping[str, object]) -> tuple[MoleculeInput, ...]:
    selfies_targets = _collect_molecule_inputs(
        row,
        keys=("target_selfies_list", "targets", "target_selfies"),
        default_representation="selfies",
    )
    smiles_targets = _collect_molecule_inputs(
        row,
        keys=("target_smiles_list", "target_smiles"),
        default_representation="smiles",
    )
    return selfies_targets + smiles_targets


def load_generation_groups_from_jsonl(
    generated_jsonl: str | Path,
    *,
    targets_jsonl: str | Path | None = None,
) -> tuple[GenerationGroup, ...]:
    generated_rows = read_jsonl(generated_jsonl)
    target_lookup: dict[str, Mapping[str, object]] = {}
    if targets_jsonl is not None:
        for index, row in enumerate(read_jsonl(targets_jsonl)):
            target_lookup[row_id(row, fallback=f"target-{index:06d}")] = row

    groups: list[GenerationGroup] = []
    for index, row in enumerate(generated_rows):
        group_id = row_id(row, fallback=f"group-{index:06d}")
        target_row = row if _target_inputs(row) else target_lookup.get(group_id, {})
        groups.append(
            GenerationGroup(
                group_id=group_id,
                candidates=_candidate_inputs(row),
                targets=_target_inputs(target_row),
            )
        )
    return tuple(groups)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute accepted-unique and internal-diversity metrics."
    )
    parser.add_argument(
        "--generated-jsonl",
        required=True,
        help="JSONL with generated molecules. Rows should include generated_selfies_list or sampled_selfies.",
    )
    parser.add_argument(
        "--targets-jsonl",
        default=None,
        help="Optional JSONL target lookup keyed by id/example_id when generated rows omit target_selfies_list.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path. Metrics are always printed to stdout.",
    )
    parser.add_argument("--acceptance-dice-threshold", type=float, default=0.7)
    parser.add_argument(
        "--compute-n-circles",
        action="store_true",
        help="Opt in to NCircles calculation. Disabled by default because exact clique search is exponential.",
    )
    parser.add_argument("--n-circles-tanimoto-threshold", type=float, default=0.6)
    parser.add_argument("--fingerprint-radius", type=int, default=2)
    parser.add_argument("--fingerprint-num-bits", type=int, default=2048)
    parser.add_argument("--n-circles-exact-max-molecules", type=int, default=64)
    parser.add_argument(
        "--include-assessments",
        action="store_true",
        help="Include per-candidate acceptance rows in the JSON output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups = load_generation_groups_from_jsonl(
        args.generated_jsonl,
        targets_jsonl=args.targets_jsonl,
    )
    result = evaluate_generation_groups(
        groups,
        config=EvaluationMetricConfig(
            acceptance_dice_threshold=args.acceptance_dice_threshold,
            compute_n_circles=bool(args.compute_n_circles),
            n_circles_tanimoto_threshold=args.n_circles_tanimoto_threshold,
            fingerprint_radius=args.fingerprint_radius,
            fingerprint_num_bits=args.fingerprint_num_bits,
            n_circles_exact_max_molecules=args.n_circles_exact_max_molecules,
        ),
    )
    payload = result.to_dict(include_assessments=bool(args.include_assessments))
    if args.output:
        write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()


__all__ = ["load_generation_groups_from_jsonl"]

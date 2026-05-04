#!/usr/bin/env python3
"""Merge segmented GFlowNet beam-evaluation outputs.

This script expects a run directory containing segment folders named like
``test_iter_000000_to_000500``. It merges raw generation rows, validates that
segments were produced with compatible checkpoint/generation/metric settings,
and recomputes final metrics globally with ``evaluate_generation_groups``.

The model is not loaded here. Final metrics are recalculated from JSONL rows
containing generated molecules and targets. Segment metrics are used only for
summaries and plots; they are not summed into the final result.

Example:
    python scripts_merge/merge_segmented_gflownet_eval.py \\
      --dir-run /content/drive/MyDrive/run/segments \\
      --output-dir /content/drive/MyDrive/run/merged \\
      --expected-test-examples 10000
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation_metrics import (  # noqa: E402
    EvaluationMetricConfig,
    GenerationGroup,
    MoleculeInput,
    evaluate_generation_groups,
)


SEGMENT_DIR_PATTERN = "test_iter_*_to_*"
SEGMENT_NAME_RE = re.compile(r"^test_iter_(?P<start>\d+)_to_(?P<end>\d+)$")
GENERATION_KEYS = (
    "generated_selfies_list",
    "sampled_selfies_list",
    "candidate_selfies_list",
    "predicted_selfies_list",
)
TARGET_KEYS = ("target_selfies_list", "targets", "target_selfies")
CONSISTENCY_KEYS = (
    "eval_run_id",
    "checkpoint_source",
    "trained_adapter_source",
    "repo_branch",
    "config_override",
    "generation_num_beams",
    "generation_num_return_sequences",
    "generation_early_stopping",
    "generation_length_penalty",
    "rollout_max_stage_new_tokens",
    "rollout_max_molecules_per_sequence",
    "acceptance_dice_threshold",
    "compute_n_circles",
    "n_circles_tanimoto_threshold",
    "n_circles_exact_max_molecules",
    "fingerprint_radius",
    "fingerprint_num_bits",
)


@dataclass(frozen=True)
class SegmentDirectory:
    path: Path
    name: str
    start_iter: int
    end_iter: int
    manifest: dict[str, Any]
    metrics: dict[str, Any] | None
    rows: tuple[dict[str, Any], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge segmented GFlowNet evaluation generations and recompute "
            "global metrics from the merged JSONL."
        )
    )
    parser.add_argument(
        "--dir-run",
        required=True,
        type=Path,
        help=(
            "Directory containing segment folders, e.g. "
            "DIR_RUN/test_iter_000000_to_000500/."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where merged metrics, merged JSONL, CSV summaries, and plots are written.",
    )
    parser.add_argument(
        "--segment-pattern",
        default=SEGMENT_DIR_PATTERN,
        help=f"Glob used to find segment folders inside --dir-run. Default: {SEGMENT_DIR_PATTERN}",
    )
    parser.add_argument(
        "--expected-test-examples",
        type=int,
        default=None,
        help="Expected full test-set size. Used to report missing dataset indices.",
    )
    parser.add_argument(
        "--strict-full-coverage",
        action="store_true",
        help="Fail if --expected-test-examples is set and any dataset index is missing.",
    )
    parser.add_argument(
        "--allow-missing-manifest",
        action="store_true",
        help="Continue when a segment has no manifest.json, inferring range from the folder name.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip matplotlib plot generation.",
    )
    parser.add_argument(
        "--omit-assessments",
        action="store_true",
        help="Do not embed candidate_assessments in metrics_merged.json. They are still written to JSONL.",
    )
    parser.add_argument(
        "--acceptance-dice-threshold",
        type=float,
        default=None,
        help="Override the acceptance Dice threshold from manifest/default config.",
    )
    parser.add_argument(
        "--compute-n-circles",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override NCircles calculation from manifest/default config.",
    )
    parser.add_argument(
        "--n-circles-tanimoto-threshold",
        type=float,
        default=None,
        help="Override the NCircles Tanimoto threshold from manifest/default config.",
    )
    parser.add_argument(
        "--fingerprint-radius",
        type=int,
        default=None,
        help="Override Morgan fingerprint radius from manifest/default config.",
    )
    parser.add_argument(
        "--fingerprint-num-bits",
        type=int,
        default=None,
        help="Override Morgan fingerprint bit count from manifest/default config.",
    )
    parser.add_argument(
        "--n-circles-exact-max-molecules",
        type=int,
        default=None,
        help="Override exact NCircles molecule limit from manifest/default config.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return load_json(path)


def load_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object in {path}:{line_number}")
            rows.append(row)
    return tuple(rows)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=False) + "\n")


def parse_segment_name(path: Path) -> tuple[int, int]:
    match = SEGMENT_NAME_RE.match(path.name)
    if match is None:
        raise ValueError(
            f"Segment folder must be named test_iter_XXXXXX_to_YYYYYY: {path}"
        )
    return int(match.group("start")), int(match.group("end"))


def load_segment_directory(path: Path, *, allow_missing_manifest: bool) -> SegmentDirectory:
    start_iter, end_iter = parse_segment_name(path)
    generations_path = path / "generations.jsonl"
    manifest_path = path / "manifest.json"
    metrics_path = path / "metrics_segment.json"

    if not generations_path.exists():
        raise FileNotFoundError(f"Missing generations.jsonl in {path}")

    manifest = load_json_if_exists(manifest_path)
    if manifest is None:
        if not allow_missing_manifest:
            raise FileNotFoundError(
                f"Missing manifest.json in {path}. Pass --allow-missing-manifest to infer range only."
            )
        manifest = {
            "segment_name": path.name,
            "start_iter": start_iter,
            "end_iter": end_iter,
            "_manifest_missing": True,
        }

    rows = load_jsonl(generations_path)
    metrics = load_json_if_exists(metrics_path)
    return SegmentDirectory(
        path=path,
        name=path.name,
        start_iter=start_iter,
        end_iter=end_iter,
        manifest=manifest,
        metrics=metrics,
        rows=rows,
    )


def discover_segments(
    dir_run: Path,
    *,
    segment_pattern: str,
    allow_missing_manifest: bool,
) -> tuple[SegmentDirectory, ...]:
    if not dir_run.exists():
        raise FileNotFoundError(f"Missing --dir-run directory: {dir_run}")
    if not dir_run.is_dir():
        raise ValueError(f"--dir-run must be a directory: {dir_run}")

    segment_paths = sorted(
        (path for path in dir_run.glob(segment_pattern) if path.is_dir()),
        key=lambda path: parse_segment_name(path),
    )
    if not segment_paths:
        raise FileNotFoundError(f"No segment directories matched {segment_pattern} in {dir_run}")
    return tuple(
        load_segment_directory(path, allow_missing_manifest=allow_missing_manifest)
        for path in segment_paths
    )


def manifest_value(manifest: Mapping[str, Any], key: str) -> Any:
    value = manifest.get(key)
    # Old drafts may use num_beams/num_return_sequences in rows or metrics; keep
    # the primary manifest contract simple but avoid false mismatches on absent keys.
    return value


def validate_manifest_consistency(segments: Sequence[SegmentDirectory]) -> list[str]:
    warnings: list[str] = []
    reference = segments[0].manifest
    for segment in segments[1:]:
        mismatches: dict[str, dict[str, Any]] = {}
        for key in CONSISTENCY_KEYS:
            ref_value = manifest_value(reference, key)
            current_value = manifest_value(segment.manifest, key)
            if ref_value != current_value:
                mismatches[key] = {
                    "reference": ref_value,
                    "current": current_value,
                    "segment": segment.name,
                }
        if mismatches:
            raise ValueError(
                "Manifest mismatch. Segments must come from the same checkpoint/config/settings: "
                + json.dumps(mismatches, indent=2)
            )

    missing_keys = [
        key for key in CONSISTENCY_KEYS if all(key not in segment.manifest for segment in segments)
    ]
    if missing_keys:
        warnings.append(
            "No segment manifest contained these consistency keys: "
            + ", ".join(missing_keys)
        )
    return warnings


def validate_segment_ranges(segments: Sequence[SegmentDirectory]) -> None:
    previous_end: int | None = None
    for segment in segments:
        manifest_start = int(segment.manifest.get("start_iter", segment.start_iter))
        manifest_end = int(segment.manifest.get("end_iter", segment.end_iter))
        if manifest_start != segment.start_iter or manifest_end != segment.end_iter:
            raise ValueError(
                f"Folder range and manifest range differ for {segment.name}: "
                f"folder=[{segment.start_iter}, {segment.end_iter}), "
                f"manifest=[{manifest_start}, {manifest_end})"
            )
        if segment.start_iter >= segment.end_iter:
            raise ValueError(f"Invalid empty/inverted segment range: {segment.name}")
        if previous_end is not None and segment.start_iter < previous_end:
            raise ValueError(
                f"Overlapping segment ranges around {segment.name}: "
                f"previous_end={previous_end}, current_start={segment.start_iter}"
            )
        previous_end = segment.end_iter


def values_as_list(row: Mapping[str, Any], keys: Sequence[str]) -> list[Any]:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            return value
        return [value]
    return []


def generated_selfies_from_row(row: Mapping[str, Any]) -> list[str]:
    molecules = values_as_list(row, GENERATION_KEYS)
    if molecules:
        return [str(value) for value in molecules if str(value).strip()]

    # Compatibility fallback for rich beam rows. New notebook rows should also
    # save generated_selfies_list so the merge format is explicit.
    flattened: list[str] = []
    for beam in values_as_list(row, ("beam_results",)):
        if not isinstance(beam, Mapping):
            continue
        for selfies in values_as_list(beam, ("selfies",)):
            text = str(selfies).strip()
            if text:
                flattened.append(text)
    return flattened


def target_selfies_from_row(row: Mapping[str, Any]) -> list[str]:
    return [str(value) for value in values_as_list(row, TARGET_KEYS) if str(value).strip()]


def row_group_id(row: Mapping[str, Any]) -> str:
    for key in ("example_id", "id", "group_id", "description_id", "rollout_id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return f"dataset_index-{int(row['dataset_index']):06d}"


def validate_and_merge_rows(
    segments: Sequence[SegmentDirectory],
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    rows_by_index: dict[int, dict[str, Any]] = {}
    empty_candidate_rows = 0

    for segment in segments:
        seen_in_segment: set[int] = set()
        for row_number, row in enumerate(segment.rows, start=1):
            if "dataset_index" not in row:
                raise ValueError(f"Missing dataset_index in {segment.path}/generations.jsonl row {row_number}")
            dataset_index = int(row["dataset_index"])
            if dataset_index < segment.start_iter or dataset_index >= segment.end_iter:
                raise ValueError(
                    f"Row dataset_index={dataset_index} is outside segment range "
                    f"[{segment.start_iter}, {segment.end_iter}) in {segment.name}"
                )
            if dataset_index in seen_in_segment:
                raise ValueError(f"Duplicate dataset_index={dataset_index} inside {segment.name}")
            if dataset_index in rows_by_index:
                raise ValueError(f"Duplicate dataset_index across segments: {dataset_index}")
            if not target_selfies_from_row(row):
                raise ValueError(f"Missing target_selfies_list for dataset_index={dataset_index}")
            if not generated_selfies_from_row(row):
                empty_candidate_rows += 1

            normalized = dict(row)
            normalized["dataset_index"] = dataset_index
            if "generated_selfies_list" not in normalized:
                normalized["generated_selfies_list"] = generated_selfies_from_row(row)
            rows_by_index[dataset_index] = normalized
            seen_in_segment.add(dataset_index)

    if empty_candidate_rows:
        warnings.append(f"{empty_candidate_rows} rows have no generated candidates.")

    return [rows_by_index[index] for index in sorted(rows_by_index)], warnings


def build_generation_groups(rows: Sequence[Mapping[str, Any]]) -> tuple[GenerationGroup, ...]:
    groups: list[GenerationGroup] = []
    for row in rows:
        dataset_index = int(row["dataset_index"])
        groups.append(
            GenerationGroup(
                group_id=row_group_id(row),
                candidates=tuple(
                    MoleculeInput(
                        text=selfies,
                        representation="selfies",
                        molecule_id=f"{dataset_index}:candidate:{index:06d}",
                    )
                    for index, selfies in enumerate(generated_selfies_from_row(row))
                ),
                targets=tuple(
                    MoleculeInput(
                        text=selfies,
                        representation="selfies",
                        molecule_id=f"{dataset_index}:target:{index:06d}",
                    )
                    for index, selfies in enumerate(target_selfies_from_row(row))
                ),
            )
        )
    return tuple(groups)


def first_available_manifest(segments: Sequence[SegmentDirectory]) -> Mapping[str, Any]:
    for segment in segments:
        if not segment.manifest.get("_manifest_missing"):
            return segment.manifest
    return segments[0].manifest


def metric_config_from_inputs(args: argparse.Namespace, segments: Sequence[SegmentDirectory]) -> EvaluationMetricConfig:
    manifest = first_available_manifest(segments)

    def choose(name: str, fallback: Any) -> Any:
        override = getattr(args, name)
        if override is not None:
            return override
        return manifest.get(name, fallback)

    return EvaluationMetricConfig(
        acceptance_dice_threshold=float(choose("acceptance_dice_threshold", 0.7)),
        compute_n_circles=bool(choose("compute_n_circles", False)),
        n_circles_tanimoto_threshold=float(choose("n_circles_tanimoto_threshold", 0.6)),
        fingerprint_radius=int(choose("fingerprint_radius", 2)),
        fingerprint_num_bits=int(choose("fingerprint_num_bits", 2048)),
        n_circles_exact_max_molecules=int(choose("n_circles_exact_max_molecules", 64)),
    )


def summarize_segment(segment: SegmentDirectory) -> dict[str, Any]:
    row_count = len(segment.rows)
    candidate_count = sum(len(generated_selfies_from_row(row)) for row in segment.rows)
    payload: dict[str, Any] = {
        "segment_name": segment.name,
        "start_iter": segment.start_iter,
        "end_iter": segment.end_iter,
        "row_count": row_count,
        "candidate_count": candidate_count,
        "metrics_path": str(segment.path / "metrics_segment.json")
        if (segment.path / "metrics_segment.json").exists()
        else "",
    }
    if segment.metrics:
        for key in (
            "num_groups",
            "num_candidates",
            "num_valid_candidates",
            "num_unique_valid_molecules",
            "num_accepted",
            "accepted_unique_count",
            "internal_diversity",
            "novelty_fraction",
            "mean_max_dice_similarity",
        ):
            if key in segment.metrics:
                payload[key] = segment.metrics[key]
        num_candidates = int(segment.metrics.get("num_candidates") or 0)
        num_valid = int(segment.metrics.get("num_valid_candidates") or 0)
        payload["valid_fraction"] = num_valid / max(num_candidates, 1)
    return payload


def write_segment_summary_csv(path: Path, summaries: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "segment_name",
        "start_iter",
        "end_iter",
        "row_count",
        "candidate_count",
        "num_groups",
        "num_candidates",
        "num_valid_candidates",
        "num_unique_valid_molecules",
        "num_accepted",
        "accepted_unique_count",
        "valid_fraction",
        "internal_diversity",
        "novelty_fraction",
        "mean_max_dice_similarity",
        "metrics_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary)


def write_candidate_assessments_jsonl(path: Path, result: Any) -> None:
    rows = [assessment.to_dict() for assessment in result.candidate_assessments]
    write_jsonl(path, rows)


def coverage_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_test_examples: int | None,
) -> dict[str, Any]:
    if not rows:
        return {
            "min_dataset_index": None,
            "max_dataset_index": None,
            "missing_count": 0,
            "missing_indices_first_50": [],
            "extra_count": 0,
            "extra_indices_first_50": [],
        }

    indices = {int(row["dataset_index"]) for row in rows}
    min_index = min(indices)
    max_index = max(indices)
    payload: dict[str, Any] = {
        "min_dataset_index": min_index,
        "max_dataset_index": max_index,
        "observed_index_span_count": max_index - min_index + 1,
        "missing_count": 0,
        "missing_indices_first_50": [],
        "extra_count": 0,
        "extra_indices_first_50": [],
    }
    if expected_test_examples is not None:
        expected = set(range(expected_test_examples))
        missing = sorted(expected - indices)
        extra = sorted(indices - expected)
        payload.update(
            {
                "missing_count": len(missing),
                "missing_indices_first_50": missing[:50],
                "extra_count": len(extra),
                "extra_indices_first_50": extra[:50],
            }
        )
    return payload


def console_summary(
    *,
    segments: Sequence[SegmentDirectory],
    rows: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    coverage: Mapping[str, Any],
    output_dir: Path,
    warnings: Sequence[str],
) -> None:
    valid_fraction = metrics["num_valid_candidates"] / max(metrics["num_candidates"], 1)
    print("\nMerged GFlowNet Evaluation")
    print("==========================")
    print(f"segments: {len(segments)}")
    print(f"merged rows: {len(rows)}")
    print(f"dataset index range: {coverage['min_dataset_index']}..{coverage['max_dataset_index']}")
    print(f"missing indices: {coverage['missing_count']}")
    print(f"extra indices: {coverage['extra_count']}")
    print("")
    print("Global metrics recomputed from merged generations")
    print(f"num_groups: {metrics['num_groups']}")
    print(f"num_candidates: {metrics['num_candidates']}")
    print(f"num_valid_candidates: {metrics['num_valid_candidates']}")
    print(f"valid_fraction: {valid_fraction:.6f}")
    print(f"num_accepted: {metrics['num_accepted']}")
    print(f"accepted_unique_count: {metrics['accepted_unique_count']}")
    print(f"novelty_count: {metrics['novelty_count']}")
    print(f"novelty_fraction: {metrics['novelty_fraction']:.6f}")
    print(f"internal_diversity: {metrics['internal_diversity']:.6f}")
    print(f"mean_max_dice_similarity: {metrics['mean_max_dice_similarity']:.6f}")
    print(f"n_circles: {metrics['n_circles']}")
    print(f"n_circles_exact: {metrics['n_circles_exact']}")
    print("")
    print(f"outputs: {output_dir}")
    if warnings:
        print("")
        print("Warnings")
        for warning in warnings:
            print(f"- {warning}")


def plot_outputs(
    *,
    output_dir: Path,
    segment_summaries: Sequence[Mapping[str, Any]],
    result: Any,
) -> list[str]:
    plot_paths: list[str] = []
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        message = "matplotlib is not installed; plot generation skipped.\n"
        (plots_dir / "plots_skipped.txt").write_text(message, encoding="utf-8")
        return plot_paths

    x = list(range(len(segment_summaries)))
    labels = [str(summary["segment_name"]) for summary in segment_summaries]

    def save_current(name: str) -> None:
        path = plots_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        plot_paths.append(str(path))

    if segment_summaries:
        accepted = [float(summary.get("accepted_unique_count", 0.0) or 0.0) for summary in segment_summaries]
        valid = [float(summary.get("valid_fraction", 0.0) or 0.0) for summary in segment_summaries]
        row_counts = [int(summary.get("row_count", 0) or 0) for summary in segment_summaries]
        candidate_counts = [int(summary.get("candidate_count", 0) or 0) for summary in segment_summaries]
        cumulative_rows: list[int] = []
        cumulative_candidates: list[int] = []
        row_total = 0
        candidate_total = 0
        for rows, candidates in zip(row_counts, candidate_counts):
            row_total += rows
            candidate_total += candidates
            cumulative_rows.append(row_total)
            cumulative_candidates.append(candidate_total)

        plt.figure(figsize=(max(8, len(x) * 0.5), 4))
        plt.plot(x, valid, marker="o")
        plt.xticks(x, labels, rotation=60, ha="right", fontsize=8)
        plt.ylabel("Valid fraction")
        plt.title("Segment valid fraction")
        save_current("segment_valid_fraction.png")

        plt.figure(figsize=(max(8, len(x) * 0.5), 4))
        plt.bar(x, accepted)
        plt.xticks(x, labels, rotation=60, ha="right", fontsize=8)
        plt.ylabel("Accepted unique count")
        plt.title("Segment accepted-unique count")
        save_current("segment_accepted_unique_count.png")

        plt.figure(figsize=(max(8, len(x) * 0.5), 4))
        plt.plot(x, cumulative_rows, marker="o")
        plt.xticks(x, labels, rotation=60, ha="right", fontsize=8)
        plt.ylabel("Rows")
        plt.title("Cumulative merged rows by segment")
        save_current("cumulative_rows_by_segment.png")

        plt.figure(figsize=(max(8, len(x) * 0.5), 4))
        plt.plot(x, cumulative_candidates, marker="o")
        plt.xticks(x, labels, rotation=60, ha="right", fontsize=8)
        plt.ylabel("Candidates")
        plt.title("Cumulative candidates by segment")
        save_current("cumulative_candidates_by_segment.png")

    assessments = list(result.candidate_assessments)
    if assessments:
        dice_values = [assessment.max_dice_similarity for assessment in assessments]
        accepted_count = sum(int(assessment.is_accepted) for assessment in assessments)
        valid_count = sum(int(assessment.is_valid) for assessment in assessments)
        invalid_count = len(assessments) - valid_count

        plt.figure(figsize=(7, 4))
        plt.hist(dice_values, bins=30)
        plt.xlabel("Max Dice similarity to targets")
        plt.ylabel("Candidate count")
        plt.title("Merged candidate max-Dice distribution")
        save_current("merged_max_dice_histogram.png")

        plt.figure(figsize=(6, 4))
        plt.bar(["invalid", "valid", "accepted"], [invalid_count, valid_count, accepted_count])
        plt.ylabel("Candidate count")
        plt.title("Merged candidate validity and acceptance")
        save_current("merged_candidate_validity_acceptance.png")

    return plot_paths


def main() -> None:
    args = parse_args()
    dir_run = args.dir_run.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    segments = discover_segments(
        dir_run,
        segment_pattern=args.segment_pattern,
        allow_missing_manifest=bool(args.allow_missing_manifest),
    )
    warnings = []
    warnings.extend(validate_manifest_consistency(segments))
    validate_segment_ranges(segments)

    merged_rows, row_warnings = validate_and_merge_rows(segments)
    warnings.extend(row_warnings)
    coverage = coverage_summary(
        merged_rows,
        expected_test_examples=args.expected_test_examples,
    )
    if (
        args.strict_full_coverage
        and args.expected_test_examples is not None
        and int(coverage["missing_count"]) > 0
    ):
        raise ValueError(
            f"Full coverage check failed: {coverage['missing_count']} missing indices."
        )

    metric_config = metric_config_from_inputs(args, segments)
    groups = build_generation_groups(merged_rows)
    result = evaluate_generation_groups(groups, config=metric_config)
    metrics_payload = {
        "schema_version": 1,
        "source_dir_run": str(dir_run),
        "source_segment_count": len(segments),
        "num_merged_rows": len(merged_rows),
        "expected_test_examples": args.expected_test_examples,
        **coverage,
        **result.to_dict(include_assessments=not args.omit_assessments),
    }

    merged_generations_path = output_dir / "generations_merged.jsonl"
    metrics_path = output_dir / "metrics_merged.json"
    assessments_path = output_dir / "candidate_assessments.jsonl"
    segment_summary_path = output_dir / "segment_summary.csv"
    merge_manifest_path = output_dir / "merge_manifest.json"

    segment_summaries = [summarize_segment(segment) for segment in segments]
    write_jsonl(merged_generations_path, merged_rows)
    write_json(metrics_path, metrics_payload)
    write_candidate_assessments_jsonl(assessments_path, result)
    write_segment_summary_csv(segment_summary_path, segment_summaries)

    plot_paths: list[str] = []
    if not args.skip_plots:
        plot_paths = plot_outputs(
            output_dir=output_dir,
            segment_summaries=segment_summaries,
            result=result,
        )

    write_json(
        merge_manifest_path,
        {
            "schema_version": 1,
            "source_dir_run": str(dir_run),
            "output_dir": str(output_dir),
            "source_segments": [
                {
                    "segment_name": segment.name,
                    "path": str(segment.path),
                    "start_iter": segment.start_iter,
                    "end_iter": segment.end_iter,
                    "row_count": len(segment.rows),
                }
                for segment in segments
            ],
            "consistency_keys": list(CONSISTENCY_KEYS),
            "metric_config": {
                "acceptance_dice_threshold": metric_config.acceptance_dice_threshold,
                "compute_n_circles": metric_config.compute_n_circles,
                "n_circles_tanimoto_threshold": metric_config.n_circles_tanimoto_threshold,
                "fingerprint_radius": metric_config.fingerprint_radius,
                "fingerprint_num_bits": metric_config.fingerprint_num_bits,
                "n_circles_exact_max_molecules": metric_config.n_circles_exact_max_molecules,
            },
            "coverage": coverage,
            "outputs": {
                "merged_generations": str(merged_generations_path),
                "metrics": str(metrics_path),
                "candidate_assessments": str(assessments_path),
                "segment_summary": str(segment_summary_path),
                "plots": plot_paths,
            },
            "warnings": warnings,
        },
    )

    console_summary(
        segments=segments,
        rows=merged_rows,
        metrics=metrics_payload,
        coverage=coverage,
        output_dir=output_dir,
        warnings=warnings,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import pytest

pytest.importorskip("selfies")
pytest.importorskip("rdkit")

from molecules.collection.filtering import CollectionMetricConfig, prepare_reference_groups
from notebooks.biot5_collection_review_support import (
    assess_biot5_native_generation_output,
    assess_generation_output,
    build_prefixed_smiles_prompt,
    build_tagged_smiles_prompt,
    clean_biot5_selfies_text,
    decode_biot5_selfies,
    derive_selfies_from_smiles,
    extract_prefixed_smiles,
    extract_smiles_candidate,
    extract_strict_smiles,
    extract_tagged_smiles,
    filter_selfies,
    summarize_biot5_native_review_records,
    summarize_review_records,
)


def test_build_prefixed_smiles_prompt_contains_expected_contract() -> None:
    prompt = build_prefixed_smiles_prompt("A simple alcohol.")

    assert "generate exactly one molecule SMILES" in prompt
    assert "SMILES: <molecule>" in prompt
    assert "Input: A simple alcohol." in prompt
    assert prompt.endswith("Output: SMILES: ")


def test_build_tagged_smiles_prompt_contains_expected_contract() -> None:
    prompt = build_tagged_smiles_prompt("A simple alcohol.")

    assert "generate exactly one molecule SMILES" in prompt
    assert "<SMILES>" in prompt
    assert "Input: A simple alcohol." in prompt
    assert prompt.endswith("Output: <SMILES>")


def test_extract_tagged_smiles_returns_inner_text() -> None:
    extracted = extract_tagged_smiles("prefix <SMILES> CCO </SMILES> suffix")

    assert extracted == "CCO"


def test_extract_prefixed_smiles_returns_payload() -> None:
    extracted = extract_prefixed_smiles("Output: SMILES: OCC")

    assert extracted == "OCC"


def test_extract_strict_smiles_prefers_tagged_output() -> None:
    extracted_smiles, extraction_mode = extract_strict_smiles(
        "SMILES: CO <SMILES>CCO</SMILES>"
    )

    assert extracted_smiles == "CCO"
    assert extraction_mode == "tag_only"


def test_extract_smiles_candidate_recovers_noisy_smiles() -> None:
    extracted = extract_smiles_candidate(
        "The generated molecule is CN(N=O)C(=N)O and should be reviewed."
    )

    assert extracted == "CN(N=O)C(=N)O"


def test_extract_strict_smiles_returns_none_for_plain_english() -> None:
    extracted_smiles, extraction_mode = extract_strict_smiles(
        "This output is explanatory prose and does not contain a tagged molecule."
    )

    assert extracted_smiles is None
    assert extraction_mode is None


def test_derive_selfies_from_smiles_uses_canonical_smiles() -> None:
    derived = derive_selfies_from_smiles("OCC")

    assert derived == "[C][C][O]"


def test_clean_biot5_selfies_text_strips_t5_wrappers_and_bom_eom() -> None:
    cleaned = clean_biot5_selfies_text(" <pad> <bom> [C] [C] [O] <eom> </s> ")

    assert cleaned == "[C][C][O]"


def test_filter_selfies_recovers_bracketed_tokens_from_noisy_text() -> None:
    filtered = filter_selfies("Answer:[C][C][O]trailing")

    assert filtered == "[C][C][O]"


def test_decode_biot5_selfies_uses_filter_fallback_when_needed() -> None:
    decoded = decode_biot5_selfies("Answer:[C][C][O]trailing")

    assert decoded["cleaned_selfies"] == "Answer:[C][C][O]trailing"
    assert decoded["parsed_selfies"] == "[C][C][O]"
    assert decoded["filtered_selfies"] == "[C][C][O]"
    assert decoded["selected_selfies"] == "[C][C][O]"
    assert decoded["used_filter_selfies_fallback"] is True
    assert decoded["decoded_smiles"] == "CCO"
    assert decoded["is_valid_selfies"] is True


def test_assess_generation_output_uses_prefix_mode_without_loose_fallback() -> None:
    metric_config = CollectionMetricConfig(
        fingerprint_radius=2,
        fingerprint_num_bits=2048,
        acceptance_dice_threshold=0.7,
    )
    records = [
        {
            "id": "desc-1",
            "description": "small alcohol molecule",
            "selfies": "[C][C][O]",
            "source_smiles": "CCO",
        }
    ]
    references = prepare_reference_groups(records, metric_config)["small alcohol molecule"]

    assessment = assess_generation_output(
        description_id="desc-1",
        description="small alcohol molecule",
        prompt_variant="prefixed_smiles",
        candidate_index=0,
        raw_prediction_text="SMILES: OCC",
        references=references,
        metric_config=metric_config,
        generation_config_name="short_contrastive_fast",
        elapsed_seconds_batch=3.0,
        seconds_per_sample_batch=0.1,
    )

    assert assessment["strict_extracted_smiles"] == "OCC"
    assert assessment["strict_extraction_mode_used"] == "prefix_only"
    assert assessment["extraction_mode_used"] == "prefix_only"
    assert assessment["used_loose_fallback"] is False
    assert assessment["canonical_smiles"] == "CCO"
    assert assessment["derived_selfies"] == "[C][C][O]"
    assert assessment["best_reference_smiles"] == "CCO"
    assert assessment["max_dice_similarity"] == pytest.approx(1.0)
    assert assessment["passes_similarity_threshold"] is True
    assert assessment["rejection_reason"] is None


def test_assess_biot5_native_generation_output_decodes_selfies_to_smiles() -> None:
    metric_config = CollectionMetricConfig(
        fingerprint_radius=2,
        fingerprint_num_bits=2048,
        acceptance_dice_threshold=0.7,
    )
    records = [
        {
            "id": "desc-native-1",
            "description": "small alcohol molecule",
            "selfies": "[C][C][O]",
            "source_smiles": "CCO",
        }
    ]
    references = prepare_reference_groups(records, metric_config)["small alcohol molecule"]

    assessment = assess_biot5_native_generation_output(
        description_id="desc-native-1",
        description="small alcohol molecule",
        prompt_variant="native_selfies",
        candidate_index=0,
        raw_prediction_text="<bom> [C][C][O] <eom>",
        references=references,
        metric_config=metric_config,
        generation_config_name="greedy_native",
        elapsed_seconds_batch=2.0,
        seconds_per_sample_batch=2.0,
    )

    assert assessment["cleaned_selfies"] == "[C][C][O]"
    assert assessment["parsed_selfies"] == "[C][C][O]"
    assert assessment["selected_selfies"] == "[C][C][O]"
    assert assessment["used_filter_selfies_fallback"] is False
    assert assessment["decoded_smiles"] == "CCO"
    assert assessment["is_valid_selfies"] is True
    assert assessment["canonical_smiles"] == "CCO"
    assert assessment["derived_selfies"] == "[C][C][O]"
    assert assessment["best_reference_smiles"] == "CCO"
    assert assessment["max_dice_similarity"] == pytest.approx(1.0)
    assert assessment["passes_similarity_threshold"] is True
    assert assessment["rejection_reason"] is None


def test_assess_biot5_native_generation_output_marks_invalid_selfies() -> None:
    metric_config = CollectionMetricConfig(
        fingerprint_radius=2,
        fingerprint_num_bits=2048,
        acceptance_dice_threshold=0.7,
    )
    records = [
        {
            "id": "desc-native-2",
            "description": "small alcohol molecule",
            "selfies": "[C][C][O]",
            "source_smiles": "CCO",
        }
    ]
    references = prepare_reference_groups(records, metric_config)["small alcohol molecule"]

    assessment = assess_biot5_native_generation_output(
        description_id="desc-native-2",
        description="small alcohol molecule",
        prompt_variant="native_selfies",
        candidate_index=0,
        raw_prediction_text="plain english output",
        references=references,
        metric_config=metric_config,
    )

    assert assessment["is_valid_selfies"] is False
    assert assessment["decoded_smiles"] is None
    assert assessment["rejection_reason"] == "invalid_selfies"


def test_assess_generation_output_marks_loose_fallback_as_diagnostic() -> None:
    metric_config = CollectionMetricConfig(
        fingerprint_radius=2,
        fingerprint_num_bits=2048,
        acceptance_dice_threshold=0.7,
    )
    records = [
        {
            "id": "desc-2",
            "description": "nitrosourea molecule",
            "selfies": "[C][N][Branch1][Ring1][N][=O][C][=Branch1][C][=N][O]",
            "source_smiles": "CN(N=O)C(=N)O",
        }
    ]
    references = prepare_reference_groups(records, metric_config)["nitrosourea molecule"]

    assessment = assess_generation_output(
        description_id="desc-2",
        description="nitrosourea molecule",
        prompt_variant="baseline_selfies",
        candidate_index=0,
        raw_prediction_text="The generated molecule is CN(N=O)C(=N)O and should be reviewed.",
        references=references,
        metric_config=metric_config,
    )

    assert assessment["strict_extracted_smiles"] is None
    assert assessment["diagnostic_loose_smiles"] == "CN(N=O)C(=N)O"
    assert assessment["extracted_smiles"] == "CN(N=O)C(=N)O"
    assert assessment["extraction_mode_used"] == "loose_fallback"
    assert assessment["used_loose_fallback"] is True
    assert assessment["is_valid_smiles"] is True


def test_summarize_review_records_reports_strict_and_loose_rates() -> None:
    rows = [
        {
            "generation_config_name": "short_contrastive_fast",
            "description_id": "desc-1",
            "description": "small alcohol molecule",
            "prompt_variant": "prefixed_smiles",
            "strict_extracted_smiles": "CCO",
            "strict_extraction_mode_used": "prefix_only",
            "used_loose_fallback": False,
            "extracted_smiles": "CCO",
            "is_valid_smiles": True,
            "canonical_smiles": "CCO",
            "max_dice_similarity": 1.0,
            "passes_similarity_threshold": True,
            "rejection_reason": None,
            "elapsed_seconds_batch": 3.0,
            "seconds_per_sample_batch": 0.1,
        },
        {
            "generation_config_name": "short_contrastive_fast",
            "description_id": "desc-1",
            "description": "small alcohol molecule",
            "prompt_variant": "prefixed_smiles",
            "strict_extracted_smiles": None,
            "strict_extraction_mode_used": None,
            "used_loose_fallback": True,
            "extracted_smiles": "CO",
            "is_valid_smiles": True,
            "canonical_smiles": "CO",
            "max_dice_similarity": 0.8,
            "passes_similarity_threshold": True,
            "rejection_reason": None,
            "elapsed_seconds_batch": 3.0,
            "seconds_per_sample_batch": 0.1,
        },
    ]

    summaries = summarize_review_records(rows)

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["generation_config_name"] == "short_contrastive_fast"
    assert summary["strict_extraction_rate"] == pytest.approx(0.5)
    assert summary["strict_valid_molecule_rate"] == pytest.approx(0.5)
    assert summary["loose_fallback_rate"] == pytest.approx(0.5)
    assert summary["loose_recovery_rate"] == pytest.approx(0.5)
    assert summary["final_valid_molecule_rate"] == pytest.approx(1.0)
    assert summary["prefix_extraction_rate"] == pytest.approx(0.5)
    assert summary["elapsed_seconds_batch"] == pytest.approx(3.0)


def test_summarize_biot5_native_review_records_reports_selfies_metrics() -> None:
    rows = [
        {
            "generation_config_name": "diverse_beam_fast",
            "description_id": "desc-native-1",
            "description": "small alcohol molecule",
            "prompt_variant": "native_selfies",
            "is_valid_selfies": True,
            "used_filter_selfies_fallback": False,
            "is_valid_smiles": True,
            "canonical_smiles": "CCO",
            "max_dice_similarity": 1.0,
            "passes_similarity_threshold": True,
            "rejection_reason": None,
            "elapsed_seconds_batch": 4.0,
            "seconds_per_sample_batch": 0.2,
        },
        {
            "generation_config_name": "diverse_beam_fast",
            "description_id": "desc-native-1",
            "description": "small alcohol molecule",
            "prompt_variant": "native_selfies",
            "is_valid_selfies": True,
            "used_filter_selfies_fallback": True,
            "is_valid_smiles": True,
            "canonical_smiles": "CO",
            "max_dice_similarity": 0.8,
            "passes_similarity_threshold": True,
            "rejection_reason": None,
            "elapsed_seconds_batch": 4.0,
            "seconds_per_sample_batch": 0.2,
        },
        {
            "generation_config_name": "diverse_beam_fast",
            "description_id": "desc-native-1",
            "description": "small alcohol molecule",
            "prompt_variant": "native_selfies",
            "is_valid_selfies": False,
            "used_filter_selfies_fallback": False,
            "is_valid_smiles": False,
            "canonical_smiles": None,
            "max_dice_similarity": 0.0,
            "passes_similarity_threshold": False,
            "rejection_reason": "invalid_selfies",
            "elapsed_seconds_batch": 4.0,
            "seconds_per_sample_batch": 0.2,
        },
    ]

    summaries = summarize_biot5_native_review_records(rows)

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["generation_config_name"] == "diverse_beam_fast"
    assert summary["valid_selfies_rate"] == pytest.approx(2 / 3)
    assert summary["filter_selfies_fallback_rate"] == pytest.approx(1 / 3)
    assert summary["filter_selfies_recovery_rate"] == pytest.approx(1 / 3)
    assert summary["valid_smiles_rate"] == pytest.approx(2 / 3)
    assert summary["invalid_selfies_rate"] == pytest.approx(1 / 3)
    assert summary["elapsed_seconds_batch"] == pytest.approx(4.0)

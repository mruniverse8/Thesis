"""Shared primitives for post-training."""

from .config import (
    detect_dataset_family,
    resolve_gflownet_config_paths,
    resolve_multi_molecule_sft_config_paths,
    resolve_ppo_config_paths,
)
from .decoding import (
    StageTokenConstraints,
    build_stage_token_constraints,
    build_stage_token_constraints_from_examples,
    load_selfies_dictionary_symbols,
    mask_logits_to_allowed_token_ids,
    resolve_stage_token_constraints,
)
from .dataset_types import GroupedMoleculeRecord, coerce_grouped_molecule_record
from .sequence import (
    MOL_SEPARATOR_TOKEN,
    STAGE_SEPARATOR,
    append_stage_to_prefix,
    build_stage_prefix,
    get_sequence_special_tokens,
    parse_molecule_sequence,
    parse_single_staged_molecule,
    parse_staged_target,
    serialize_molecule_sequence,
    serialize_staged_molecule,
    serialize_staged_target,
)

__all__ = [
    "GroupedMoleculeRecord",
    "MOL_SEPARATOR_TOKEN",
    "STAGE_SEPARATOR",
    "StageTokenConstraints",
    "append_stage_to_prefix",
    "build_stage_token_constraints",
    "build_stage_token_constraints_from_examples",
    "build_stage_prefix",
    "coerce_grouped_molecule_record",
    "detect_dataset_family",
    "resolve_gflownet_config_paths",
    "get_sequence_special_tokens",
    "load_selfies_dictionary_symbols",
    "mask_logits_to_allowed_token_ids",
    "parse_molecule_sequence",
    "parse_single_staged_molecule",
    "parse_staged_target",
    "resolve_multi_molecule_sft_config_paths",
    "resolve_ppo_config_paths",
    "resolve_stage_token_constraints",
    "serialize_molecule_sequence",
    "serialize_staged_molecule",
    "serialize_staged_target",
]

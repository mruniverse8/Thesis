from __future__ import annotations

import re
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any

import torch

from molecules.selfies import looks_like_selfies, normalize_selfies_text
from src.constants import BOM_TOKEN, EOM_TOKEN


STAGE_SEPARATOR = " "
MOL_SEPARATOR_TOKEN = STAGE_SEPARATOR
_STAGED_MOLECULE_PATTERN = re.compile(
    rf"{re.escape(BOM_TOKEN)}(.*?){re.escape(EOM_TOKEN)}",
    flags=re.DOTALL,
)


@dataclass(frozen=True)
class StageProjectionResult:
    stage_text: str
    sampled_selfies: str | None
    action_token_ids: tuple[int, ...]
    metadata: dict[str, Any]


def _tokenize_text_ids(tokenizer: Any, text: str) -> tuple[int, ...]:
    if tokenizer is None or not callable(getattr(tokenizer, "__call__", None)):
        raise ValueError("Tokenizer must be callable to encode projected stage text.")

    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
        )
    except TypeError:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
        )

    input_ids = encoded["input_ids"]
    if isinstance(input_ids, torch.Tensor):
        if input_ids.ndim == 0:
            return (int(input_ids.item()),)
        if input_ids.ndim == 1:
            return tuple(int(token_id) for token_id in input_ids.tolist())
        if input_ids.ndim == 2:
            return tuple(int(token_id) for token_id in input_ids[0].tolist())
        raise ValueError("Unsupported input_ids tensor rank while encoding projected stage text.")
    if input_ids and isinstance(input_ids[0], list):
        return tuple(int(token_id) for token_id in input_ids[0])
    return tuple(int(token_id) for token_id in input_ids)


def _invalid_projection_result(
    *,
    raw_stage_text: str,
    raw_sampled_selfies: str | None,
    failure_reason: str,
) -> StageProjectionResult:
    return StageProjectionResult(
        stage_text="",
        sampled_selfies=None,
        action_token_ids=(),
        metadata={
            "raw_stage_text": raw_stage_text,
            "raw_sampled_selfies": raw_sampled_selfies,
            "projection_applied": False,
            "projection_changed": False,
            "projection_failure_reason": failure_reason,
        },
    )


def project_sampled_stage_to_no_h(
    tokenizer: Any,
    raw_stage_text: str,
    *,
    drop_terminal_eom_from_action_ids: bool = False,
) -> StageProjectionResult:
    stripped_stage_text = str(raw_stage_text).strip()
    raw_sampled_selfies = parse_single_staged_molecule(stripped_stage_text)
    if raw_sampled_selfies is None:
        return _invalid_projection_result(
            raw_stage_text=stripped_stage_text,
            raw_sampled_selfies=None,
            failure_reason="invalid_raw_stage_text",
        )

    try:
        import selfies as sf
    except ImportError as exc:  # pragma: no cover - dependency should exist in runtime envs
        return _invalid_projection_result(
            raw_stage_text=stripped_stage_text,
            raw_sampled_selfies=raw_sampled_selfies,
            failure_reason=f"projection_dependency_unavailable:{type(exc).__name__}",
        )

    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover - dependency should exist in runtime envs
        return _invalid_projection_result(
            raw_stage_text=stripped_stage_text,
            raw_sampled_selfies=raw_sampled_selfies,
            failure_reason=f"projection_dependency_unavailable:{type(exc).__name__}",
        )

    try:
        from molecules.parsing import parse_molecule_text
    except ImportError as exc:  # pragma: no cover - dependency should exist in runtime envs
        return _invalid_projection_result(
            raw_stage_text=stripped_stage_text,
            raw_sampled_selfies=raw_sampled_selfies,
            failure_reason=f"projection_dependency_unavailable:{type(exc).__name__}",
        )

    parsed_record = parse_molecule_text(raw_sampled_selfies, representation="selfies")
    if not parsed_record.is_valid or parsed_record.mol is None:
        failure_reason = parsed_record.error or "selfies_parse_failed"
        return _invalid_projection_result(
            raw_stage_text=stripped_stage_text,
            raw_sampled_selfies=raw_sampled_selfies,
            failure_reason=f"projection_parse_failed:{failure_reason}",
        )

    try:
        no_h_mol = Chem.RemoveHs(parsed_record.mol)
    except Exception as exc:
        return _invalid_projection_result(
            raw_stage_text=stripped_stage_text,
            raw_sampled_selfies=raw_sampled_selfies,
            failure_reason=f"projection_remove_hs_failed:{type(exc).__name__}: {exc}",
        )
    if no_h_mol is None:
        return _invalid_projection_result(
            raw_stage_text=stripped_stage_text,
            raw_sampled_selfies=raw_sampled_selfies,
            failure_reason="projection_remove_hs_failed:returned_none",
        )

    try:
        canonical_smiles = str(Chem.MolToSmiles(no_h_mol, canonical=True)).strip()
    except Exception as exc:
        return _invalid_projection_result(
            raw_stage_text=stripped_stage_text,
            raw_sampled_selfies=raw_sampled_selfies,
            failure_reason=f"projection_canonical_smiles_failed:{type(exc).__name__}: {exc}",
        )
    if not canonical_smiles:
        return _invalid_projection_result(
            raw_stage_text=stripped_stage_text,
            raw_sampled_selfies=raw_sampled_selfies,
            failure_reason="projection_canonical_smiles_empty",
        )

    try:
        projected_selfies = normalize_selfies_text(sf.encoder(canonical_smiles))
    except Exception as exc:
        return _invalid_projection_result(
            raw_stage_text=stripped_stage_text,
            raw_sampled_selfies=raw_sampled_selfies,
            failure_reason=f"projection_selfies_encode_failed:{type(exc).__name__}: {exc}",
        )
    if not projected_selfies:
        return _invalid_projection_result(
            raw_stage_text=stripped_stage_text,
            raw_sampled_selfies=raw_sampled_selfies,
            failure_reason="projection_selfies_encode_empty",
        )

    try:
        projected_stage_text = serialize_staged_molecule(projected_selfies)
        projected_action_token_ids = _tokenize_text_ids(tokenizer, projected_stage_text)
    except Exception as exc:
        return _invalid_projection_result(
            raw_stage_text=stripped_stage_text,
            raw_sampled_selfies=raw_sampled_selfies,
            failure_reason=f"projection_retokenize_failed:{type(exc).__name__}: {exc}",
        )

    if drop_terminal_eom_from_action_ids:
        if tokenizer is None or not hasattr(tokenizer, "convert_tokens_to_ids"):
            return _invalid_projection_result(
                raw_stage_text=stripped_stage_text,
                raw_sampled_selfies=raw_sampled_selfies,
                failure_reason="projection_missing_eom_tokenizer_support",
            )
        eom_token_id = int(tokenizer.convert_tokens_to_ids(EOM_TOKEN))
        if not projected_action_token_ids or int(projected_action_token_ids[-1]) != eom_token_id:
            return _invalid_projection_result(
                raw_stage_text=stripped_stage_text,
                raw_sampled_selfies=raw_sampled_selfies,
                failure_reason="projection_missing_terminal_eom_token",
            )
        projected_action_token_ids = projected_action_token_ids[:-1]

    return StageProjectionResult(
        stage_text=projected_stage_text,
        sampled_selfies=projected_selfies,
        action_token_ids=projected_action_token_ids,
        metadata={
            "raw_stage_text": stripped_stage_text,
            "raw_sampled_selfies": raw_sampled_selfies,
            "projection_applied": True,
            "projection_changed": projected_stage_text != stripped_stage_text,
            "projection_failure_reason": None,
            "projected_canonical_smiles": canonical_smiles,
        },
    )


def get_sequence_special_tokens(separator_token: str = STAGE_SEPARATOR) -> list[str]:
    del separator_token
    return [BOM_TOKEN, EOM_TOKEN]


def serialize_staged_molecule(selfies_text: str) -> str:
    normalized = normalize_selfies_text(selfies_text)
    if not normalized:
        raise ValueError("Expected a non-empty molecule when serializing a stage.")
    return f"{BOM_TOKEN}{normalized}{EOM_TOKEN}"


def serialize_staged_target(
    selfies_list: Sequence[str],
    separator_token: str = STAGE_SEPARATOR,
) -> str:
    normalized = [normalize_selfies_text(item) for item in selfies_list]
    if not normalized:
        raise ValueError("Expected at least one molecule when serializing a staged target.")
    if any(not item for item in normalized):
        raise ValueError("Encountered an empty molecule while serializing a staged target.")
    return separator_token.join(serialize_staged_molecule(item) for item in normalized)


def strip_sequence_wrappers(text: str) -> str:
    return str(text).replace(BOM_TOKEN, "").replace(EOM_TOKEN, "").strip()


def _is_valid_staged_payload(text: str) -> bool:
    normalized = normalize_selfies_text(text)
    return bool(normalized) and looks_like_selfies(normalized)


def _is_separator_segment(text: str, separator_token: str) -> bool:
    chunk = str(text)
    if not chunk:
        return True
    if not separator_token.strip():
        return not chunk.strip()
    remainder = chunk.replace(separator_token, "")
    return not remainder.strip()


def parse_staged_target(
    text: str,
    separator_token: str = STAGE_SEPARATOR,
) -> list[str]:
    raw_text = str(text)
    molecules: list[str] = []
    cursor = 0
    for match in _STAGED_MOLECULE_PATTERN.finditer(raw_text):
        if not _is_separator_segment(raw_text[cursor : match.start()], separator_token):
            return []
        molecule = normalize_selfies_text(match.group(1))
        if not _is_valid_staged_payload(molecule):
            return []
        molecules.append(molecule)
        cursor = match.end()
    if not _is_separator_segment(raw_text[cursor:], separator_token):
        return []
    return molecules


def parse_single_staged_molecule(text: str) -> str | None:
    match = re.fullmatch(
        rf"\s*{re.escape(BOM_TOKEN)}(.*?){re.escape(EOM_TOKEN)}\s*",
        str(text),
        flags=re.DOTALL,
    )
    if match is None:
        return None
    molecule = normalize_selfies_text(match.group(1))
    if not _is_valid_staged_payload(molecule):
        return None
    return molecule


def serialize_molecule_sequence(
    selfies_list: Sequence[str],
    separator_token: str = STAGE_SEPARATOR,
) -> str:
    return serialize_staged_target(selfies_list, separator_token=separator_token)


def parse_molecule_sequence(
    text: str,
    separator_token: str = STAGE_SEPARATOR,
) -> list[str]:
    return parse_staged_target(text, separator_token=separator_token)


def build_stage_prefix(
    previous_selfies_list: Sequence[str],
    separator_token: str = STAGE_SEPARATOR,
) -> str:
    if not previous_selfies_list:
        return ""
    return f"{serialize_staged_target(previous_selfies_list, separator_token=separator_token)}{separator_token}"


def append_stage_to_prefix(
    prefix_text: str,
    molecule_selfies: str,
    stop_token: str,
    separator_token: str = STAGE_SEPARATOR,
) -> str:
    stage_text = serialize_staged_molecule(molecule_selfies)
    normalized_prefix = str(prefix_text)
    if stop_token == separator_token:
        return f"{normalized_prefix}{stage_text}{separator_token}"
    if stop_token == EOM_TOKEN:
        return f"{normalized_prefix}{stage_text}"
    raise ValueError(f"Unsupported stop token: {stop_token!r}")

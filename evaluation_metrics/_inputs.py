from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


ROW_ID_KEYS = ("id", "example_id", "description_id", "group_id", "rollout_id")


@dataclass(frozen=True)
class MoleculeInputParts:
    text: str
    representation: str
    molecule_id: str | None = None


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def row_id(row: Mapping[str, Any], *, fallback: str) -> str:
    for key in ROW_ID_KEYS:
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return fallback


def _first_present_text(
    row: Mapping[str, Any],
    *,
    keys: Sequence[str],
) -> str:
    for key in keys:
        value = row.get(key)
        if value:
            return str(value).strip()
    return ""


def _infer_representation(
    row: Mapping[str, Any],
    *,
    default_representation: str,
    representation_key_groups: Sequence[tuple[str, Sequence[str]]],
) -> str:
    explicit = str(row.get("representation") or "").strip()
    if explicit:
        return explicit
    for representation, keys in representation_key_groups:
        if any(row.get(key) for key in keys):
            return representation
    return default_representation


def molecule_input_parts_from_value(
    value: Any,
    *,
    default_representation: str,
    fallback_id: str | None = None,
    mapping_text_keys: Sequence[str],
    representation_key_groups: Sequence[tuple[str, Sequence[str]]],
    strip_scalar_text: bool,
    drop_empty_text: bool,
) -> MoleculeInputParts | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip() if strip_scalar_text else value
        if drop_empty_text and not text:
            return None
        return MoleculeInputParts(
            text=text,
            representation=default_representation,
            molecule_id=fallback_id,
        )
    if not isinstance(value, Mapping):
        text = str(value).strip() if strip_scalar_text else str(value)
        if drop_empty_text and not text:
            return None
        return MoleculeInputParts(
            text=text,
            representation=default_representation,
            molecule_id=fallback_id,
        )

    text = _first_present_text(value, keys=mapping_text_keys)
    if drop_empty_text and not text:
        return None
    molecule_id = str(value.get("id") or value.get("molecule_id") or fallback_id or "").strip()
    return MoleculeInputParts(
        text=text,
        representation=_infer_representation(
            value,
            default_representation=default_representation,
            representation_key_groups=representation_key_groups,
        ),
        molecule_id=molecule_id or None,
    )


def collect_molecule_input_parts(
    row: Mapping[str, Any],
    *,
    keys: Iterable[str],
    default_representation: str,
    mapping_text_keys: Sequence[str],
    representation_key_groups: Sequence[tuple[str, Sequence[str]]],
) -> tuple[MoleculeInputParts, ...]:
    molecules: list[MoleculeInputParts] = []
    for key in keys:
        for index, value in enumerate(as_list(row.get(key))):
            molecule = molecule_input_parts_from_value(
                value,
                default_representation=default_representation,
                fallback_id=f"{key}-{index:06d}",
                mapping_text_keys=mapping_text_keys,
                representation_key_groups=representation_key_groups,
                strip_scalar_text=True,
                drop_empty_text=True,
            )
            if molecule is not None:
                molecules.append(molecule)
    return tuple(molecules)

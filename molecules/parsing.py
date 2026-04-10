from __future__ import annotations

from molecules.representations import MoleculeRecord, MoleculeRepresentation
from molecules.selfies import looks_like_selfies

try:
    from rdkit import Chem
except ImportError as exc:  # pragma: no cover - exercised only when RDKit is missing
    Chem = None
    _RDKIT_IMPORT_ERROR = exc
else:
    _RDKIT_IMPORT_ERROR = None

import selfies as sf


def ensure_rdkit() -> None:
    if Chem is None:  # pragma: no cover - guarded in normal test env
        raise ImportError(
            "molecules requires RDKit. Install it with "
            "`conda install -n thesis_biot5_sft -c conda-forge rdkit`."
        ) from _RDKIT_IMPORT_ERROR


def normalize_molecule_text(text: str) -> str:
    return text.strip()


def _invalid_record(
    text: str,
    representation: str,
    *,
    normalized_input: str,
    used_selfies_decoder: bool,
    error: str,
) -> MoleculeRecord:
    return MoleculeRecord(
        input_text=text,
        input_representation=representation,
        normalized_input=normalized_input,
        smiles=None,
        canonical_smiles=None,
        is_valid=False,
        used_selfies_decoder=used_selfies_decoder,
        error=error,
        mol=None,
    )


def _record_from_mol(
    text: str,
    representation: str,
    *,
    normalized_input: str,
    smiles: str,
    used_selfies_decoder: bool,
    mol: object,
) -> MoleculeRecord:
    ensure_rdkit()
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    return MoleculeRecord(
        input_text=text,
        input_representation=representation,
        normalized_input=normalized_input,
        smiles=smiles,
        canonical_smiles=canonical_smiles,
        is_valid=True,
        used_selfies_decoder=used_selfies_decoder,
        error=None,
        mol=mol,
    )


def _parse_smiles(text: str, representation: str) -> MoleculeRecord:
    ensure_rdkit()
    normalized = normalize_molecule_text(text)
    mol = Chem.MolFromSmiles(normalized)
    if mol is None:
        return _invalid_record(
            text,
            representation,
            normalized_input=normalized,
            used_selfies_decoder=False,
            error="RDKit could not parse the SMILES string.",
        )
    return _record_from_mol(
        text,
        representation,
        normalized_input=normalized,
        smiles=normalized,
        used_selfies_decoder=False,
        mol=mol,
    )


def _parse_selfies(text: str, representation: str) -> MoleculeRecord:
    ensure_rdkit()
    normalized = normalize_molecule_text(text).replace(" ", "")
    if not looks_like_selfies(normalized):
        return _invalid_record(
            text,
            representation,
            normalized_input=normalized,
            used_selfies_decoder=True,
            error="Input does not match the expected bracketed SELFIES token format.",
        )
    try:
        decoded_smiles = sf.decoder(normalized)
    except Exception as exc:
        return _invalid_record(
            text,
            representation,
            normalized_input=normalized,
            used_selfies_decoder=True,
            error=f"SELFIES decoding failed: {exc}",
        )
    mol = Chem.MolFromSmiles(decoded_smiles)
    if mol is None:
        return _invalid_record(
            text,
            representation,
            normalized_input=normalized,
            used_selfies_decoder=True,
            error="RDKit could not parse the SMILES decoded from SELFIES.",
        )
    return _record_from_mol(
        text,
        representation,
        normalized_input=normalized,
        smiles=decoded_smiles,
        used_selfies_decoder=True,
        mol=mol,
    )


def parse_molecule_text(text: str, representation: MoleculeRepresentation = "auto") -> MoleculeRecord:
    normalized = normalize_molecule_text(text)
    if not normalized:
        return _invalid_record(
            text,
            representation,
            normalized_input=normalized,
            used_selfies_decoder=False,
            error="Empty molecule string.",
        )

    if representation == "smiles":
        return _parse_smiles(text, representation)
    if representation == "selfies":
        return _parse_selfies(text, representation)

    parsers = [_parse_smiles]
    if looks_like_selfies(normalized):
        parsers = [_parse_selfies, _parse_smiles]

    errors: list[str] = []
    for parser in parsers:
        record = parser(text, representation)
        if record.is_valid:
            return record
        if record.error:
            errors.append(record.error)

    return _invalid_record(
        text,
        representation,
        normalized_input=normalized,
        used_selfies_decoder=looks_like_selfies(normalized),
        error=" | ".join(errors) if errors else "Failed to parse molecule.",
    )


def record_from_rdkit_mol(mol: object) -> MoleculeRecord:
    ensure_rdkit()
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    return MoleculeRecord(
        input_text=canonical_smiles,
        input_representation="rdkit",
        normalized_input=canonical_smiles,
        smiles=canonical_smiles,
        canonical_smiles=canonical_smiles,
        is_valid=True,
        used_selfies_decoder=False,
        error=None,
        mol=mol,
    )


def is_duplicate_candidate(candidate: MoleculeRecord, previous_candidates: list[MoleculeRecord]) -> bool:
    if not candidate.is_valid or candidate.canonical_smiles is None:
        return False
    return any(
        previous.is_valid and previous.canonical_smiles == candidate.canonical_smiles
        for previous in previous_candidates
    )


__all__ = [
    "MoleculeRecord",
    "MoleculeRepresentation",
    "ensure_rdkit",
    "is_duplicate_candidate",
    "looks_like_selfies",
    "normalize_molecule_text",
    "parse_molecule_text",
    "record_from_rdkit_mol",
]

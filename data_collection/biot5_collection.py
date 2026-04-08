from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Sequence

from tqdm import tqdm

from evaluation.config import MoleculeMetricConfig
from evaluation.grouping import build_reference_index
from reward_utils.fingerprints import build_morgan_fingerprint
from reward_utils.validation import parse_molecule_text
from src.io_utils import dump_yaml, ensure_dir, read_jsonl, set_seed, write_json, write_jsonl
from src.prompting import build_text2mol_prompt, normalize_free_text
from src.selfies_utils import normalize_generated_selfies, parse_generated_selfies

if TYPE_CHECKING:
    import torch

try:
    from rdkit import DataStructs
except ImportError:  # pragma: no cover - guarded by dependency checks in tests/runtime
    DataStructs = None


class CandidateGenerator(Protocol):
    def generate_candidates(self, prompt_text: str, target_count: int) -> list[str]:
        ...


@dataclass(frozen=True)
class PreparedReference:
    canonical_smiles: str
    fingerprint: object


def build_contrastive_generation_kwargs(
    generation_config: dict[str, Any],
    *,
    bad_words_ids: Sequence[Sequence[int]] | None = None,
) -> dict[str, Any]:
    penalty_alpha = float(generation_config["penalty_alpha"])
    top_k = int(generation_config["top_k"])
    if penalty_alpha <= 0:
        raise ValueError("contrastive search requires `penalty_alpha > 0`")
    if top_k <= 1:
        raise ValueError("contrastive search requires `top_k > 1`")

    kwargs: dict[str, Any] = {
        "max_new_tokens": int(generation_config["max_new_tokens"]),
        "do_sample": False,
        "num_beams": 1,
        "penalty_alpha": penalty_alpha,
        "top_k": top_k,
    }
    if bad_words_ids:
        kwargs["bad_words_ids"] = [list(item) for item in bad_words_ids if item]
    return kwargs


class StaticCandidateGenerator:
    def __init__(self, outputs_by_description: dict[str, list[str]]) -> None:
        self.outputs_by_description = outputs_by_description

    def generate_candidates(self, prompt_text: str, target_count: int) -> list[str]:
        for description, outputs in self.outputs_by_description.items():
            if description in prompt_text:
                return list(outputs[:target_count])
        raise AssertionError(f"Unexpected prompt text: {prompt_text}")


class BioT5ContrastiveGenerator:
    def __init__(
        self,
        *,
        model_name_or_path: str,
        tokenizer_name: str,
        base_tokenizer_name: str,
        selfies_vocab_path: str | Path,
        device_name: str,
        max_source_length: int,
        generation_config: dict[str, Any],
    ) -> None:
        import torch
        from transformers import T5ForConditionalGeneration

        from src.tokenizer_utils import build_decoder_tokenizer, prepare_training_tokenizer
        from src.training import choose_device

        self.training_tokenizer, _ = prepare_training_tokenizer(
            tokenizer_name=tokenizer_name,
            selfies_vocab_path=selfies_vocab_path,
        )
        self.decoder_tokenizer = build_decoder_tokenizer(
            base_tokenizer_name=base_tokenizer_name,
            training_tokenizer=self.training_tokenizer,
        )
        self.device = choose_device(device_name)
        self.max_source_length = int(max_source_length)
        self.generation_config = dict(generation_config)
        self.torch = torch

        self.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        if self.model.get_input_embeddings().weight.size(0) != len(self.training_tokenizer):
            self.model.resize_token_embeddings(len(self.training_tokenizer))
        self.model.to(self.device)
        self.model.eval()

    def _blocked_sequence_ids(self, generated_ids: Sequence[int]) -> list[int]:
        decoder_start_token_id = getattr(self.model.config, "decoder_start_token_id", None)
        eos_token_id = self.training_tokenizer.eos_token_id
        pad_token_id = self.training_tokenizer.pad_token_id

        cleaned: list[int] = []
        for token_id in generated_ids:
            if decoder_start_token_id is not None and token_id == decoder_start_token_id and not cleaned:
                continue
            if pad_token_id is not None and token_id == pad_token_id and not cleaned:
                continue
            cleaned.append(int(token_id))
            if eos_token_id is not None and token_id == eos_token_id:
                break

        while cleaned and pad_token_id is not None and cleaned[-1] == pad_token_id:
            cleaned.pop()
        return cleaned

    def generate_candidates(self, prompt_text: str, target_count: int) -> list[str]:
        encoded = self.training_tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_source_length,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        raw_outputs: list[str] = []
        blocked_sequences: list[list[int]] = []
        seen_blocked_sequences: set[tuple[int, ...]] = set()

        for _ in range(int(target_count)):
            generation_kwargs = build_contrastive_generation_kwargs(
                self.generation_config,
                bad_words_ids=blocked_sequences,
            )
            with self.torch.no_grad():
                generated_ids = self.model.generate(**encoded, **generation_kwargs)

            raw_text = self.decoder_tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )[0]
            raw_outputs.append(raw_text)

            blocked = self._blocked_sequence_ids(generated_ids[0].tolist())
            blocked_key = tuple(blocked)
            if blocked and blocked_key not in seen_blocked_sequences:
                blocked_sequences.append(blocked)
                seen_blocked_sequences.add(blocked_key)

        return raw_outputs


def _select_collection_records(
    records: list[dict[str, Any]],
    *,
    description_offset: int,
    max_descriptions: int | None,
) -> list[dict[str, Any]]:
    unique_records: list[dict[str, Any]] = []
    seen_descriptions: set[str] = set()
    for record in records:
        description = normalize_free_text(str(record.get("description", "")))
        if not description or description in seen_descriptions:
            continue
        seen_descriptions.add(description)
        unique_records.append(record)

    selected = unique_records[description_offset:]
    if max_descriptions is not None:
        selected = selected[:max_descriptions]
    return selected


def _prepare_reference_groups(
    records: list[dict[str, Any]],
    metric_config: MoleculeMetricConfig,
) -> dict[str, tuple[PreparedReference, ...]]:
    if DataStructs is None:
        raise ImportError("RDKit is required for BioT5 data collection.")

    reference_index = build_reference_index(records)
    prepared_by_description: dict[str, tuple[PreparedReference, ...]] = {}

    for description, references in reference_index.references_by_description.items():
        prepared: list[PreparedReference] = []
        for reference in references:
            record = parse_molecule_text(reference.molecule_text, representation=reference.representation)
            fingerprint = build_morgan_fingerprint(
                record,
                radius=metric_config.fingerprint_radius,
                n_bits=metric_config.fingerprint_num_bits,
            )
            if not record.is_valid or fingerprint is None or record.canonical_smiles is None:
                continue
            prepared.append(
                PreparedReference(
                    canonical_smiles=record.canonical_smiles,
                    fingerprint=fingerprint,
                )
            )
        if not prepared:
            raise ValueError(f"No valid reference molecules available for description: {description!r}")
        prepared_by_description[description] = tuple(prepared)

    return prepared_by_description


def _assess_candidate(
    *,
    candidate_id: str,
    description_id: str,
    description: str,
    raw_prediction_text: str,
    references: Sequence[PreparedReference],
    metric_config: MoleculeMetricConfig,
) -> dict[str, Any]:
    normalized_prediction = normalize_generated_selfies(raw_prediction_text)
    assessment = {
        "id": candidate_id,
        "description_id": description_id,
        "description": description,
        "raw_prediction_text": raw_prediction_text,
        "normalized_prediction_selfies": normalized_prediction,
        "parsed_selfies": None,
        "decoded_smiles": None,
        "canonical_smiles": None,
        "used_repair": False,
        "accepted": False,
        "max_dice_similarity": 0.0,
        "best_reference_smiles": None,
        "rejection_reason": None,
    }

    if not normalized_prediction:
        assessment["rejection_reason"] = "empty_output"
        return assessment

    selfies_text, decoded_smiles, used_repair = parse_generated_selfies(raw_prediction_text)
    assessment["parsed_selfies"] = selfies_text
    assessment["decoded_smiles"] = decoded_smiles
    assessment["used_repair"] = used_repair

    if not selfies_text or not decoded_smiles:
        assessment["rejection_reason"] = "invalid_selfies"
        return assessment

    candidate_record = parse_molecule_text(selfies_text, representation="selfies")
    if not candidate_record.is_valid or candidate_record.canonical_smiles is None:
        assessment["rejection_reason"] = "invalid_molecule"
        return assessment

    fingerprint = build_morgan_fingerprint(
        candidate_record,
        radius=metric_config.fingerprint_radius,
        n_bits=metric_config.fingerprint_num_bits,
    )
    if fingerprint is None:
        assessment["rejection_reason"] = "invalid_molecule"
        return assessment

    best_similarity = 0.0
    best_reference_smiles: str | None = None
    for reference in references:
        similarity = float(DataStructs.DiceSimilarity(fingerprint, reference.fingerprint))
        if similarity > best_similarity:
            best_similarity = similarity
            best_reference_smiles = reference.canonical_smiles

    assessment["canonical_smiles"] = candidate_record.canonical_smiles
    assessment["max_dice_similarity"] = best_similarity
    assessment["best_reference_smiles"] = best_reference_smiles

    if best_similarity <= metric_config.acceptance_dice_threshold:
        assessment["rejection_reason"] = "unaccepted_molecule"
        return assessment

    assessment["accepted"] = True
    return assessment


def collect_biot5_training_data(
    config: dict[str, Any],
    generator: CandidateGenerator | None = None,
) -> dict[str, Any]:
    seed = int(config.get("seed", 42))
    set_seed(seed)

    train_path = Path(config["data"]["train_file"])
    staging_dir = ensure_dir(config["data"]["staging_dir"])
    derived_train_file = Path(config["data"]["derived_train_file"])
    description_offset = int(config.get("runtime", {}).get("description_offset", 0))
    max_descriptions = config.get("runtime", {}).get("max_descriptions")
    if max_descriptions is not None:
        max_descriptions = int(max_descriptions)

    metric_config = MoleculeMetricConfig(
        fingerprint_radius=int(config["filtering"]["fingerprint_radius"]),
        fingerprint_num_bits=int(config["filtering"]["fingerprint_num_bits"]),
        acceptance_dice_threshold=float(config["filtering"]["acceptance_dice_threshold"]),
    )
    max_molecules_per_example = int(config["filtering"]["max_molecules_per_example"])
    target_molecules = int(config["generation"]["target_molecules_per_description"])

    all_train_records = read_jsonl(train_path)
    selected_records = _select_collection_records(
        all_train_records,
        description_offset=description_offset,
        max_descriptions=max_descriptions,
    )
    reference_groups = _prepare_reference_groups(all_train_records, metric_config)

    if generator is None:
        generator = BioT5ContrastiveGenerator(
            model_name_or_path=config["model"]["model_name_or_path"],
            tokenizer_name=config["model"]["tokenizer_name"],
            base_tokenizer_name=config["model"]["base_tokenizer_name"],
            selfies_vocab_path=config["model"]["selfies_vocab_path"],
            device_name=config["model"].get("device", "auto"),
            max_source_length=int(config["generation"]["max_source_length"]),
            generation_config=config["generation"],
        )

    dump_yaml(staging_dir / "resolved_config.yaml", config)

    input_records: list[dict[str, Any]] = []
    raw_candidate_records: list[dict[str, Any]] = []
    candidate_assessments: list[dict[str, Any]] = []
    accepted_grouped_records: list[dict[str, Any]] = []
    derived_train_records: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()

    for record in tqdm(selected_records, desc="Collecting BioT5 candidates", leave=False):
        description_id = str(record.get("id") or "").strip()
        description = normalize_free_text(str(record.get("description", "")))
        if not description:
            continue
        if not description_id:
            description_id = f"train-{len(input_records):06d}"

        references = reference_groups[description]
        prompt_text = build_text2mol_prompt(description)
        input_records.append(
            {
                "id": description_id,
                "description": description,
                "prompt_text": prompt_text,
                "reference_smiles_list": [reference.canonical_smiles for reference in references],
            }
        )

        raw_predictions = generator.generate_candidates(prompt_text, target_molecules)
        accepted_candidates: list[dict[str, Any]] = []
        seen_canonical_smiles: set[str] = set()

        for candidate_index, raw_prediction in enumerate(raw_predictions):
            candidate_id = f"{description_id}-candidate-{candidate_index:03d}"
            raw_candidate_records.append(
                {
                    "id": candidate_id,
                    "description_id": description_id,
                    "description": description,
                    "candidate_index": candidate_index,
                    "raw_prediction_text": raw_prediction,
                    "normalized_prediction_selfies": normalize_generated_selfies(raw_prediction),
                }
            )

            assessment = _assess_candidate(
                candidate_id=candidate_id,
                description_id=description_id,
                description=description,
                raw_prediction_text=raw_prediction,
                references=references,
                metric_config=metric_config,
            )

            canonical_smiles = assessment["canonical_smiles"]
            if assessment["accepted"] and canonical_smiles in seen_canonical_smiles:
                assessment["accepted"] = False
                assessment["rejection_reason"] = "duplicate_smiles"

            if assessment["accepted"] and canonical_smiles is not None:
                seen_canonical_smiles.add(canonical_smiles)
                accepted_candidates.append(assessment)
            elif assessment["rejection_reason"]:
                rejection_counts[assessment["rejection_reason"]] += 1

            candidate_assessments.append(assessment)

        accepted_candidates = sorted(
            accepted_candidates,
            key=lambda item: (str(item["canonical_smiles"]), str(item["parsed_selfies"])),
        )
        if not accepted_candidates:
            continue

        accepted_grouped_records.append(
            {
                "id": description_id,
                "description": description,
                "reference_smiles_list": [reference.canonical_smiles for reference in references],
                "accepted_target_selfies_list": [
                    item["parsed_selfies"] for item in accepted_candidates if item["parsed_selfies"]
                ],
                "accepted_target_smiles_list": [
                    item["canonical_smiles"] for item in accepted_candidates if item["canonical_smiles"]
                ],
                "accepted_candidate_count": len(accepted_candidates),
                "raw_candidate_count": len(raw_predictions),
            }
        )

        retained_candidates = accepted_candidates[:max_molecules_per_example]
        derived_train_records.append(
            {
                "id": description_id,
                "description": description,
                "target_selfies_list": [
                    item["parsed_selfies"] for item in retained_candidates if item["parsed_selfies"]
                ],
                "target_smiles_list": [
                    item["canonical_smiles"] for item in retained_candidates if item["canonical_smiles"]
                ],
                "accepted_candidate_count": len(accepted_candidates),
            }
        )

    write_jsonl(staging_dir / "inputs.jsonl", input_records)
    write_jsonl(staging_dir / "raw_candidates.jsonl", raw_candidate_records)
    write_jsonl(staging_dir / "candidate_assessments.jsonl", candidate_assessments)
    write_jsonl(staging_dir / "accepted_grouped.jsonl", accepted_grouped_records)
    write_jsonl(derived_train_file, derived_train_records)

    summary = {
        "train_file": str(train_path),
        "staging_dir": str(staging_dir),
        "derived_train_file": str(derived_train_file),
        "seed": seed,
        "selected_descriptions": len(selected_records),
        "descriptions_with_accepted_molecules": len(accepted_grouped_records),
        "raw_candidates": len(raw_candidate_records),
        "accepted_candidates": sum(record["accepted_candidate_count"] for record in accepted_grouped_records),
        "derived_examples": len(derived_train_records),
        "target_molecules_per_description": target_molecules,
        "max_molecules_per_example": max_molecules_per_example,
        "rejections_by_reason": dict(sorted(rejection_counts.items())),
        "files": {
            "inputs": str(staging_dir / "inputs.jsonl"),
            "raw_candidates": str(staging_dir / "raw_candidates.jsonl"),
            "candidate_assessments": str(staging_dir / "candidate_assessments.jsonl"),
            "accepted_grouped": str(staging_dir / "accepted_grouped.jsonl"),
            "derived_train_file": str(derived_train_file),
            "resolved_config": str(staging_dir / "resolved_config.yaml"),
            "summary": str(staging_dir / "summary.json"),
        },
    }
    write_json(staging_dir / "summary.json", summary)
    return summary

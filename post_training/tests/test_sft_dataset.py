import pytest
import torch

pytest.importorskip("selfies")

from post_training.sft_dataset import (
    MultiMoleculeCollator,
    MultiMoleculeDataset,
    build_multi_molecule_processed_record,
)


class DummyTokenizer:
    pad_token_id = 0

    def _encode(self, text: str, max_length: int | None) -> list[int]:
        encoded = [index + 1 for index, _ in enumerate(text)]
        if max_length is not None:
            encoded = encoded[:max_length]
        return encoded or [1]

    def __call__(
        self,
        texts=None,
        *,
        text_target=None,
        padding=True,
        truncation=True,
        max_length=None,
        return_tensors=None,
    ):
        del padding, truncation, return_tensors
        items = text_target if text_target is not None else texts
        if isinstance(items, str):
            items = [items]

        encoded = [self._encode(item, max_length) for item in items]
        padded_length = max(len(item) for item in encoded)
        padded = [item + [self.pad_token_id] * (padded_length - len(item)) for item in encoded]
        attention_mask = [
            [1] * len(item) + [0] * (padded_length - len(item)) for item in encoded
        ]
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def test_build_multi_molecule_processed_record_deduplicates_and_converts_smiles() -> None:
    raw_record = {
        "id": "example-1",
        "description": "  Example description  ",
        "targets": [
            {"selfies": "[C][O]"},
            {"selfies": "[C][O]"},
            {"smiles": "CC"},
            {"selfies": "[bad_token]"},
        ],
    }

    processed = build_multi_molecule_processed_record(raw_record, split="train", index=0)

    assert processed["id"] == "example-1"
    assert processed["description"] == "Example description"
    assert processed["target_selfies_list"] == ["[C][O]", "[C][C]"]


def test_multi_molecule_dataset_and_collator_emit_sequence_targets() -> None:
    dataset = MultiMoleculeDataset(
        [
            {
                "id": "example-2",
                "description": "A simple set",
                "target_selfies_list": ["[C][O]", "[C][C]"],
            }
        ]
    )

    item = dataset[0]
    assert item["target_text"] == "<bom>[C][O]<mol_sep>[C][C]<eom>"

    collator = MultiMoleculeCollator(
        tokenizer=DummyTokenizer(),
        max_source_length=128,
        max_target_length=128,
    )
    batch = collator([item])

    assert batch["labels"].shape[0] == 1
    assert batch["target_selfies_lists"] == [["[C][O]", "[C][C]"]]
    assert batch["target_texts"] == ["<bom>[C][O]<mol_sep>[C][C]<eom>"]

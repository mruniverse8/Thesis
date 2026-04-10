from __future__ import annotations

from typing import Any


class MultiMoleculeCollator:
    def __init__(
        self,
        tokenizer: Any,
        max_source_length: int,
        max_target_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        prompts = [example["prompt"] for example in batch]
        targets = [example["target_text"] for example in batch]

        model_inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_source_length,
            return_tensors="pt",
        )
        target_tokens = self.tokenizer(
            text_target=targets,
            padding=True,
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        )

        labels = target_tokens["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels

        model_inputs["example_ids"] = [example["id"] for example in batch]
        model_inputs["prompts"] = prompts
        model_inputs["target_texts"] = targets
        model_inputs["descriptions"] = [example["description"] for example in batch]
        model_inputs["target_selfies_lists"] = [
            list(example["target_selfies_list"]) for example in batch
        ]
        return model_inputs

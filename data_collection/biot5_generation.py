from __future__ import annotations

import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def build_diverse_beam_generation_kwargs(
    generation_config: dict[str, Any],
    *,
    target_count: int,
) -> dict[str, Any]:
    target_count = int(target_count)
    num_beams = int(generation_config["num_beams"])
    num_return_sequences = int(generation_config.get("num_return_sequences", target_count))

    kwargs: dict[str, Any] = {
        "do_sample": False,
        "use_cache": True,
        "num_beams": num_beams,
        "num_return_sequences": num_return_sequences,
    }

    if "max_new_tokens" in generation_config:
        kwargs["max_new_tokens"] = int(generation_config["max_new_tokens"])
    elif "max_length" in generation_config:
        kwargs["max_length"] = int(generation_config["max_length"])
    else:
        raise KeyError("diverse beam search requires `max_new_tokens` or `max_length`")

    if num_return_sequences != target_count:
        raise ValueError(
            f"generation config expected {num_return_sequences} outputs but target_count={target_count}"
        )
    if num_beams < num_return_sequences:
        raise ValueError(
            f"num_beams must be >= num_return_sequences, got {num_beams} < {num_return_sequences}"
        )

    if "num_beam_groups" in generation_config:
        num_beam_groups = int(generation_config["num_beam_groups"])
        diversity_penalty = float(generation_config["diversity_penalty"])
        if num_beam_groups <= 1:
            raise ValueError("diverse beam search requires num_beam_groups > 1")
        if num_beams % num_beam_groups != 0:
            raise ValueError(
                "num_beams must be divisible by num_beam_groups, "
                f"got {num_beams} and {num_beam_groups}"
            )
        if diversity_penalty <= 0:
            raise ValueError("diverse beam search requires diversity_penalty > 0")
        kwargs["num_beam_groups"] = num_beam_groups
        kwargs["diversity_penalty"] = diversity_penalty

    if "early_stopping" in generation_config:
        kwargs["early_stopping"] = bool(generation_config["early_stopping"])
    if "length_penalty" in generation_config:
        kwargs["length_penalty"] = float(generation_config["length_penalty"])

    return kwargs


class BioT5DiverseBeamGenerator:
    def __init__(
        self,
        *,
        model_name_or_path: str,
        device_name: str,
        max_source_length: int | None = None,
        model_max_length: int | None = None,
        generation_config: dict[str, Any],
        tokenizer_name: str | None = None,
        base_tokenizer_name: str | None = None,
        selfies_vocab_path: str | Path | None = None,
    ) -> None:
        import torch
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        from src.training import choose_device

        del tokenizer_name
        del base_tokenizer_name
        del selfies_vocab_path

        resolved_model_max_length = self._resolve_model_max_length(
            generation_config=generation_config,
            model_max_length=model_max_length,
            max_source_length=max_source_length,
        )
        self.tokenizer = T5Tokenizer.from_pretrained(
            model_name_or_path,
            model_max_length=resolved_model_max_length,
        )
        self.device = choose_device(device_name)
        self.model_max_length = resolved_model_max_length
        self.max_source_length = resolved_model_max_length
        self.generation_config = dict(generation_config)
        self.torch = torch

        self.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        self.model.eval()
        self.supports_remote_group_beam_search = self._supports_remote_group_beam_search()
        self.supports_custom_generate = self.supports_remote_group_beam_search

    @staticmethod
    def _resolve_model_max_length(
        *,
        generation_config: dict[str, Any],
        model_max_length: int | None,
        max_source_length: int | None,
    ) -> int:
        if model_max_length is not None:
            return int(model_max_length)
        if max_source_length is not None:
            return int(max_source_length)
        if "max_source_length" in generation_config:
            return int(generation_config["max_source_length"])
        if "max_length" in generation_config:
            return int(generation_config["max_length"])
        if "max_new_tokens" in generation_config:
            return int(generation_config["max_new_tokens"])
        return 512

    @staticmethod
    def _needs_remote_group_beam_search(exc: Exception) -> bool:
        message = str(exc)
        return (
            "Group Beam Search requires `trust_remote_code=True`" in message
            or "transformers-community/group-beam-search" in message
        )

    @staticmethod
    def _remote_group_beam_generation_kwargs(generation_kwargs: dict[str, Any]) -> dict[str, Any]:
        remote_kwargs = dict(generation_kwargs)
        remote_kwargs["custom_generate"] = "transformers-community/group-beam-search"
        remote_kwargs["trust_remote_code"] = True
        return remote_kwargs

    def _supports_remote_group_beam_search(self) -> bool:
        try:
            parameters = inspect.signature(self.model.generate).parameters
        except (TypeError, ValueError):
            return False
        has_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        return "custom_generate" in parameters and (
            "trust_remote_code" in parameters or has_kwargs
        )

    def generate_candidates(self, prompt_text: str, target_count: int) -> list[str]:
        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_source_length,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        generation_kwargs = build_diverse_beam_generation_kwargs(
            self.generation_config,
            target_count=target_count,
        )
        use_remote_generation = (
            "num_beam_groups" in generation_kwargs and self.supports_remote_group_beam_search
        )
        primary_generation_kwargs = (
            self._remote_group_beam_generation_kwargs(generation_kwargs)
            if use_remote_generation
            else generation_kwargs
        )

        with self.torch.no_grad():
            try:
                generated_ids = self.model.generate(**encoded, **primary_generation_kwargs)
            except ValueError as exc:
                if not self._needs_remote_group_beam_search(exc):
                    raise
                if not self.supports_remote_group_beam_search:
                    raise RuntimeError(
                        "Installed transformers generate() does not expose remote group-beam support."
                    ) from exc
                if use_remote_generation:
                    raise
                generated_ids = self.model.generate(
                    **encoded,
                    **self._remote_group_beam_generation_kwargs(generation_kwargs),
                )

        raw_outputs = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )
        outputs = list(raw_outputs[: int(target_count)])
        if len(outputs) < int(target_count):
            raise RuntimeError(
                f"Requested {int(target_count)} outputs but decoded only {len(outputs)} candidates."
            )
        return outputs


__all__ = [
    "BioT5DiverseBeamGenerator",
    "build_diverse_beam_generation_kwargs",
]

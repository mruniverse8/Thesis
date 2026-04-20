from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from src.constants import BOM_TOKEN, EOM_TOKEN

from .sequence import STAGE_SEPARATOR, serialize_staged_molecule


def _sorted_unique_ints(values: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted({int(value) for value in values}))


def _tokenize_text_ids(tokenizer: Any, text: str) -> tuple[int, ...]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
    )
    input_ids = encoded["input_ids"]
    if isinstance(input_ids, torch.Tensor):
        if input_ids.ndim == 0:
            return (int(input_ids.item()),)
        if input_ids.ndim == 1:
            return tuple(int(token_id) for token_id in input_ids.tolist())
        if input_ids.ndim == 2:
            return tuple(int(token_id) for token_id in input_ids[0].tolist())
        raise ValueError("Unsupported input_ids tensor rank while building token constraints.")
    if input_ids and isinstance(input_ids[0], list):
        return tuple(int(token_id) for token_id in input_ids[0])
    return tuple(int(token_id) for token_id in input_ids)


def _iter_target_selfies_lists(
    examples: Sequence[Any] | Any,
) -> Iterable[Sequence[str]]:
    if examples is None:
        return
    records = getattr(examples, "records", examples)
    for item in records:
        if hasattr(item, "target_selfies_list"):
            target_selfies_list = getattr(item, "target_selfies_list")
        elif isinstance(item, dict):
            target_selfies_list = item.get("target_selfies_list", ())
        else:
            continue
        if not target_selfies_list:
            continue
        yield tuple(str(value) for value in target_selfies_list)


@dataclass(frozen=True)
class StageTokenConstraints:
    bom_token_id: int
    eom_token_id: int
    content_token_ids: tuple[int, ...]
    separator_token_ids: tuple[int, ...] = ()
    min_content_tokens_before_eom: int = 1

    @property
    def enabled(self) -> bool:
        return (
            self.bom_token_id >= 0
            and self.eom_token_id >= 0
            and bool(self.content_token_ids)
        )

    def allowed_token_ids_for_prefix(
        self,
        action_token_ids: Sequence[int],
    ) -> tuple[int, ...]:
        if not self.enabled:
            return ()
        if not action_token_ids:
            return (int(self.bom_token_id),)
        if self.eom_token_id in {int(token_id) for token_id in action_token_ids}:
            return ()
        if int(action_token_ids[0]) != int(self.bom_token_id):
            return (int(self.bom_token_id),)

        emitted_content_tokens = sum(
            1
            for token_id in action_token_ids
            if int(token_id) not in {int(self.bom_token_id), int(self.eom_token_id)}
        )
        allowed = list(self.content_token_ids)
        if emitted_content_tokens >= int(self.min_content_tokens_before_eom):
            allowed.append(int(self.eom_token_id))
        return _sorted_unique_ints(allowed)


def _validate_constraint_tokenizer(tokenizer: Any) -> tuple[int, int]:
    if tokenizer is None or not hasattr(tokenizer, "convert_tokens_to_ids"):
        raise ValueError("Tokenizer must define convert_tokens_to_ids for constrained decoding.")
    if not callable(getattr(tokenizer, "__call__", None)):
        raise ValueError("Tokenizer must be callable for constrained decoding.")

    bom_token_id = int(tokenizer.convert_tokens_to_ids(BOM_TOKEN))
    eom_token_id = int(tokenizer.convert_tokens_to_ids(EOM_TOKEN))
    if bom_token_id < 0 or eom_token_id < 0:
        raise ValueError("Tokenizer is missing required staged wrapper tokens.")
    return bom_token_id, eom_token_id


def _coerce_selfies_dict_cache_key(selfies_dict_path: str | Path) -> str:
    return str(Path(selfies_dict_path).expanduser())


@lru_cache(maxsize=None)
def _load_selfies_dictionary_symbols_cached(cache_key: str) -> tuple[str, ...]:
    path = Path(cache_key)
    if not path.is_file():
        raise FileNotFoundError(f"SELFIES dictionary file was not found: {path}")

    symbols: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        symbols.append(stripped)

    if not symbols:
        raise ValueError(f"SELFIES dictionary file is empty: {path}")
    return tuple(symbols)


def load_selfies_dictionary_symbols(selfies_dict_path: str | Path) -> tuple[str, ...]:
    return _load_selfies_dictionary_symbols_cached(
        _coerce_selfies_dict_cache_key(selfies_dict_path)
    )


def _collect_content_token_ids_from_selfies_dict(
    tokenizer: Any,
    selfies_dict_path: str | Path,
) -> set[int]:
    token_ids: set[int] = set()
    for symbol in load_selfies_dictionary_symbols(selfies_dict_path):
        token_ids.update(_tokenize_text_ids(tokenizer, symbol))
    return token_ids


def _collect_stage_token_ids_from_examples(
    tokenizer: Any,
    examples: Sequence[Any] | Any,
) -> set[int]:
    stage_token_ids: set[int] = set()
    for target_selfies_list in _iter_target_selfies_lists(examples):
        for molecule_selfies in target_selfies_list:
            stage_text = serialize_staged_molecule(str(molecule_selfies))
            stage_token_ids.update(_tokenize_text_ids(tokenizer, stage_text))
    return stage_token_ids


def build_stage_token_constraints(
    tokenizer: Any,
    examples: Sequence[Any] | Any = None,
    *,
    selfies_dict_path: str | Path | None = "molecules/dict/selfies_dict.txt",
    separator_token: str = STAGE_SEPARATOR,
    min_content_tokens_before_eom: int = 1,
) -> StageTokenConstraints:
    bom_token_id, eom_token_id = _validate_constraint_tokenizer(tokenizer)

    stage_token_ids: set[int] = set()
    if selfies_dict_path is not None:
        stage_token_ids.update(
            _collect_content_token_ids_from_selfies_dict(tokenizer, selfies_dict_path)
        )
    stage_token_ids.update(_collect_stage_token_ids_from_examples(tokenizer, examples))
    if not stage_token_ids:
        raise ValueError(
            "Constrained decoding requires at least one token id from the SELFIES dictionary "
            "or training targets."
        )

    try:
        separator_token_ids = _tokenize_text_ids(tokenizer, separator_token)
    except Exception:
        separator_token_ids = ()

    content_token_ids = _sorted_unique_ints(
        token_id
        for token_id in stage_token_ids
        if int(token_id) not in {bom_token_id, eom_token_id}
    )
    if not content_token_ids:
        raise ValueError("Constrained decoding produced no content token ids.")

    return StageTokenConstraints(
        bom_token_id=bom_token_id,
        eom_token_id=eom_token_id,
        content_token_ids=content_token_ids,
        separator_token_ids=_sorted_unique_ints(separator_token_ids),
        min_content_tokens_before_eom=max(1, int(min_content_tokens_before_eom)),
    )


def build_stage_token_constraints_from_examples(
    tokenizer: Any,
    examples: Sequence[Any] | Any,
    *,
    separator_token: str = STAGE_SEPARATOR,
    min_content_tokens_before_eom: int = 1,
) -> StageTokenConstraints | None:
    try:
        return build_stage_token_constraints(
            tokenizer,
            examples,
            selfies_dict_path=None,
            separator_token=separator_token,
            min_content_tokens_before_eom=min_content_tokens_before_eom,
        )
    except Exception:
        return None


def mask_logits_to_allowed_token_ids(
    logits: torch.Tensor,
    allowed_token_ids: Sequence[int],
) -> torch.Tensor:
    if not allowed_token_ids:
        raise ValueError("allowed_token_ids must be non-empty when masking logits.")

    allowed_indices = torch.tensor(
        [int(token_id) for token_id in allowed_token_ids],
        dtype=torch.long,
        device=logits.device,
    )
    masked_logits = torch.full_like(logits, float("-inf"))
    masked_logits[..., allowed_indices] = logits[..., allowed_indices]
    return masked_logits


def resolve_stage_token_constraints(
    owner: Any,
    stage_token_constraints: StageTokenConstraints | None = None,
) -> StageTokenConstraints | None:
    if stage_token_constraints is not None:
        return stage_token_constraints

    getter = getattr(owner, "get_stage_token_constraints", None)
    if callable(getter):
        return getter()

    stored_constraints = getattr(owner, "_stage_token_constraints", None)
    if isinstance(stored_constraints, StageTokenConstraints):
        return stored_constraints
    return None

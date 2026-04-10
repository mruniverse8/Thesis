#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molecules.selfies import (
    decode_selfies_to_smiles,
    filter_selfies,
    unwrap_selfies_target,
    wrap_selfies_target,
)
from src.datasets import TextToSelfiesDataset, build_processed_record
from src.io_utils import write_jsonl
from src.prompting import build_text2mol_prompt


def run_check(name: str, fn) -> tuple[str, bool, str]:
    try:
        fn()
        return name, True, ""
    except Exception as exc:
        return name, False, str(exc)


def check_prompt_shape() -> None:
    prompt = build_text2mol_prompt("A simple alcohol.")
    assert "Definition: You are given a molecule description in English." in prompt
    assert "Input: A simple alcohol." in prompt
    assert prompt.endswith("Output: ")


def check_target_wrap_round_trip() -> None:
    wrapped = wrap_selfies_target("[C] [O]")
    assert wrapped == "<bom>[C][O]<eom>"
    assert unwrap_selfies_target(wrapped) == "[C][O]"


def check_filter_selfies_recovers_tokens() -> None:
    recovered = filter_selfies("noise <bom>[C][C][O]<eom> trailing text")
    assert recovered == "[C][C][O]"


def check_decode_clean_selfies() -> None:
    smiles, used_repair = decode_selfies_to_smiles("[C][C][O]")
    assert smiles == "CCO"
    assert used_repair is False


def check_decode_noisy_selfies_with_repair() -> None:
    smiles, used_repair = decode_selfies_to_smiles("junk [C][C][O] junk")
    assert smiles == "CCO"
    assert used_repair is True


def check_processed_record_with_existing_selfies() -> None:
    record = build_processed_record(
        {
            "CID": 123,
            "description": "  Ethanol description. ",
            "SELFIES": "[C] [C] [O]",
            "SMILES": "CCO",
        },
        split="train",
        index=0,
    )
    assert record["id"] == "123"
    assert record["description"] == "Ethanol description."
    assert record["selfies"] == "[C][C][O]"
    assert record["source_smiles"] == "CCO"


def check_smiles_fallback_and_dataset_round_trip() -> None:
    record = build_processed_record(
        {
            "CID": 7,
            "description": "Methanol",
            "SELFIES": None,
            "SMILES": "CO",
        },
        split="validation",
        index=0,
    )
    assert record["selfies"]
    smiles, used_repair = decode_selfies_to_smiles(record["selfies"])
    assert smiles == "CO"
    assert used_repair is False

    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset_path = Path(tmp_dir) / "sample.jsonl"
        write_jsonl(dataset_path, [record])
        dataset = TextToSelfiesDataset.from_jsonl(dataset_path)
        example = dataset[0]
        assert example["description"] == "Methanol"
        assert example["prompt"].endswith("Output: ")
        assert example["target_text"].startswith("<bom>")
        assert example["target_text"].endswith("<eom>")


def main() -> None:
    checks = [
        ("prompt shape", check_prompt_shape),
        ("target wrap and unwrap", check_target_wrap_round_trip),
        ("SELFIES token filtering", check_filter_selfies_recovers_tokens),
        ("clean SELFIES decoding", check_decode_clean_selfies),
        ("noisy SELFIES repair decoding", check_decode_noisy_selfies_with_repair),
        ("processed record with SELFIES", check_processed_record_with_existing_selfies),
        ("SMILES fallback and dataset round trip", check_smiles_fallback_and_dataset_round_trip),
    ]

    results = [run_check(name, fn) for name, fn in checks]
    passed = 0
    for index, (name, ok, detail) in enumerate(results, start=1):
        status = "PASS" if ok else "FAIL"
        print(f"[{index}/{len(results)}] {status} - {name}")
        if detail:
            print(f"    {detail}")
        passed += int(ok)

    print(f"\nSummary: {passed}/{len(results)} checks passed")
    if passed != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

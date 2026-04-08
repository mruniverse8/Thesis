from src.datasets import build_processed_record


def test_build_processed_record_prefers_existing_selfies():
    raw_record = {
        "CID": 123,
        "description": "  Example molecule description. ",
        "SELFIES": "[C] [O]",
        "SMILES": "CO",
    }

    processed = build_processed_record(raw_record, split="train", index=0)

    assert processed["id"] == "123"
    assert processed["description"] == "Example molecule description."
    assert processed["selfies"] == "[C][O]"
    assert processed["source_smiles"] == "CO"


def test_build_processed_record_falls_back_to_smiles_when_selfies_missing():
    raw_record = {
        "CID": 7,
        "description": "Methanol",
        "SELFIES": None,
        "SMILES": "CO",
    }

    processed = build_processed_record(raw_record, split="validation", index=1)

    assert processed["id"] == "7"
    assert processed["description"] == "Methanol"
    assert processed["selfies"]

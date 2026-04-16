from __future__ import annotations

from pathlib import Path

from src.checkpoint_bootstrap import default_checkpoint_archive_path
from src.training import save_checkpoint


class DummyModel:
    def save_pretrained(self, output_dir: str | Path) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "model.bin").write_text("ok", encoding="utf-8")


class DummyTokenizer:
    def save_pretrained(self, output_dir: str | Path) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer.json").write_text("ok", encoding="utf-8")


def test_save_checkpoint_does_not_write_decoder_tokenizer_artifact(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"

    save_checkpoint(
        checkpoint_dir=checkpoint_dir,
        model=DummyModel(),
        training_tokenizer=DummyTokenizer(),
        config={"seed": 42},
        metrics={"loss": 1.0},
    )

    assert (checkpoint_dir / "model.bin").exists()
    assert (checkpoint_dir / "tokenizer.json").exists()
    assert (checkpoint_dir / "config.yaml").exists()
    assert (checkpoint_dir / "metrics.json").exists()
    assert not (checkpoint_dir / "decoder_tokenizer").exists()

def test_save_checkpoint_writes_archive_when_requested(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"

    archive_path = save_checkpoint(
        checkpoint_dir=checkpoint_dir,
        model=DummyModel(),
        training_tokenizer=DummyTokenizer(),
        config={"seed": 42},
        metrics={"loss": 1.0},
        create_archive=True,
    )

    assert archive_path == default_checkpoint_archive_path(checkpoint_dir)
    assert archive_path is not None and archive_path.exists()


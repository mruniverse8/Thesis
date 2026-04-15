from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pytest

from src.train_dataset_bootstrap import (
    DEFAULT_TRAIN_DATASET_FILE_ID,
    dataset_is_ready,
    download_google_drive_zip,
    extract_dataset_archive,
    missing_dataset_paths,
    required_dataset_paths,
)


def _build_zip_bytes() -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("mini_post_training/post_training_processed/train_multimol.jsonl", "{}\n")
        archive.writestr("mini_post_training/grouped_splits/train_multimol.jsonl", "{}\n")
        archive.writestr("mini_post_training/grouped_splits/validation_multimol.jsonl", "{}\n")
        archive.writestr("mini_post_training/grouped_splits/test_multimol.jsonl", "{}\n")
    return buffer.getvalue()


class _FakeResponse:
    def __init__(
        self,
        content: bytes,
        *,
        headers: dict[str, str] | None = None,
        cookies: dict[str, str] | None = None,
        text: str = "",
    ) -> None:
        self._content = content
        self.headers = headers or {}
        self.cookies = cookies or {}
        self.text = text

    def iter_content(self, chunk_size: int = 1024):
        for start in range(0, len(self._content), chunk_size):
            yield self._content[start : start + chunk_size]

    def raise_for_status(self) -> None:
        return None

    def close(self) -> None:
        return None


class _FakeSession:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def get(self, url: str, *, params: dict[str, str], stream: bool, timeout: int):
        self.calls.append(
            {
                "url": url,
                "params": dict(params),
                "stream": stream,
                "timeout": timeout,
            }
        )
        if not self._responses:
            raise AssertionError("No fake responses left")
        return self._responses.pop(0)

    def close(self) -> None:
        return None


def test_required_dataset_paths_match_expected_layout(tmp_path: Path) -> None:
    extract_dir = tmp_path / "mini_post_training"
    expected = [
        extract_dir / "post_training_processed" / "train_multimol.jsonl",
        extract_dir / "grouped_splits" / "train_multimol.jsonl",
        extract_dir / "grouped_splits" / "validation_multimol.jsonl",
        extract_dir / "grouped_splits" / "test_multimol.jsonl",
    ]

    assert required_dataset_paths(extract_dir) == expected


def test_dataset_is_ready_requires_all_expected_files(tmp_path: Path) -> None:
    extract_dir = tmp_path / "mini_post_training"
    extract_dir.mkdir(parents=True)
    for path in required_dataset_paths(extract_dir)[:-1]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")

    assert dataset_is_ready(extract_dir) is False
    assert missing_dataset_paths(extract_dir) == [required_dataset_paths(extract_dir)[-1]]


def test_extract_dataset_archive_normalizes_nested_zip_root(tmp_path: Path) -> None:
    archive_path = tmp_path / "mini_post_training.zip"
    archive_path.write_bytes(_build_zip_bytes())

    destination = extract_dataset_archive(archive_path, tmp_path / "mini_post_training")

    assert destination == tmp_path / "mini_post_training"
    assert dataset_is_ready(destination) is True


def test_download_google_drive_zip_handles_confirm_token(tmp_path: Path) -> None:
    archive_path = tmp_path / "dataset.zip"
    zip_bytes = _build_zip_bytes()
    session = _FakeSession(
        [
            _FakeResponse(
                b"<html>confirm</html>",
                headers={"content-type": "text/html; charset=utf-8"},
                cookies={"download_warning_123": "token123"},
                text="<a href='?confirm=token123'>continue</a>",
            ),
            _FakeResponse(zip_bytes, headers={"content-type": "application/zip"}),
        ]
    )

    download_google_drive_zip(DEFAULT_TRAIN_DATASET_FILE_ID, archive_path, session=session)

    assert archive_path.exists()
    assert len(session.calls) == 2
    assert session.calls[0]["params"] == {"export": "download", "id": DEFAULT_TRAIN_DATASET_FILE_ID}
    assert session.calls[1]["params"] == {
        "export": "download",
        "id": DEFAULT_TRAIN_DATASET_FILE_ID,
        "confirm": "token123",
    }


def test_download_google_drive_zip_rejects_non_zip_payload(tmp_path: Path) -> None:
    archive_path = tmp_path / "dataset.zip"
    session = _FakeSession(
        [
            _FakeResponse(b"not a zip", headers={"content-type": "text/plain"}),
        ]
    )

    with pytest.raises(RuntimeError, match="not a valid zip archive"):
        download_google_drive_zip(DEFAULT_TRAIN_DATASET_FILE_ID, archive_path, session=session)

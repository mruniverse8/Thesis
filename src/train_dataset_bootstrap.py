from __future__ import annotations

import re
import shutil
import zipfile
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

import requests

from .io_utils import ensure_dir

DEFAULT_TRAIN_DATASET_FILE_ID = "1fwRIHrcq0nGA1OJCCWqdxvGgcbW2oKY3"
DEFAULT_TRAIN_DATASET_ZIP_PATH = Path("data") / "mini_post_training.zip"
DEFAULT_TRAIN_DATASET_EXTRACT_DIR = Path("data") / "mini_post_training"
GOOGLE_DRIVE_DOWNLOAD_URL = "https://drive.google.com/uc"
MINI_POST_TRAINING_REQUIRED_RELATIVE_PATHS = (
    Path("post_training_processed") / "train_multimol.jsonl",
    Path("grouped_splits") / "train_multimol.jsonl",
    Path("grouped_splits") / "validation_multimol.jsonl",
    Path("grouped_splits") / "test_multimol.jsonl",
)


def required_dataset_paths(extract_dir: str | Path) -> list[Path]:
    root = Path(extract_dir)
    return [root / relative_path for relative_path in MINI_POST_TRAINING_REQUIRED_RELATIVE_PATHS]


def missing_dataset_paths(extract_dir: str | Path) -> list[Path]:
    return [path for path in required_dataset_paths(extract_dir) if not path.exists()]


def dataset_is_ready(extract_dir: str | Path) -> bool:
    return not missing_dataset_paths(extract_dir)


def reset_download_targets(zip_path: str | Path, extract_dir: str | Path) -> None:
    zip_destination = Path(zip_path)
    extract_destination = Path(extract_dir)
    if zip_destination.exists():
        zip_destination.unlink()
    if extract_destination.exists():
        shutil.rmtree(extract_destination)


def _extract_confirm_token(response: requests.Response) -> str | None:
    for cookie_name, cookie_value in response.cookies.items():
        if cookie_name.startswith("download_warning"):
            return cookie_value

    content_type = (response.headers.get("content-type") or "").lower()
    if "text/html" not in content_type:
        return None

    try:
        text = response.text
    except Exception:
        return None

    match = re.search(r'confirm=([0-9A-Za-z_]+)', text)
    if match is None:
        return None
    return match.group(1)


class _GoogleDriveConfirmFormParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.form_action: str | None = None
        self.form_inputs: dict[str, str] = {}
        self._inside_target_form = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {name: value for name, value in attrs}
        if tag == "form" and self.form_action is None:
            action = attributes.get("action")
            if action:
                self.form_action = action
                self.form_inputs = {}
                self._inside_target_form = True
            return

        if tag != "input" or not self._inside_target_form:
            return

        name = attributes.get("name")
        value = attributes.get("value")
        if name is None or value is None:
            return
        self.form_inputs[name] = value

    def handle_endtag(self, tag: str) -> None:
        if tag == "form" and self._inside_target_form:
            self._inside_target_form = False


def _extract_confirm_form(response: requests.Response) -> tuple[str, dict[str, str]] | None:
    content_type = (response.headers.get("content-type") or "").lower()
    if "text/html" not in content_type:
        return None

    try:
        text = response.text
    except Exception:
        return None

    parser = _GoogleDriveConfirmFormParser()
    parser.feed(text)

    if parser.form_action is None:
        return None
    if parser.form_inputs.get("confirm") != "t":
        return None

    response_url = getattr(response, "url", GOOGLE_DRIVE_DOWNLOAD_URL)
    return urljoin(response_url, unescape(parser.form_action)), parser.form_inputs


def _write_response_content(response: requests.Response, destination_path: Path) -> None:
    ensure_dir(destination_path.parent)
    with destination_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)


def download_google_drive_zip(
    file_id: str,
    destination_path: str | Path,
    *,
    session: requests.Session | None = None,
) -> Path:
    target_path = Path(destination_path)
    active_session = session or requests.Session()
    should_close_session = session is None
    response: requests.Response | None = None

    try:
        response = active_session.get(
            GOOGLE_DRIVE_DOWNLOAD_URL,
            params={"export": "download", "id": file_id},
            stream=True,
            timeout=120,
        )
        response.raise_for_status()

        confirm_token = _extract_confirm_token(response)
        if confirm_token is not None:
            response.close()
            response = active_session.get(
                GOOGLE_DRIVE_DOWNLOAD_URL,
                params={"export": "download", "id": file_id, "confirm": confirm_token},
                stream=True,
                timeout=120,
            )
            response.raise_for_status()
        else:
            confirm_form = _extract_confirm_form(response)
            if confirm_form is not None:
                confirm_url, confirm_params = confirm_form
                response.close()
                response = active_session.get(
                    confirm_url,
                    params=confirm_params,
                    stream=True,
                    timeout=120,
                )
                response.raise_for_status()

        _write_response_content(response, target_path)
    finally:
        if response is not None:
            response.close()
        if should_close_session:
            active_session.close()

    if not zipfile.is_zipfile(target_path):
        content_type = response.headers.get("content-type") if response is not None else None
        raise RuntimeError(
            "Downloaded file is not a valid zip archive. "
            f"file_id={file_id!r}, destination={target_path}, content_type={content_type!r}"
        )
    return target_path


def _find_extracted_dataset_root(extracted_root: Path) -> Path | None:
    if dataset_is_ready(extracted_root):
        return extracted_root
    for child in sorted(extracted_root.iterdir()):
        if child.is_dir() and dataset_is_ready(child):
            return child
    return None


def extract_dataset_archive(zip_path: str | Path, extract_dir: str | Path) -> Path:
    archive_path = Path(zip_path)
    destination_root = Path(extract_dir)
    staging_root = destination_root.parent / f".{destination_root.name}.extracting"

    if not archive_path.exists():
        raise FileNotFoundError(f"Missing dataset archive: {archive_path}")
    if not zipfile.is_zipfile(archive_path):
        raise ValueError(f"Expected a zip archive at {archive_path}")

    if staging_root.exists():
        shutil.rmtree(staging_root)
    ensure_dir(staging_root)

    try:
        with zipfile.ZipFile(archive_path, "r") as handle:
            handle.extractall(staging_root)

        source_root = _find_extracted_dataset_root(staging_root)
        if source_root is None:
            raise RuntimeError(
                "The extracted archive does not contain the expected mini post-training dataset layout."
            )

        if destination_root.exists():
            shutil.rmtree(destination_root)

        if source_root == staging_root:
            shutil.move(str(staging_root), str(destination_root))
        else:
            shutil.move(str(source_root), str(destination_root))
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)

    return destination_root

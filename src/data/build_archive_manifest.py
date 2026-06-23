from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import pandas as pd


ARCHIVE_METADATA_COLUMNS = [
    "speaker_id",
    "sample_name",
    "audio_path",
    "audio_url",
    "native_language",
    "all_native_languages",
    "country_of_origin",
    "transcript",
    "birth_place",
    "other_languages",
    "age",
    "sex",
    "age_of_english_onset",
    "english_learning_method",
    "english_residence",
    "length_of_english_residence",
    "detail_url",
    "source_metadata_rows",
    "duration_sec",
    "split",
]


def _clean_text(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _normalize_label(value: Any) -> str:
    text = _clean_text(value) or ""
    return re.sub(r"\s+", " ", text.lower()).strip()


def _country_from_birth_place(value: Any) -> str | None:
    text = _normalize_label(value)
    text = re.sub(r"\s*\(map\)\s*$", "", text).strip()
    parts = [part.strip() for part in text.split(",") if part.strip()]
    return parts[-1] if parts else None


def _first_nonempty(values: pd.Series) -> str | None:
    for value in values:
        text = _clean_text(value)
        if text:
            return text
    return None


def _joined_unique(values: pd.Series) -> str:
    labels = sorted({_normalize_label(value) for value in values if _normalize_label(value)})
    return ";".join(labels)


def _resolve_audio_path(row: pd.Series, audio_dir: Path | None) -> str | None:
    audio_path = _clean_text(row.get("audio_path"))
    if audio_path and Path(audio_path).exists():
        return audio_path

    if audio_dir is None:
        return audio_path

    sample_name = _clean_text(row.get("sample_name"))
    if not sample_name:
        return audio_path

    matches = sorted(audio_dir.glob(f"{sample_name}.*"))
    if matches:
        return str(matches[0])
    return audio_path


def build_archive_manifest(
    metadata_path: str | Path,
    audio_dir: str | Path | None = None,
    default_transcript: str | None = None,
) -> pd.DataFrame:
    """Build a one-row-per-recording manifest from full Speech Accent Archive metadata."""
    metadata_file = Path(metadata_path)
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {metadata_file}")

    metadata = pd.read_csv(metadata_file)
    required = {"speaker_id", "sample_name", "audio_url"}
    missing = required.difference(metadata.columns)
    if missing:
        raise KeyError(f"Metadata is missing required columns: {sorted(missing)}")

    audio_path = Path(audio_dir) if audio_dir is not None else None
    if audio_path is not None and not audio_path.exists():
        raise FileNotFoundError(f"Audio directory does not exist: {audio_path}")

    prepared = metadata.copy()
    prepared["resolved_audio_path"] = prepared.apply(_resolve_audio_path, axis=1, audio_dir=audio_path)
    prepared = prepared[prepared["resolved_audio_path"].map(lambda value: bool(_clean_text(value)) and Path(str(value)).exists())]

    if prepared.empty:
        raise ValueError(
            "No metadata rows could be paired with downloaded audio files. "
            "Run `python -m src.cli download-archive-audio` first."
        )

    rows: list[dict[str, Any]] = []
    for _, group in prepared.groupby("audio_url", sort=True, dropna=False):
        first = group.iloc[0]
        birth_place = _first_nonempty(group.get("birth_place", pd.Series(dtype=object)))
        transcript = _first_nonempty(group.get("transcript", pd.Series(dtype=object))) or default_transcript or ""
        all_native_languages = _joined_unique(group["native_language"]) if "native_language" in group else ""

        row = {
            "speaker_id": _clean_text(first.get("speaker_id")),
            "sample_name": _clean_text(first.get("sample_name")),
            "audio_path": _clean_text(first.get("resolved_audio_path")),
            "audio_url": _clean_text(first.get("audio_url")),
            "native_language": all_native_languages.split(";")[0] if all_native_languages else None,
            "all_native_languages": all_native_languages,
            "country_of_origin": _country_from_birth_place(birth_place),
            "transcript": transcript,
            "birth_place": birth_place,
            "other_languages": _first_nonempty(group.get("other_languages", pd.Series(dtype=object))),
            "age": _first_nonempty(group.get("age", pd.Series(dtype=object))),
            "sex": _first_nonempty(group.get("sex", pd.Series(dtype=object))),
            "age_of_english_onset": _first_nonempty(group.get("age_of_english_onset", pd.Series(dtype=object))),
            "english_learning_method": _first_nonempty(group.get("english_learning_method", pd.Series(dtype=object))),
            "english_residence": _first_nonempty(group.get("english_residence", pd.Series(dtype=object))),
            "length_of_english_residence": _first_nonempty(group.get("length_of_english_residence", pd.Series(dtype=object))),
            "detail_url": _first_nonempty(group.get("detail_url", pd.Series(dtype=object))),
            "source_metadata_rows": int(len(group)),
            "duration_sec": None,
            "split": None,
        }
        rows.append(row)

    manifest = pd.DataFrame(rows)
    manifest = manifest.sort_values(["native_language", "speaker_id", "sample_name"], na_position="last").reset_index(drop=True)
    for column in ARCHIVE_METADATA_COLUMNS:
        if column not in manifest.columns:
            manifest[column] = None
    return manifest[ARCHIVE_METADATA_COLUMNS]

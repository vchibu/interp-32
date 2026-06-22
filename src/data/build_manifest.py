from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import re
from typing import Any

import pandas as pd


@dataclass(slots=True)
class ManifestRow:
    speaker_id: str
    audio_path: str
    transcript: str
    country: str | None = None
    native_language: str | None = None
    sex: str | None = None
    age: str | None = None
    accent_group: str | None = None
    duration_sec: float | None = None
    split: str | None = None


CANONICAL_COLUMNS = [
    "speaker_id",
    "audio_path",
    "transcript",
    "country",
    "native_language",
    "sex",
    "age",
    "accent_group",
    "duration_sec",
    "split",
]

METADATA_PATTERNS: dict[str, list[str]] = {
    "speaker_id": ["speaker_id", "speaker", "id", "participant_id", "filename", "file_name", "audio_id"],
    "audio_path": ["audio_path", "path", "file_path", "audio_file", "wav_path", "mp3_path"],
    "country": ["country", "birth_country", "origin_country"],
    "birth_place": ["birth_place", "birthplace", "place_of_birth"],
    "native_language": ["native_language", "l1", "language", "first_language"],
    "sex": ["sex", "gender"],
    "age": ["age"],
    "transcript": ["transcript", "text", "prompt", "sentence", "reading_text"],
}

AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}


def _normalize_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _discover_audio_files(raw_path: Path) -> list[Path]:
    return sorted(path for path in raw_path.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES)


def _discover_metadata_files(raw_path: Path) -> list[Path]:
    candidates: list[Path] = []
    for suffix in (".csv", ".tsv", ".txt", ".json", ".jsonl"):
        candidates.extend(path for path in raw_path.rglob(f"*{suffix}") if path.is_file())
    return sorted(candidates)


def _read_metadata_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(path, sep=None, engine="python")
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict):
            for value in data.values():
                if isinstance(value, list):
                    return pd.DataFrame(value)
        raise ValueError(f"Unsupported JSON metadata layout in {path}")
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return pd.DataFrame(rows)
    raise ValueError(f"Unsupported metadata file: {path}")


def _score_metadata_frame(df: pd.DataFrame) -> int:
    normalized_columns = [_normalize_name(column) for column in df.columns]
    joined = " ".join(normalized_columns)
    score = 0
    for aliases in METADATA_PATTERNS.values():
        if any(alias in joined for alias in aliases):
            score += 1
    return score


def _pick_metadata_frame(raw_path: Path) -> tuple[Path, pd.DataFrame]:
    best_path: Path | None = None
    best_df: pd.DataFrame | None = None
    best_score = -1

    for path in _discover_metadata_files(raw_path):
        try:
            df = _read_metadata_file(path)
        except Exception:
            continue
        if df.empty:
            continue
        score = _score_metadata_frame(df)
        if score > best_score:
            best_path = path
            best_df = df
            best_score = score

    if best_path is None or best_df is None:
        raise FileNotFoundError(
            "Could not find a readable metadata table in data/raw/. "
            "Add a CSV/TSV/JSON export from the dataset first."
        )
    return best_path, best_df


def _find_column(df: pd.DataFrame, logical_name: str) -> str | None:
    aliases = METADATA_PATTERNS[logical_name]
    normalized = {_normalize_name(column): column for column in df.columns}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    for alias in aliases:
        for normalized_name, original_name in normalized.items():
            if alias == normalized_name or alias in normalized_name:
                return original_name
    return None


def _audio_lookup(audio_files: list[Path], raw_path: Path) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    for file_path in audio_files:
        relative = file_path.relative_to(raw_path)
        keys = {
            file_path.stem.lower(),
            file_path.name.lower(),
            str(relative).replace("\\", "/").lower(),
        }
        for key in keys:
            lookup.setdefault(key, file_path)
    return lookup


def _derive_accent_group(country: str | None) -> str | None:
    if not country or not isinstance(country, str):
        return None
    key = country.strip().lower()
    region_map = {
        "netherlands": "europe",
        "germany": "europe",
        "france": "europe",
        "spain": "europe",
        "italy": "europe",
        "poland": "europe",
        "romania": "europe",
        "united kingdom": "europe",
        "ireland": "europe",
        "india": "south_asia",
        "pakistan": "south_asia",
        "bangladesh": "south_asia",
        "china": "east_asia",
        "japan": "east_asia",
        "korea": "east_asia",
        "south korea": "east_asia",
        "nigeria": "africa",
        "ghana": "africa",
        "kenya": "africa",
        "south africa": "africa",
        "united states": "north_america",
        "canada": "north_america",
        "mexico": "latin_america",
        "brazil": "latin_america",
        "argentina": "latin_america",
        "colombia": "latin_america",
        "australia": "oceania",
        "new zealand": "oceania",
    }
    return region_map.get(key)


def _extract_country_from_birth_place(birth_place: str | None) -> str | None:
    if not birth_place:
        return None
    parts = [part.strip() for part in birth_place.split(",") if part.strip()]
    if not parts:
        return None
    return parts[-1]


def _resolve_audio_path(
    row: pd.Series,
    audio_path_column: str | None,
    speaker_id_column: str | None,
    lookup: dict[str, Path],
    raw_path: Path,
) -> Path | None:
    if audio_path_column and pd.notna(row[audio_path_column]):
        raw_value = str(row[audio_path_column]).strip()
        raw_key = raw_value.replace("\\", "/").lower()
        candidate = lookup.get(raw_key)
        if candidate is not None:
            return candidate
        path_candidate = Path(raw_value)
        if path_candidate.exists():
            return path_candidate
        direct_path = raw_path / raw_value
        if direct_path.exists():
            return direct_path
        for key in (path_candidate.name.lower(), path_candidate.stem.lower()):
            candidate = lookup.get(key)
            if candidate is not None:
                return candidate

    if speaker_id_column and pd.notna(row[speaker_id_column]):
        speaker_value = str(row[speaker_id_column]).strip()
        for key in (speaker_value.lower(), Path(speaker_value).stem.lower()):
            candidate = lookup.get(key)
            if candidate is not None:
                return candidate
    return None


def _coerce_string(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def build_manifest(raw_dir: str | Path, default_transcript: str | None = None) -> pd.DataFrame:
    """Create a canonical manifest from raw dataset files with best-effort autodetection."""
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_path}")

    metadata_path, metadata_df = _pick_metadata_frame(raw_path)
    audio_files = _discover_audio_files(raw_path)
    if not audio_files:
        raise FileNotFoundError("No audio files were found in data/raw/.")

    speaker_id_column = _find_column(metadata_df, "speaker_id")
    audio_path_column = _find_column(metadata_df, "audio_path")
    country_column = _find_column(metadata_df, "country")
    birth_place_column = _find_column(metadata_df, "birth_place")
    native_language_column = _find_column(metadata_df, "native_language")
    sex_column = _find_column(metadata_df, "sex")
    age_column = _find_column(metadata_df, "age")
    transcript_column = _find_column(metadata_df, "transcript")

    lookup = _audio_lookup(audio_files, raw_path)
    rows: list[ManifestRow] = []

    for row_index, row in metadata_df.iterrows():
        resolved_audio = _resolve_audio_path(row, audio_path_column, speaker_id_column, lookup, raw_path)
        if resolved_audio is None:
            continue

        speaker_id = _coerce_string(row[speaker_id_column]) if speaker_id_column else None
        if speaker_id is None:
            speaker_id = resolved_audio.stem

        country = _coerce_string(row[country_column]) if country_column else None
        if country is None and birth_place_column:
            country = _extract_country_from_birth_place(_coerce_string(row[birth_place_column]))
        native_language = _coerce_string(row[native_language_column]) if native_language_column else None
        transcript = _coerce_string(row[transcript_column]) if transcript_column else default_transcript

        rows.append(
            ManifestRow(
                speaker_id=speaker_id,
                audio_path=str(resolved_audio),
                transcript=transcript or "",
                country=country,
                native_language=native_language,
                sex=_coerce_string(row[sex_column]) if sex_column else None,
                age=_coerce_string(row[age_column]) if age_column else None,
                accent_group=_derive_accent_group(country),
                duration_sec=None,
                split=None,
            )
        )

    if not rows:
        raise ValueError(
            f"Found metadata file at {metadata_path}, but could not pair any rows with audio files. "
            "Check that speaker IDs or audio path columns match the filenames."
        )

    manifest = pd.DataFrame(asdict(row) for row in rows)
    manifest = manifest.drop_duplicates(subset=["speaker_id"]).reset_index(drop=True)
    for column in CANONICAL_COLUMNS:
        if column not in manifest.columns:
            manifest[column] = None
    return manifest[CANONICAL_COLUMNS]


def save_manifest(df: pd.DataFrame, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return output

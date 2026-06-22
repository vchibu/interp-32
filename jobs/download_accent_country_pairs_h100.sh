#!/usr/bin/env bash
#SBATCH --job-name=accent-pairs
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd /home/scur0061/interp-project/interp-32
mkdir -p logs

source .venv/bin/activate
export PYTHONUNBUFFERED=1

python -u - <<'PY'
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pandas as pd


USER_AGENT = "speech-accent-wav2vec2-analysis/0.1"
PAIR_TABLE = Path("outputs/analysis/accent_group_country_breakdown.csv")
METADATA_FILE = Path("data/raw/accent_archive_metadata.csv")
OUTPUT_DIR = Path("data/pair_downloads_min_10")
MIN_TOTAL_SPEAKERS = 10
MAX_SPEAKERS_PER_PAIR = 50


@dataclass(slots=True)
class PairSpeaker:
    speaker_id: str
    sample_name: str
    native_language: str
    country_of_origin: str
    detail_url: str
    audio_url: str
    local_audio_path: str
    transcript: str | None = None
    birth_place: str | None = None
    other_languages: str | None = None
    age: str | None = None
    sex: str | None = None
    age_of_english_onset: str | None = None
    english_learning_method: str | None = None
    english_residence: str | None = None
    length_of_english_residence: str | None = None


def _normalize_text(value: object) -> str:
    return str(value).strip().lower()


def _normalize_country(value: object) -> str:
    text = _normalize_text(value)
    text = re.sub(r"\s*\(map\)\s*$", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_country_from_birth_place(value: object) -> str:
    text = _normalize_country(value)
    parts = [part.strip() for part in text.split(",") if part.strip()]
    return parts[-1] if parts else ""


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    return re.sub(r"_+", "_", slug).strip("_") or "unknown"


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": USER_AGENT})
    response = urlopen(request, timeout=90)
    try:
        destination.write_bytes(response.read())
    finally:
        response.close()


def _log(message: str) -> None:
    print(message, flush=True)


def _prepare_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    required = {"speaker_id", "sample_name", "native_language", "audio_url", "detail_url"}
    missing = required.difference(metadata.columns)
    if missing:
        raise KeyError(f"Missing required columns in metadata: {sorted(missing)}")

    if "birth_place" not in metadata.columns:
        raise KeyError("Missing required column in metadata: ['birth_place']")

    prepared = metadata.copy()
    prepared["native_language"] = prepared["native_language"].map(_normalize_text)
    prepared["country_of_origin"] = prepared["birth_place"].map(_extract_country_from_birth_place)
    prepared = prepared[(prepared["native_language"] != "") & (prepared["country_of_origin"] != "")].copy()
    prepared["speaker_id"] = prepared["speaker_id"].astype(str).str.strip()
    prepared["sample_name"] = prepared["sample_name"].astype(str).str.strip()
    return prepared


def _pair_folder(accent_group: str, country_of_origin: str) -> Path:
    return OUTPUT_DIR / f"{_slugify(accent_group)}__{_slugify(country_of_origin)}"


def _write_pair_metadata(pair_dir: Path, rows: list[PairSpeaker]) -> None:
    with (pair_dir / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(asdict(rows[0]).keys()) if rows else list(PairSpeaker.__dataclass_fields__.keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> None:
    _log("Loading pair table and metadata...")
    pair_table = pd.read_csv(PAIR_TABLE)
    metadata = pd.read_csv(METADATA_FILE)
    metadata = _prepare_metadata(metadata)

    if not {"accent_group", "country_of_origin", "total_speakers"}.issubset(pair_table.columns):
        raise KeyError("pair table must contain accent_group, country_of_origin, and total_speakers")

    pairs = pair_table.copy()
    pairs["accent_group"] = pairs["accent_group"].map(_normalize_text)
    pairs["country_of_origin"] = pairs["country_of_origin"].map(_normalize_country)
    pairs = pairs[pairs["total_speakers"] >= MIN_TOTAL_SPEAKERS].drop_duplicates(
        subset=["accent_group", "country_of_origin"]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _log(f"Downloading {len(pairs)} accent-group/country pairs into {OUTPUT_DIR}")

    for pair_position, (_, pair) in enumerate(pairs.iterrows(), start=1):
        accent_group = str(pair["accent_group"]).strip()
        country_of_origin = str(pair["country_of_origin"]).strip()
        pair_dir = _pair_folder(accent_group, country_of_origin)
        audio_dir = pair_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        _log(
            f"[{pair_position}/{len(pairs)}] Starting pair: {accent_group} / {country_of_origin} "
            f"(expected speakers: {int(pair['total_speakers'])})"
        )

        pair_rows = metadata[
            (metadata["native_language"] == accent_group)
            & (metadata["country_of_origin"] == country_of_origin)
        ].copy()
        pair_rows = pair_rows.sort_values(["speaker_id", "sample_name"], ascending=[True, True]).reset_index(drop=True)
        pair_rows = pair_rows.head(MAX_SPEAKERS_PER_PAIR).reset_index(drop=True)

        _log(f"[{pair_position}/{len(pairs)}] Found {len(pair_rows)} matching speaker rows (capped at {MAX_SPEAKERS_PER_PAIR})")

        downloaded_rows: list[PairSpeaker] = []
        for speaker_position, (_, row) in enumerate(pair_rows.iterrows(), start=1):
            audio_url = str(row["audio_url"]).strip()
            suffix = Path(urlparse(audio_url).path).suffix or ".mp3"
            local_audio = audio_dir / f"{_slugify(str(row['speaker_id']))}{suffix}"
            if not local_audio.exists():
                try:
                    _download_file(audio_url, local_audio)
                except (HTTPError, URLError, TimeoutError) as exc:
                    _log(f"[{pair_position}/{len(pairs)}] Skipped {accent_group}/{country_of_origin} speaker {row['speaker_id']} because download failed: {exc}")
                    continue

            if speaker_position == 1 or speaker_position % 10 == 0 or speaker_position == len(pair_rows):
                _log(
                    f"[{pair_position}/{len(pairs)}] Downloaded {speaker_position}/{len(pair_rows)} speakers "
                    f"for {accent_group} / {country_of_origin}"
                )

            downloaded_rows.append(
                PairSpeaker(
                    speaker_id=str(row["speaker_id"]).strip(),
                    sample_name=str(row["sample_name"]).strip(),
                    native_language=str(row["native_language"]).strip(),
                    country_of_origin=str(row["country_of_origin"]).strip(),
                    detail_url=str(row["detail_url"]).strip(),
                    audio_url=audio_url,
                    local_audio_path=str(local_audio.relative_to(pair_dir)),
                    transcript=None if pd.isna(row.get("transcript")) else str(row.get("transcript")).strip(),
                    birth_place=None if pd.isna(row.get("birth_place")) else str(row.get("birth_place")).strip(),
                    other_languages=None if pd.isna(row.get("other_languages")) else str(row.get("other_languages")).strip(),
                    age=None if pd.isna(row.get("age")) else str(row.get("age")).strip(),
                    sex=None if pd.isna(row.get("sex")) else str(row.get("sex")).strip(),
                    age_of_english_onset=None if pd.isna(row.get("age_of_english_onset")) else str(row.get("age_of_english_onset")).strip(),
                    english_learning_method=None if pd.isna(row.get("english_learning_method")) else str(row.get("english_learning_method")).strip(),
                    english_residence=None if pd.isna(row.get("english_residence")) else str(row.get("english_residence")).strip(),
                    length_of_english_residence=None if pd.isna(row.get("length_of_english_residence")) else str(row.get("length_of_english_residence")).strip(),
                )
            )

        _write_pair_metadata(pair_dir, downloaded_rows)
        pair_rows.to_csv(pair_dir / "pair_summary.csv", index=False)
        _log(f"[{pair_position}/{len(pairs)}] Wrote {len(downloaded_rows)} files for {accent_group} / {country_of_origin} into {pair_dir}")

    _log("All pair downloads complete.")


if __name__ == "__main__":
    main()
PY
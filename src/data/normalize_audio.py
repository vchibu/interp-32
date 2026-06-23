from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import pandas as pd
import soundfile as sf
from tqdm import tqdm


def normalize_audio_file(
    source_path: str | Path,
    target_path: str | Path,
    sample_rate: int = 16000,
) -> Path:
    """Load audio, resample to mono 16 kHz, and write WAV output."""
    audio, _ = librosa.load(source_path, sr=sample_rate, mono=True)
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(target, audio, sample_rate)
    return target


def _normalize_manifest_row(
    row: dict,
    processed_audio_dir: str,
    sample_rate: int,
    overwrite: bool,
) -> dict:
    output_dir = Path(processed_audio_dir)
    source_path = Path(row["audio_path"])
    target_path = output_dir / f"{row['speaker_id']}.wav"

    if overwrite or not target_path.exists() or target_path.stat().st_size == 0:
        normalize_audio_file(source_path, target_path, sample_rate=sample_rate)

    updated_row = dict(row)
    updated_row["normalized_audio_path"] = str(target_path)
    updated_row["duration_sec"] = float(librosa.get_duration(path=str(target_path)))
    return updated_row


def normalize_from_manifest(
    manifest: pd.DataFrame,
    processed_audio_dir: str | Path,
    sample_rate: int = 16000,
    overwrite: bool = False,
    num_workers: int = 1,
) -> pd.DataFrame:
    """Normalize every manifest row and append normalized paths and durations."""
    output_dir = Path(processed_audio_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = manifest.to_dict(orient="records")
    workers = max(1, int(num_workers))
    if workers == 1:
        normalized_rows = [
            _normalize_manifest_row(row, str(output_dir), sample_rate, overwrite)
            for row in tqdm(rows, desc="Normalizing audio")
        ]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            normalized_rows = list(
                tqdm(
                    executor.map(
                        _normalize_manifest_row,
                        rows,
                        [str(output_dir)] * len(rows),
                        [sample_rate] * len(rows),
                        [overwrite] * len(rows),
                    ),
                    total=len(rows),
                    desc=f"Normalizing audio ({workers} workers)",
                )
            )

    return pd.DataFrame(normalized_rows)

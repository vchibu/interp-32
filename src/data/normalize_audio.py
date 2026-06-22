from __future__ import annotations

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


def normalize_from_manifest(
    manifest: pd.DataFrame,
    processed_audio_dir: str | Path,
    sample_rate: int = 16000,
) -> pd.DataFrame:
    """Normalize every manifest row and append normalized paths and durations."""
    output_dir = Path(processed_audio_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized_rows = []
    for row in tqdm(manifest.to_dict(orient="records"), desc="Normalizing audio"):
        source_path = Path(row["audio_path"])
        target_path = output_dir / f"{row['speaker_id']}.wav"
        normalize_audio_file(source_path, target_path, sample_rate=sample_rate)
        duration = librosa.get_duration(path=str(target_path))

        updated_row = dict(row)
        updated_row["normalized_audio_path"] = str(target_path)
        updated_row["duration_sec"] = float(duration)
        normalized_rows.append(updated_row)

    return pd.DataFrame(normalized_rows)

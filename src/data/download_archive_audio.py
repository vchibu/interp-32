from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pandas as pd
from tqdm import tqdm


USER_AGENT = "speech-accent-wav2vec2-analysis/0.1"


@dataclass(slots=True)
class AudioDownloadRecord:
    audio_url: str
    sample_name: str
    speaker_id: str
    local_audio_path: str
    status: str
    error: str | None = None


def _slug_from_sample_name(sample_name: object, fallback: str) -> str:
    text = str(sample_name).strip()
    if not text or text.lower() == "nan":
        return fallback
    return text.replace("/", "_").replace("\\", "_")


def _audio_destination(audio_url: str, sample_name: object, speaker_id: object, audio_dir: Path) -> Path:
    parsed = urlparse(audio_url)
    suffix = Path(parsed.path).suffix or ".mp3"
    fallback = f"speaker_{str(speaker_id).strip()}"
    filename = f"{_slug_from_sample_name(sample_name, fallback)}{suffix}"
    return audio_dir / filename


def _download_file(url: str, destination: Path, timeout: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": USER_AGENT})
    response = urlopen(request, timeout=timeout)
    try:
        destination.write_bytes(response.read())
    finally:
        response.close()


def download_audio_from_metadata(
    metadata_path: str | Path,
    audio_dir: str | Path,
    output_metadata_path: str | Path,
    download_index_path: str | Path,
    timeout: int = 90,
) -> tuple[Path, Path]:
    """Download every unique audio URL in the archive metadata without filtering or caps."""
    metadata_file = Path(metadata_path)
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {metadata_file}")

    metadata = pd.read_csv(metadata_file)
    required = {"speaker_id", "sample_name", "audio_url"}
    missing = required.difference(metadata.columns)
    if missing:
        raise KeyError(f"Metadata is missing required columns: {sorted(missing)}")

    audio_path = Path(audio_dir)
    audio_path.mkdir(parents=True, exist_ok=True)

    unique_audio = (
        metadata.dropna(subset=["audio_url"])
        .drop_duplicates(subset=["audio_url"])
        .loc[:, ["audio_url", "sample_name", "speaker_id"]]
        .reset_index(drop=True)
    )

    records: list[AudioDownloadRecord] = []
    url_to_local_path: dict[str, str] = {}

    for row in tqdm(unique_audio.to_dict(orient="records"), desc="Downloading archive audio"):
        url = str(row["audio_url"]).strip()
        destination = _audio_destination(url, row["sample_name"], row["speaker_id"], audio_path)
        status = "downloaded"
        error = None

        if destination.exists() and destination.stat().st_size > 0:
            status = "already_exists"
        else:
            try:
                _download_file(url, destination, timeout=timeout)
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                status = "failed"
                error = str(exc)

        local_path = str(destination) if status != "failed" else ""
        url_to_local_path[url] = local_path
        records.append(
            AudioDownloadRecord(
                audio_url=url,
                sample_name=str(row["sample_name"]).strip(),
                speaker_id=str(row["speaker_id"]).strip(),
                local_audio_path=local_path,
                status=status,
                error=error,
            )
        )

    updated_metadata = metadata.copy()
    updated_metadata["audio_path"] = updated_metadata["audio_url"].map(lambda value: url_to_local_path.get(str(value).strip(), ""))

    output_metadata = Path(output_metadata_path)
    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    updated_metadata.to_csv(output_metadata, index=False)

    download_index = Path(download_index_path)
    download_index.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(asdict(record) for record in records).to_csv(download_index, index=False)

    failed_count = sum(record.status == "failed" for record in records)
    print(
        f"Downloaded/indexed {len(records) - failed_count}/{len(records)} unique audio files. "
        f"Failed: {failed_count}."
    )
    print(f"Metadata with local audio paths written to {output_metadata}")
    print(f"Audio download index written to {download_index}")
    return output_metadata, download_index

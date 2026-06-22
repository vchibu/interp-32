from __future__ import annotations

from dataclasses import dataclass, asdict
from html import unescape
from pathlib import Path
import csv
import math
import re
import time
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin
from urllib.request import Request, urlopen


BASE_URL = "https://accent.gmu.edu/"
USER_AGENT = "speech-accent-wav2vec2-analysis/0.1"


@dataclass(slots=True)
class SpeakerRecord:
    speaker_id: str
    sample_name: str
    native_language: str
    detail_url: str
    audio_url: str | None = None
    audio_path: str | None = None
    transcript: str | None = None
    birth_place: str | None = None
    other_languages: str | None = None
    age: str | None = None
    sex: str | None = None
    age_of_english_onset: str | None = None
    english_learning_method: str | None = None
    english_residence: str | None = None
    length_of_english_residence: str | None = None


def _fetch_text(url: str, timeout: int = 30) -> str:
    request = urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=timeout)
    try:
        return request.read().decode("utf-8", errors="replace")
    finally:
        request.close()


def _download_file(url: str, destination: Path, timeout: int = 60) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=timeout)
    try:
        destination.write_bytes(response.read())
    finally:
        response.close()
    return destination


def _strip_tags(text: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", text)
    without_entities = unescape(without_tags)
    return re.sub(r"\s+", " ", without_entities).strip()


def discover_languages() -> list[str]:
    page = _fetch_text(urljoin(BASE_URL, "browse_language.php"))
    matches = re.findall(r'browse_language\.php\?function=find&language=([^"]+)"', page)
    languages = [unescape(match).strip() for match in matches]
    seen: set[str] = set()
    ordered = []
    for language in languages:
        if language and language not in seen:
            seen.add(language)
            ordered.append(language)
    return ordered


def list_speakers_for_language(language: str) -> list[tuple[str, str]]:
    language_url = urljoin(BASE_URL, f"browse_language.php?function=find&language={quote(language)}")
    page = _fetch_text(language_url)
    pattern = re.compile(
        r'browse_language\.php\?function=detail&speakerid=(\d+)">([^<]+),</a>',
        flags=re.IGNORECASE,
    )
    return [(speaker_id, sample_name.strip()) for speaker_id, sample_name in pattern.findall(page)]


def _parse_detail_page(detail_url: str, speaker_id: str, sample_name: str, language: str) -> SpeakerRecord:
    page = _fetch_text(detail_url)

    source_match = re.search(r'<source[^>]+src="([^"]+)"', page, flags=re.IGNORECASE)
    transcript_match = re.search(r'<p class="transtext">\s*(.*?)\s*</p>', page, flags=re.IGNORECASE | re.DOTALL)
    bio_block_match = re.search(r'<ul class="bio">(.*?)</ul>', page, flags=re.IGNORECASE | re.DOTALL)

    record = SpeakerRecord(
        speaker_id=speaker_id,
        sample_name=sample_name,
        native_language=language,
        detail_url=detail_url,
        audio_url=urljoin(BASE_URL, source_match.group(1)) if source_match else None,
        transcript=_strip_tags(transcript_match.group(1)) if transcript_match else None,
    )

    if bio_block_match:
        bio_items = re.findall(r"<li>(.*?)</li>", bio_block_match.group(1), flags=re.IGNORECASE | re.DOTALL)
        for item in bio_items:
            clean_item = _strip_tags(item)
            if ":" not in clean_item:
                continue
            key, value = clean_item.split(":", maxsplit=1)
            normalized_key = key.strip().lower()
            normalized_value = value.strip()
            if normalized_key == "birth place":
                record.birth_place = normalized_value
            elif normalized_key == "other language(s)":
                record.other_languages = normalized_value
            elif normalized_key == "age, sex":
                if "," in normalized_value:
                    age, sex = normalized_value.split(",", maxsplit=1)
                    record.age = age.strip()
                    record.sex = sex.strip()
                else:
                    record.age = normalized_value
            elif normalized_key == "age of english onset":
                record.age_of_english_onset = normalized_value
            elif normalized_key == "english learning method":
                record.english_learning_method = normalized_value
            elif normalized_key == "english residence":
                record.english_residence = normalized_value
            elif normalized_key == "length of english residence":
                record.length_of_english_residence = normalized_value

    return record


def scrape_archive(
    raw_dir: str | Path,
    languages: Iterable[str] | None = None,
    download_audio: bool = True,
    max_speakers: int | None = None,
    delay_seconds: float = 0.2,
) -> Path:
    """Scrape metadata and optionally audio files from the public archive."""
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    audio_dir = raw_path / "audio"
    metadata_path = raw_path / "accent_archive_metadata.csv"

    target_languages = list(languages) if languages is not None else discover_languages()
    total_languages = len(target_languages)
    print(f"Starting archive scrape for {total_languages} language groups.")
    if download_audio:
        print("Audio download is enabled.")
    else:
        print("Metadata-only mode is enabled; audio files will not be downloaded.")

    rows: list[SpeakerRecord] = []
    scraped = 0
    for language_index, language in enumerate(target_languages, start=1):
        speakers = list_speakers_for_language(language)
        print(
            f"[{language_index}/{total_languages}] Language '{language}': "
            f"{len(speakers)} speaker pages queued."
        )
        for speaker_id, sample_name in speakers:
            detail_url = urljoin(BASE_URL, f"browse_language.php?function=detail&speakerid={speaker_id}")
            try:
                record = _parse_detail_page(detail_url, speaker_id, sample_name, language)
                if download_audio and record.audio_url:
                    audio_extension = Path(record.audio_url).suffix or ".mp3"
                    audio_filename = f"{record.sample_name}{audio_extension}"
                    local_audio_path = audio_dir / audio_filename
                    if not local_audio_path.exists():
                        _download_file(record.audio_url, local_audio_path)
                    record.audio_path = str(local_audio_path)
                rows.append(record)
                scraped += 1
                if scraped <= 5 or scraped % 50 == 0:
                    print(
                        f"  Processed {scraped} speaker pages so far. "
                        f"Latest: {record.sample_name} ({record.native_language})"
                    )
            except (HTTPError, URLError, TimeoutError):
                print(f"  Skipped speaker {speaker_id} ({sample_name}) because the page fetch failed.")
                continue

            if delay_seconds > 0:
                time.sleep(delay_seconds)
            if max_speakers is not None and scraped >= max_speakers:
                break
        if max_speakers is not None and scraped >= max_speakers:
            break

    print(f"Finished scraping {scraped} speaker pages.")

    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()) if rows else list(SpeakerRecord.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    print(f"Metadata written to {metadata_path}")
    return metadata_path

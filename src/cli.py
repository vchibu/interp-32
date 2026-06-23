from __future__ import annotations

import argparse
import os
from pathlib import Path
import re

import pandas as pd

from src.analysis.accent_country_breakdown import save_breakdown, summarize_accent_groups_by_country
from src.analysis.dataset_summary import save_summary, summarize_top_accent_groups
from src.analysis.dimension_stats import rank_dimensions_by_mi
from src.analysis.layerwise_probes import run_probe_analysis
from src.config import load_config
from src.data.build_archive_manifest import build_archive_manifest
from src.data.build_manifest import build_manifest, save_manifest
from src.data.download_archive_audio import download_audio_from_metadata
from src.data.normalize_audio import normalize_from_manifest
from src.data.scrape_accent_archive import scrape_archive
from src.extract.extract_hidden_states import extract_from_manifest
from src.models.load_wav2vec2 import load_wav2vec2


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def _slugify_model_name(model_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", model_name.strip().lower())
    return re.sub(r"_+", "_", slug).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Speech Accent Archive starter CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser("build-manifest", help="Build canonical manifest")
    manifest_parser.add_argument("--config", default="configs/base.yaml")

    scrape_parser = subparsers.add_parser("scrape-archive", help="Scrape the public archive into data/raw")
    scrape_parser.add_argument("--config", default="configs/base.yaml")
    scrape_parser.add_argument("--languages", nargs="*", default=None, help="Optional subset of native languages to scrape")
    scrape_parser.add_argument("--max-speakers", type=int, default=None, help="Optional cap for validation runs")
    scrape_parser.add_argument("--metadata-only", action="store_true", help="Scrape metadata without downloading audio")

    download_parser = subparsers.add_parser(
        "download-archive-audio",
        help="Download every unique archive audio URL from the raw metadata with no filtering or caps",
    )
    download_parser.add_argument("--config", default="configs/base.yaml")
    download_parser.add_argument("--metadata-path", default=None)
    download_parser.add_argument("--audio-dir", default=None)
    download_parser.add_argument("--output-metadata-path", default=None)
    download_parser.add_argument("--download-index-path", default=None)
    download_parser.add_argument("--timeout", type=int, default=90)

    normalize_parser = subparsers.add_parser("normalize-audio", help="Normalize audio from manifest")
    normalize_parser.add_argument("--config", default="configs/base.yaml")
    normalize_parser.add_argument("--overwrite", action="store_true", help="Renormalize WAV files even when they already exist")
    normalize_parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Parallel normalization workers; defaults to SLURM_CPUS_PER_TASK or 1",
    )

    extract_parser = subparsers.add_parser("extract-hidden-states", help="Extract wav2vec2 hidden states")
    extract_parser.add_argument("--config", default="configs/base.yaml")
    extract_parser.add_argument("--model-name", default=None, help="Hugging Face model name; defaults to config model.model_name")
    extract_parser.add_argument("--feature-dir", default=None, help="Output directory for .npz feature files")
    extract_parser.add_argument("--feature-index-path", default=None, help="Output CSV path for the feature index")
    extract_parser.add_argument("--overwrite", action="store_true", help="Recompute features even when .npz files already exist")

    probes_parser = subparsers.add_parser("run-probes", help="Run layer-wise linear probes")
    probes_parser.add_argument("--config", default="configs/base.yaml")
    probes_parser.add_argument("--model-name", default=None, help="Model name used to choose the model-specific feature index")
    probes_parser.add_argument("--feature-index-path", default=None, help="Feature index CSV to analyze")
    probes_parser.add_argument("--analysis-dir", default=None, help="Directory for probe outputs")
    probes_parser.add_argument("--layer", type=int, default=0, help="Layer to use for MI ranking output")

    summary_parser = subparsers.add_parser("summarize-accent-groups", help="Summarize largest accent groups from raw metadata")
    summary_parser.add_argument("--config", default="configs/base.yaml")
    summary_parser.add_argument("--top-n", type=int, default=20)
    summary_parser.add_argument("--min-total-speakers", type=int, default=None, help="Only keep groups with at least this many total speakers")

    country_parser = subparsers.add_parser("summarize-accent-countries", help="Summarize accent groups by country of origin from the manifest")
    country_parser.add_argument("--config", default="configs/base.yaml")

    args = parser.parse_args()
    config = load_config(args.config)
    manifest_dir = Path(config["paths"]["manifest_dir"])

    if args.command == "scrape-archive":
        metadata_path = scrape_archive(
            raw_dir=config["paths"]["raw_dir"],
            languages=args.languages,
            download_audio=not args.metadata_only,
            max_speakers=args.max_speakers,
        )
        print(f"Archive metadata written to {metadata_path}")
        return

    if args.command == "build-manifest":
        output_path = manifest_dir / "manifest.csv"
        raw_dir = Path(config["paths"]["raw_dir"])
        enriched_metadata_path = raw_dir / "accent_archive_metadata_with_audio.csv"
        if enriched_metadata_path.exists():
            df = build_archive_manifest(
                metadata_path=enriched_metadata_path,
                audio_dir=raw_dir / "audio",
                default_transcript=config["dataset"].get("transcript_text"),
            )
        else:
            df = build_manifest(
                config["paths"]["raw_dir"],
                default_transcript=config["dataset"].get("transcript_text"),
            )
        save_manifest(df, output_path)
        print(f"Manifest written to {output_path}")
        return

    if args.command == "download-archive-audio":
        raw_dir = Path(config["paths"]["raw_dir"])
        metadata_path = Path(args.metadata_path) if args.metadata_path else raw_dir / "accent_archive_metadata.csv"
        audio_dir = Path(args.audio_dir) if args.audio_dir else raw_dir / "audio"
        output_metadata_path = (
            Path(args.output_metadata_path)
            if args.output_metadata_path
            else raw_dir / "accent_archive_metadata_with_audio.csv"
        )
        download_index_path = (
            Path(args.download_index_path)
            if args.download_index_path
            else manifest_dir / "archive_audio_download_index.csv"
        )
        download_audio_from_metadata(
            metadata_path=metadata_path,
            audio_dir=audio_dir,
            output_metadata_path=output_metadata_path,
            download_index_path=download_index_path,
            timeout=args.timeout,
        )
        return

    if args.command == "normalize-audio":
        input_path = manifest_dir / "manifest.csv"
        output_path = manifest_dir / "manifest_normalized.csv"
        manifest = _read_csv(input_path)
        normalized = normalize_from_manifest(
            manifest=manifest,
            processed_audio_dir=config["paths"]["processed_audio_dir"],
            sample_rate=config["audio"]["target_sample_rate"],
            overwrite=args.overwrite,
            num_workers=args.num_workers or int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        )
        save_manifest(normalized, output_path)
        print(f"Normalized manifest written to {output_path}")
        return

    if args.command == "extract-hidden-states":
        input_path = manifest_dir / "manifest_normalized.csv"
        model_name = args.model_name or config["model"]["model_name"]
        model_slug = _slugify_model_name(model_name)
        output_path = (
            Path(args.feature_index_path)
            if args.feature_index_path
            else manifest_dir / f"feature_index_{model_slug}.csv"
        )
        feature_dir = (
            Path(args.feature_dir)
            if args.feature_dir
            else Path(config["paths"]["feature_dir"]) / model_slug
        )
        manifest = _read_csv(input_path)
        bundle = load_wav2vec2(
            model_name=model_name,
            device_name=config["model"]["device"],
        )
        feature_index = extract_from_manifest(
            manifest=manifest,
            bundle=bundle,
            output_dir=feature_dir,
            sample_rate=config["audio"]["target_sample_rate"],
            save_frame_level=config["model"]["save_frame_level"],
            overwrite=args.overwrite,
        )
        feature_index["model_slug"] = model_slug
        save_manifest(feature_index, output_path)
        print(f"Feature index written to {output_path}")
        return

    if args.command == "run-probes":
        model_name = args.model_name or config["model"]["model_name"]
        model_slug = _slugify_model_name(model_name)
        input_path = (
            Path(args.feature_index_path)
            if args.feature_index_path
            else manifest_dir / f"feature_index_{model_slug}.csv"
        )
        analysis_dir = Path(args.analysis_dir) if args.analysis_dir else Path(config["paths"]["analysis_dir"]) / model_slug
        analysis_dir.mkdir(parents=True, exist_ok=True)
        feature_index = _read_csv(input_path)

        probe_results = run_probe_analysis(
            feature_index=feature_index,
            label_field=config["dataset"]["label_field"],
            test_size=config["analysis"]["probe_test_size"],
            random_state=config["analysis"]["random_seed"],
        )
        probe_path = analysis_dir / "layerwise_probe_results.csv"
        probe_results.to_csv(probe_path, index=False)

        mi_results = rank_dimensions_by_mi(
            feature_index=feature_index,
            label_field=config["dataset"]["label_field"],
            layer_idx=args.layer,
        )
        mi_path = analysis_dir / f"layer_{args.layer}_dimension_mi.csv"
        mi_results.to_csv(mi_path, index=False)

        print(f"Probe results written to {probe_path}")
        print(f"Dimension ranking written to {mi_path}")
        return

    if args.command == "summarize-accent-groups":
        metadata_path = Path(config["paths"]["raw_dir"]) / "accent_archive_metadata.csv"
        analysis_dir = Path(config["paths"]["analysis_dir"])
        analysis_dir.mkdir(parents=True, exist_ok=True)
        metadata = _read_csv(metadata_path)
        summary = summarize_top_accent_groups(
            metadata=metadata,
            top_n=args.top_n,
            min_total_speakers=args.min_total_speakers,
        )
        if args.min_total_speakers is not None:
            output_name = f"accent_groups_min_{args.min_total_speakers}.csv"
        else:
            output_name = f"top_{args.top_n}_accent_groups.csv"
        output_path = analysis_dir / output_name
        save_summary(summary, output_path)
        print(summary.to_string(index=False))
        print(f"Summary written to {output_path}")

    if args.command == "summarize-accent-countries":
        input_path = Path(config["paths"]["raw_dir"]) / "accent_archive_metadata.csv"
        accent_groups_path = Path(config["paths"]["analysis_dir"]) / "accent_groups_min_10.csv"
        output_path = Path(config["paths"]["analysis_dir"]) / "accent_group_country_breakdown.csv"
        metadata = _read_csv(input_path)
        accent_groups = _read_csv(accent_groups_path)["accent_group"].dropna().astype(str).str.strip().tolist()
        breakdown = summarize_accent_groups_by_country(metadata=metadata, allowed_groups=accent_groups)
        save_breakdown(breakdown, output_path)
        print(breakdown.to_string(index=False))
        print(f"Country breakdown written to {output_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis.accent_country_breakdown import save_breakdown, summarize_accent_groups_by_country
from src.analysis.dataset_summary import save_summary, summarize_top_accent_groups
from src.analysis.dimension_stats import rank_dimensions_by_mi
from src.analysis.layerwise_probes import run_probe_analysis
from src.config import load_config
from src.data.build_manifest import build_manifest, save_manifest
from src.data.normalize_audio import normalize_from_manifest
from src.data.scrape_accent_archive import scrape_archive
from src.extract.extract_hidden_states import extract_from_manifest
from src.models.load_wav2vec2 import load_wav2vec2


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


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

    normalize_parser = subparsers.add_parser("normalize-audio", help="Normalize audio from manifest")
    normalize_parser.add_argument("--config", default="configs/base.yaml")

    extract_parser = subparsers.add_parser("extract-hidden-states", help="Extract wav2vec2 hidden states")
    extract_parser.add_argument("--config", default="configs/base.yaml")

    probes_parser = subparsers.add_parser("run-probes", help="Run layer-wise linear probes")
    probes_parser.add_argument("--config", default="configs/base.yaml")
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
        df = build_manifest(
            config["paths"]["raw_dir"],
            default_transcript=config["dataset"].get("transcript_text"),
        )
        save_manifest(df, output_path)
        print(f"Manifest written to {output_path}")
        return

    if args.command == "normalize-audio":
        input_path = manifest_dir / "manifest.csv"
        output_path = manifest_dir / "manifest_normalized.csv"
        manifest = _read_csv(input_path)
        normalized = normalize_from_manifest(
            manifest=manifest,
            processed_audio_dir=config["paths"]["processed_audio_dir"],
            sample_rate=config["audio"]["target_sample_rate"],
        )
        save_manifest(normalized, output_path)
        print(f"Normalized manifest written to {output_path}")
        return

    if args.command == "extract-hidden-states":
        input_path = manifest_dir / "manifest_normalized.csv"
        output_path = manifest_dir / "feature_index.csv"
        manifest = _read_csv(input_path)
        bundle = load_wav2vec2(
            model_name=config["model"]["model_name"],
            device_name=config["model"]["device"],
        )
        feature_index = extract_from_manifest(
            manifest=manifest,
            bundle=bundle,
            output_dir=config["paths"]["feature_dir"],
            sample_rate=config["audio"]["target_sample_rate"],
            save_frame_level=config["model"]["save_frame_level"],
        )
        save_manifest(feature_index, output_path)
        print(f"Feature index written to {output_path}")
        return

    if args.command == "run-probes":
        input_path = manifest_dir / "feature_index.csv"
        analysis_dir = Path(config["paths"]["analysis_dir"])
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

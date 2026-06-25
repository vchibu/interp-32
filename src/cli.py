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
from src.analysis.probe_group_summary import (
    DEFAULT_ENGLISH_SPEAKING_COUNTRIES,
    add_accent_country_probe_filter,
    balanced_sample_by_label_and_sex,
    keep_groups_with_min_balanced_sex_pairs,
    save_probe_group_summary,
    summarize_accent_country_probe_groups,
)
from src.config import load_config
from src.data.build_archive_manifest import build_archive_manifest
from src.data.build_manifest import build_manifest, save_manifest
from src.data.download_archive_audio import download_audio_from_metadata
from src.data.normalize_audio import normalize_from_manifest
from src.data.scrape_accent_archive import scrape_archive
from src.extract.extract_hidden_states import extract_from_manifest
from src.models.load_wav2vec2 import load_speech_model


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
    probes_parser.add_argument("--label-field", default=None, help="Metadata column to predict")
    probes_parser.add_argument("--min-examples-per-label", type=int, default=None)
    probes_parser.add_argument("--top-n-labels", type=int, default=None, help="Keep only the largest N eligible labels")
    probes_parser.add_argument("--exclude-labels", nargs="*", default=None, help="Labels to exclude before probing")
    probes_parser.add_argument("--n-splits", type=int, default=None, help="Repeated stratified train/test splits")
    probes_parser.add_argument("--test-size", type=float, default=None)
    probes_parser.add_argument("--random-seed", type=int, default=None)
    probes_parser.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    probes_parser.add_argument("--max-iter", type=int, default=2000)
    probes_parser.add_argument(
        "--accent-country-filter",
        action="store_true",
        help="Use first_language + country_of_origin labels and exclude >2y English residence outside English-speaking birth countries",
    )
    probes_parser.add_argument("--max-non-english-country-residence-years", type=float, default=2.0)
    probes_parser.add_argument(
        "--english-speaking-countries",
        nargs="*",
        default=list(DEFAULT_ENGLISH_SPEAKING_COUNTRIES),
    )
    probes_parser.add_argument(
        "--min-balanced-sex-pairs",
        type=int,
        default=None,
        help="Only keep labels with at least this many female/male pairs; requires --accent-country-filter",
    )
    probes_parser.add_argument(
        "--balanced-sex-sample-size",
        type=int,
        default=None,
        help="Sample exactly this many female and male speakers per label; requires --accent-country-filter",
    )

    summary_parser = subparsers.add_parser("summarize-accent-groups", help="Summarize largest accent groups from raw metadata")
    summary_parser.add_argument("--config", default="configs/base.yaml")
    summary_parser.add_argument("--top-n", type=int, default=20)
    summary_parser.add_argument("--min-total-speakers", type=int, default=None, help="Only keep groups with at least this many total speakers")

    country_parser = subparsers.add_parser("summarize-accent-countries", help="Summarize accent groups by country of origin from the manifest")
    country_parser.add_argument("--config", default="configs/base.yaml")

    probe_groups_parser = subparsers.add_parser(
        "summarize-probe-groups",
        help="Summarize accent-country probe candidates after English-residence filtering",
    )
    probe_groups_parser.add_argument("--config", default="configs/base.yaml")
    probe_groups_parser.add_argument("--manifest-path", default=None)
    probe_groups_parser.add_argument("--output-path", default=None)
    probe_groups_parser.add_argument("--filtered-output-path", default=None)
    probe_groups_parser.add_argument("--accent-field", default="first_language")
    probe_groups_parser.add_argument("--country-field", default="country_of_origin")
    probe_groups_parser.add_argument("--sex-field", default="sex")
    probe_groups_parser.add_argument("--english-residence-years-field", default="length_of_english_residence")
    probe_groups_parser.add_argument("--max-non-english-country-residence-years", type=float, default=2.0)
    probe_groups_parser.add_argument(
        "--english-speaking-countries",
        nargs="*",
        default=list(DEFAULT_ENGLISH_SPEAKING_COUNTRIES),
    )
    probe_groups_parser.add_argument(
        "--min-balanced-sex-pairs",
        type=int,
        default=None,
        help="Print only groups with at least this many balanced female/male pairs",
    )

    plot_parser = subparsers.add_parser("plot-probe-comparison", help="Plot layer-wise probe curves for multiple models")
    plot_parser.add_argument("--config", default="configs/base.yaml")
    plot_parser.add_argument("--metric", default="macro_f1")
    plot_parser.add_argument("--output-path", default=None)
    plot_parser.add_argument("--summary", nargs=2, action="append", metavar=("LABEL", "CSV_PATH"), required=True)

    cluster_parser = subparsers.add_parser(
        "cluster-centroids",
        help="Cluster accent-country group centroids in hidden-state embedding space",
    )
    cluster_parser.add_argument("--config", default="configs/base.yaml")
    cluster_parser.add_argument("--feature-index-path", required=True)
    cluster_parser.add_argument("--probe-results-path", default=None)
    cluster_parser.add_argument("--output-dir", required=True)
    cluster_parser.add_argument("--label-field", default="accent_country_label")
    cluster_parser.add_argument("--layer", default="best", help="Layer index or 'best' from probe results")
    cluster_parser.add_argument("--best-layer-metric", default="macro_f1_mean")
    cluster_parser.add_argument("--distance-metric", default="cosine")
    cluster_parser.add_argument("--linkage-method", default="average")

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
        bundle = load_speech_model(
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
        balanced_pair_counts = None
        balanced_sample_counts = None
        if args.accent_country_filter:
            feature_index = add_accent_country_probe_filter(
                manifest=feature_index,
                max_non_english_country_residence_years=args.max_non_english_country_residence_years,
                english_speaking_countries=args.english_speaking_countries,
            )
            feature_index = feature_index[feature_index["eligible_for_accent_country_probe"]].copy()
            if args.min_balanced_sex_pairs is not None:
                feature_index, balanced_pair_counts = keep_groups_with_min_balanced_sex_pairs(
                    manifest=feature_index,
                    label_field="accent_country_label",
                    sex_field="probe_sex",
                    min_balanced_sex_pairs=args.min_balanced_sex_pairs,
                )
            if args.balanced_sex_sample_size is not None:
                feature_index, balanced_sample_counts = balanced_sample_by_label_and_sex(
                    manifest=feature_index,
                    label_field="accent_country_label",
                    sex_field="probe_sex",
                    n_per_sex=args.balanced_sex_sample_size,
                    random_state=args.random_seed if args.random_seed is not None else config["analysis"]["random_seed"],
                )
        elif args.min_balanced_sex_pairs is not None:
            raise ValueError("--min-balanced-sex-pairs requires --accent-country-filter")
        elif args.balanced_sex_sample_size is not None:
            raise ValueError("--balanced-sex-sample-size requires --accent-country-filter")
        label_field = args.label_field or ("accent_country_label" if args.accent_country_filter else config["dataset"]["label_field"])
        min_examples_per_label = (
            args.min_examples_per_label
            if args.min_examples_per_label is not None
            else int(config["dataset"].get("min_examples_per_label", 20))
        )
        test_size = args.test_size if args.test_size is not None else config["analysis"]["probe_test_size"]
        random_seed = args.random_seed if args.random_seed is not None else config["analysis"]["random_seed"]
        n_splits = args.n_splits if args.n_splits is not None else int(config["analysis"].get("probe_n_splits", 5))
        class_weight = None if args.class_weight == "none" else args.class_weight

        probe_result = run_probe_analysis(
            feature_index=feature_index,
            label_field=label_field,
            min_examples_per_label=min_examples_per_label,
            top_n_labels=args.top_n_labels,
            exclude_labels=args.exclude_labels,
            test_size=test_size,
            random_state=random_seed,
            n_splits=n_splits,
            class_weight=class_weight,
            max_iter=args.max_iter,
        )
        probe_path = analysis_dir / "layerwise_probe_results.csv"
        split_path = analysis_dir / "layerwise_probe_split_results.csv"
        class_counts_path = analysis_dir / "probe_class_counts.csv"
        balanced_pair_counts_path = analysis_dir / "probe_balanced_pair_counts.csv"
        balanced_sample_counts_path = analysis_dir / "probe_balanced_sample_counts.csv"
        filtered_index_path = analysis_dir / "probe_feature_index_filtered.csv"
        confusion_path = analysis_dir / "best_layer_confusion_matrix.csv"
        plot_path = analysis_dir / "layerwise_probe_macro_f1.png"
        probe_result.layer_summary.to_csv(probe_path, index=False)
        probe_result.split_results.to_csv(split_path, index=False)
        probe_result.class_counts.to_csv(class_counts_path, index=False)
        if balanced_pair_counts is not None:
            balanced_pair_counts.to_csv(balanced_pair_counts_path, index=False)
        if balanced_sample_counts is not None:
            balanced_sample_counts.to_csv(balanced_sample_counts_path, index=False)
        probe_result.filtered_feature_index.to_csv(filtered_index_path, index=False)
        probe_result.best_layer_confusion_matrix.to_csv(confusion_path, index=False)
        from src.analysis.plot_probe_results import plot_layerwise_probe_summary

        plot_layerwise_probe_summary(
            summary=probe_result.layer_summary,
            output_path=plot_path,
            metric="macro_f1",
            title=f"{model_name}: {label_field}",
        )

        mi_results = rank_dimensions_by_mi(
            feature_index=probe_result.filtered_feature_index,
            label_field=label_field,
            layer_idx=args.layer,
        )
        mi_path = analysis_dir / f"layer_{args.layer}_dimension_mi.csv"
        mi_results.to_csv(mi_path, index=False)

        best_row = probe_result.layer_summary[
            (probe_result.layer_summary["model"] == "linear_probe")
            & (probe_result.layer_summary["layer"] == probe_result.best_layer)
        ].iloc[0]
        print(f"Probe label field: {label_field}")
        print(
            "Probe examples: "
            f"{len(probe_result.filtered_feature_index)} across "
            f"{probe_result.filtered_feature_index[label_field].nunique()} labels"
        )
        print(
            "Best layer by macro F1: "
            f"{probe_result.best_layer} "
            f"(macro_f1={best_row['macro_f1_mean']:.4f}, "
            f"balanced_accuracy={best_row['balanced_accuracy_mean']:.4f})"
        )
        print(f"Probe summary written to {probe_path}")
        print(f"Probe split metrics written to {split_path}")
        print(f"Probe class counts written to {class_counts_path}")
        if balanced_pair_counts is not None:
            print(f"Balanced pair counts written to {balanced_pair_counts_path}")
        if balanced_sample_counts is not None:
            print(f"Balanced sample counts written to {balanced_sample_counts_path}")
        print(f"Best-layer confusion matrix written to {confusion_path}")
        print(f"Layer plot written to {plot_path}")
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
        return

    if args.command == "summarize-probe-groups":
        input_path = Path(args.manifest_path) if args.manifest_path else manifest_dir / "manifest_normalized.csv"
        analysis_dir = Path(config["paths"]["analysis_dir"])
        output_path = (
            Path(args.output_path)
            if args.output_path
            else analysis_dir / "accent_country_probe_candidates.csv"
        )
        filtered_output_path = (
            Path(args.filtered_output_path)
            if args.filtered_output_path
            else analysis_dir / "accent_country_probe_manifest_with_filter.csv"
        )
        manifest = _read_csv(input_path)
        summary, prepared = summarize_accent_country_probe_groups(
            manifest=manifest,
            accent_field=args.accent_field,
            country_field=args.country_field,
            sex_field=args.sex_field,
            english_residence_years_field=args.english_residence_years_field,
            english_speaking_countries=args.english_speaking_countries,
            max_non_english_country_residence_years=args.max_non_english_country_residence_years,
        )
        save_probe_group_summary(summary, output_path)
        save_probe_group_summary(prepared, filtered_output_path)
        display_summary = summary
        if args.min_balanced_sex_pairs is not None:
            display_summary = summary[
                summary["n_balanced_female_male_pairs"] >= args.min_balanced_sex_pairs
            ].copy()
        n_eligible = int(prepared["eligible_for_accent_country_probe"].sum())
        n_excluded_residence = int(prepared["excluded_for_english_residence"].sum())
        print(display_summary.head(30).to_string(index=False))
        print(f"Eligible recordings after filter: {n_eligible}")
        if args.min_balanced_sex_pairs is not None:
            print(f"Groups with at least {args.min_balanced_sex_pairs} balanced female/male pairs: {len(display_summary)}")
        print(f"Excluded for >{args.max_non_english_country_residence_years:g} years English residence outside English-speaking birth countries: {n_excluded_residence}")
        print(f"Probe group summary written to {output_path}")
        print(f"Manifest with probe filter columns written to {filtered_output_path}")
        return

    if args.command == "plot-probe-comparison":
        output_path = (
            Path(args.output_path)
            if args.output_path
            else Path(config["paths"]["analysis_dir"]) / "probe_model_comparison_macro_f1.png"
        )
        summaries = [(label, _read_csv(Path(path))) for label, path in args.summary]
        from src.analysis.plot_probe_results import plot_probe_comparison

        plot_probe_comparison(
            summaries=summaries,
            output_path=output_path,
            metric=args.metric,
            title=f"Layer-wise probe comparison ({args.metric})",
        )
        print(f"Probe comparison plot written to {output_path}")
        return

    if args.command == "cluster-centroids":
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        feature_index = _read_csv(Path(args.feature_index_path))

        from src.analysis.centroid_clustering import (
            best_probe_layer,
            plot_centroid_pca,
            plot_dendrogram,
            plot_distance_heatmap,
            run_centroid_clustering,
        )

        if str(args.layer).lower() == "best":
            if args.probe_results_path is None:
                raise ValueError("--probe-results-path is required when --layer best")
            probe_results = _read_csv(Path(args.probe_results_path))
            layer = best_probe_layer(probe_results, metric=args.best_layer_metric)
        else:
            layer = int(args.layer)

        result = run_centroid_clustering(
            feature_index=feature_index,
            label_field=args.label_field,
            layer=layer,
            distance_metric=args.distance_metric,
            linkage_method=args.linkage_method,
        )
        result.centroids.to_csv(output_dir / f"layer_{layer}_centroids.csv", index=False)
        result.group_counts.to_csv(output_dir / f"layer_{layer}_group_counts.csv", index=False)
        result.distance_matrix.to_csv(output_dir / f"layer_{layer}_centroid_distances.csv")
        result.distance_long.to_csv(output_dir / f"layer_{layer}_centroid_distances_long.csv", index=False)
        result.cluster_order.to_csv(output_dir / f"layer_{layer}_cluster_order.csv", index=False)
        heatmap_path = plot_distance_heatmap(
            distance_matrix=result.distance_matrix,
            output_path=output_dir / f"layer_{layer}_centroid_distance_heatmap.png",
            linkage_method=args.linkage_method,
            title=f"Centroid {args.distance_metric} distance, layer {layer}",
        )
        dendrogram_path = plot_dendrogram(
            distance_matrix=result.distance_matrix,
            output_path=output_dir / f"layer_{layer}_centroid_dendrogram.png",
            linkage_method=args.linkage_method,
            title=f"Centroid clustering, layer {layer}",
        )
        pca_path = plot_centroid_pca(
            centroids=result.centroids,
            output_path=output_dir / f"layer_{layer}_centroid_pca.png",
            title=f"Centroid PCA, layer {layer}",
        )

        print(f"Centroid clustering layer: {layer}")
        print(f"Groups clustered: {len(result.group_counts)}")
        print("Closest centroid pairs:")
        print(result.distance_long.head(10).to_string(index=False))
        print(f"Centroids written to {output_dir / f'layer_{layer}_centroids.csv'}")
        print(f"Distance matrix written to {output_dir / f'layer_{layer}_centroid_distances.csv'}")
        print(f"Distance long table written to {output_dir / f'layer_{layer}_centroid_distances_long.csv'}")
        print(f"Cluster order written to {output_dir / f'layer_{layer}_cluster_order.csv'}")
        print(f"Heatmap written to {heatmap_path}")
        print(f"Dendrogram written to {dendrogram_path}")
        print(f"PCA plot written to {pca_path}")
        return


if __name__ == "__main__":
    main()

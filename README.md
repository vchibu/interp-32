# Speech Accent Archive + Wav2Vec2 Analysis

Starter scaffold for preparing the Speech Accent Archive dataset, extracting hidden states from `wav2vec2`, and analyzing internal dimensions layer by layer.

## Intended workflow

1. Scrape or provide the full Speech Accent Archive metadata in `data/raw/`.
2. Download every unique archive audio file with no label filters or speaker caps.
3. Build a canonical manifest with one row per unique recording, local audio paths, and speaker metadata.
4. Normalize audio to mono 16 kHz WAV in `data/processed/audio/`.
5. Run frozen `wav2vec2` extraction and save pooled or frame-level hidden states.
6. Analyze layer-wise separability and hidden dimensions.

## CLI

The starter CLI now supports:

- `python -m src.cli scrape-archive`
- `python -m src.cli download-archive-audio`
- `python -m src.cli build-manifest`
- `python -m src.cli normalize-audio`
- `python -m src.cli extract-hidden-states`
- `python -m src.cli run-probes`
- `python -m src.cli summarize-accent-groups`

Example full-dataset bootstrap flow:

1. `python -m src.cli scrape-archive --metadata-only`
2. `python -m src.cli download-archive-audio`
3. `python -m src.cli build-manifest`
4. `python -m src.cli normalize-audio`
5. `python -m src.cli extract-hidden-states --model-name facebook/wav2vec2-base`
6. `python -m src.cli run-probes --model-name facebook/wav2vec2-base`

If `data/raw/accent_archive_metadata.csv` already exists, start at step 2. On the cluster, submit `jobs/download_full_archive_audio.sh` to download all audio from the existing metadata table.

The full-audio downloader writes:

- `data/raw/audio/`: downloaded MP3 files, one per unique `audio_url`
- `data/raw/accent_archive_metadata_with_audio.csv`: complete metadata with local `audio_path` filled in
- `data/manifests/archive_audio_download_index.csv`: status for each unique audio URL

The manifest builder writes `data/manifests/manifest.csv`. This is the table used by the rest of the pipeline: one row per unique recording, local `audio_path`, transcript, country, sex, age, English-residence fields, and language labels. The original metadata is kept complete, while duplicate language listings for the same recording are aggregated in `all_native_languages`.

On Snellius, run the pipeline as Slurm jobs. The model extraction step uses an H100 GPU:

```bash
download_job=$(sbatch --parsable jobs/download_full_archive_audio.sh)
manifest_job=$(sbatch --parsable --dependency=afterok:$download_job jobs/build_manifest.sh)
normalize_job=$(sbatch --parsable --dependency=afterok:$manifest_job jobs/normalize_audio.sh)
feature_job=$(sbatch --parsable --dependency=afterok:$normalize_job jobs/extract_hidden_states_base_h100.sh)
```

If a normalization or feature extraction job stops midway, resubmit the same job. Existing WAV and `.npz` files are reused unless `--overwrite` is passed.

Feature caches are stored by model checkpoint. For example:

- `outputs/features/facebook_wav2vec2_base/`
- `outputs/features/facebook_wav2vec2_base_960h/`
- `data/manifests/feature_index_facebook_wav2vec2_base.csv`
- `data/manifests/feature_index_facebook_wav2vec2_base_960h.csv`

Probe outputs are also grouped by model:

- `outputs/analysis/facebook_wav2vec2_base/`
- `outputs/analysis/facebook_wav2vec2_base_960h/`

To extract the English-ASR-fine-tuned checkpoint for comparison, submit:

```bash
sbatch jobs/extract_hidden_states_base_960h_h100.sh
```

To run the current probe analysis for both cached models, submit:

```bash
sbatch jobs/run_probes_base.sh
sbatch jobs/run_probes_base_960h.sh
```

By default, `run-probes` predicts `dataset.label_field` from `configs/base.yaml`, keeps labels with at least `dataset.min_examples_per_label` examples, and runs five repeated stratified train/test splits. It writes:

- `layerwise_probe_results.csv`: mean and standard deviation by layer for accuracy, balanced accuracy, macro F1, and weighted F1
- `layerwise_probe_split_results.csv`: one row per layer and split, including majority and stratified dummy baselines
- `probe_class_counts.csv`: label counts and whether each label was included
- `probe_feature_index_filtered.csv`: exact recordings used for the probe
- `best_layer_confusion_matrix.csv`: summed confusion matrix across splits for the best layer by macro F1
- `layer_<N>_dimension_mi.csv`: mutual-information ranking of dimensions for the requested layer

Useful probe options:

```bash
python -m src.cli run-probes \
  --model-name facebook/wav2vec2-base-960h \
  --label-field native_language \
  --min-examples-per-label 20 \
  --top-n-labels 20
```

To inspect accent-country probe candidates before choosing a minimum group size, run:

```bash
python -m src.cli summarize-probe-groups
```

This writes `outputs/analysis/accent_country_probe_candidates.csv` with speaker counts, female/male counts, and balanced female/male pairs for each `native_language` + `country_of_origin` group. It also writes `outputs/analysis/accent_country_probe_manifest_with_filter.csv`, which marks which recordings remain after excluding speakers born outside the configured English-speaking countries with more than two years of English residence.

To run probes directly on those accent-country labels:

```bash
python -m src.cli run-probes \
  --model-name facebook/wav2vec2-base-960h \
  --accent-country-filter \
  --min-examples-per-label 20
```

For the main accent-country experiment with all speakers from groups that have at least 15 female and 15 male speakers, submit:

```bash
base_job=$(sbatch --parsable jobs/run_probes_accent_country_min15_pairs_base.sh)
finetuned_job=$(sbatch --parsable jobs/run_probes_accent_country_min15_pairs_base_960h.sh)
sbatch --dependency=afterok:$base_job:$finetuned_job jobs/plot_probe_comparison_accent_country_min15_pairs.sh
```

Each job writes a layer-wise plot to its analysis directory:

- `outputs/analysis/accent_country_min15_pairs/facebook_wav2vec2_base/layerwise_probe_macro_f1.png`
- `outputs/analysis/accent_country_min15_pairs/facebook_wav2vec2_base_960h/layerwise_probe_macro_f1.png`

The dependency job makes a single model-comparison plot. To run that plot manually instead:

```bash
python -m src.cli plot-probe-comparison \
  --output-path outputs/analysis/accent_country_min15_pairs/model_comparison_macro_f1.png \
  --summary base outputs/analysis/accent_country_min15_pairs/facebook_wav2vec2_base/layerwise_probe_results.csv \
  --summary base_960h outputs/analysis/accent_country_min15_pairs/facebook_wav2vec2_base_960h/layerwise_probe_results.csv
```

To cluster accent-country centroids at each model's best probe layer, submit:

```bash
sbatch jobs/cluster_centroids_accent_country_min15_pairs_base.sh
sbatch jobs/cluster_centroids_accent_country_min15_pairs_base_960h.sh
```

Each clustering job uses `probe_feature_index_filtered.csv`, reads the best layer from `layerwise_probe_results.csv`, and writes:

- `layer_<N>_centroids.csv`
- `layer_<N>_group_counts.csv`
- `layer_<N>_centroid_distances.csv`
- `layer_<N>_centroid_distances_long.csv`
- `layer_<N>_cluster_order.csv`
- `layer_<N>_centroid_distance_heatmap.png`
- `layer_<N>_centroid_dendrogram.png`
- `layer_<N>_centroid_pca.png`

Dataset summary example:

`python -m src.cli summarize-accent-groups --top-n 20`

## Project layout

- `configs/`: experiment and path configuration
- `data/`: raw data, processed audio, manifests
- `docs/`: dataset and experiment design notes
- `notebooks/`: exploratory analysis notebooks
- `src/data/`: dataset preparation utilities
- `src/models/`: model loading utilities
- `src/extract/`: embedding and hidden-state extraction
- `src/analysis/`: probing and dimensional analysis
- `outputs/`: extracted features, figures, metrics

## Notes on the dataset

This scaffold assumes:

- one utterance per speaker is the common case
- recordings may arrive in mixed formats and sampling rates
- metadata will need canonicalization into a single manifest table
- the primary transcript is the shared "Please call Stella" passage

## Recommended first experiment

- Use a frozen `Wav2Vec2Model`
- Extract all hidden layers with `output_hidden_states=True`
- Save utterance-level mean pooled embeddings for every layer
- Start with broad accent or native-language group labels rather than fine country labels

# Speech Accent Representation Experiments

This document is the single unified write-up for the project. It describes the current label definition, all four standardized datasets, the probing procedure, centroid clustering, output folders, and commands for running the full grid.

## Research Goal

We study where accent-related information is represented inside frozen Wav2Vec2 models. The main questions are:

1. Which layers make first-language/country labels most linearly decodable?
2. Does English ASR fine-tuning change where this information appears?
3. Which first-language/country groups are close in embedding space?
4. Are conclusions robust to class-size and sex imbalance?

## Data

The data come from the Speech Accent Archive. Each speaker reads the same elicitation passage, which reduces lexical-content variation across recordings.

Pipeline artifacts:

- `data/manifests/manifest.csv`: one row per downloaded recording.
- `data/manifests/manifest_normalized.csv`: normalized mono 16 kHz WAV manifest.
- `data/manifests/feature_index_facebook_wav2vec2_base.csv`: cached feature index for `facebook/wav2vec2-base`.
- `data/manifests/feature_index_facebook_wav2vec2_base_960h.csv`: cached feature index for `facebook/wav2vec2-base-960h`.

Raw audio and feature caches are not tracked in git. Analysis outputs are small enough to track under `outputs/analysis/`.

## Corrected Label Definition

Earlier experiments used the archive's `native_language` field directly. That caused a problem because the archive sometimes uses broad super-group labels. The most important case is `chinese`, which is often paired with a more specific first-language label in `all_native_languages`.

The current experiments derive:

```text
first_language
```

from `all_native_languages`, then build the final probe label as:

```text
first_language__country_of_origin
```

Examples:

```text
chinese;mandarin -> mandarin
cantonese;chinese -> cantonese
chinese;wu -> wu
chinese;taiwanese -> taiwanese
```

So examples of final labels are:

```text
mandarin__china
cantonese__china
english__usa
portuguese__brazil
```

This is why new outputs are written under `first_language_country_*` folders. The older `accent_country_*` folders are legacy outputs from the broader label definition and are not updated by the current grid.

## English-Residence Filter

For speakers born outside configured English-speaking birth countries, we exclude speakers with more than two years of English residence:

```text
length_of_english_residence > 2 years
```

Default English-speaking birth countries:

```text
usa, uk, canada, australia, ireland, new zealand, south africa
```

This attempts to reduce long-term English immersion as a confound for non-English birth-country groups.

## Models

Each experiment is run on two frozen checkpoints:

```text
facebook/wav2vec2-base
facebook/wav2vec2-base-960h
```

Both models are frozen. Only the linear probes are trained.

Feature shape per recording:

```text
pooled_hidden_states: (13, 768)
```

Layer 0 is the feature-projection output; layers 1-12 are Transformer layers.

## Standardized Four-Dataset Grid

The current grid is:

| Dataset | Inclusion threshold | Speakers used | Output directory |
|---|---:|---|---|
| min5 all-speaker | at least 5 female/male pairs | all speakers in each included label | `outputs/analysis/first_language_country_min5_pairs/` |
| min5 balanced-sex | at least 5 female/male pairs | exactly 5F + 5M per label | `outputs/analysis/first_language_country_min5_pairs_balanced_sex/` |
| min15 all-speaker | at least 15 female/male pairs | all speakers in each included label | `outputs/analysis/first_language_country_min15_pairs/` |
| min15 balanced-sex | at least 15 female/male pairs | exactly 15F + 15M per label | `outputs/analysis/first_language_country_min15_pairs_balanced_sex/` |

All four datasets use the same code path, label definition, English-residence filter, models, metrics, plotting, and centroid clustering.

The min5 datasets include more labels but have smaller classes. The min15 datasets include fewer labels but provide more reliable estimates. The min5 balanced-sex dataset has only 10 examples per class, so it is mainly a strict stress test. The min15 balanced-sex dataset has 30 examples per class and is the preferred balanced robustness check.

## Layer-Wise Linear Probing

For each dataset and model:

1. Load pooled hidden states.
2. For each layer, train a logistic-regression probe.
3. Use repeated stratified train/test splits.
4. Compare against majority and stratified dummy baselines.
5. Report accuracy, balanced accuracy, macro F1, and weighted F1.

Macro F1 and balanced accuracy are the main metrics because label counts are uneven.

Per-model outputs:

```text
layerwise_probe_results.csv
layerwise_probe_split_results.csv
probe_class_counts.csv
probe_balanced_pair_counts.csv
probe_feature_index_filtered.csv
best_layer_confusion_matrix.csv
layerwise_probe_macro_f1.png
```

Balanced-sex runs additionally write:

```text
probe_balanced_sample_counts.csv
```

Each dataset writes:

```text
model_comparison_macro_f1.png
```

## Centroid Clustering

After probing, each model/dataset run uses the best probe layer to compute one centroid per label:

```text
centroid(label) = mean embedding of all selected speakers in that label
```

Distances are cosine distances between centroids. Outputs:

```text
centroid_clustering_best_layer/layer_<N>_centroids.csv
centroid_clustering_best_layer/layer_<N>_group_counts.csv
centroid_clustering_best_layer/layer_<N>_centroid_distances.csv
centroid_clustering_best_layer/layer_<N>_centroid_distances_long.csv
centroid_clustering_best_layer/layer_<N>_cluster_order.csv
centroid_clustering_best_layer/layer_<N>_centroid_distance_heatmap.png
centroid_clustering_best_layer/layer_<N>_centroid_dendrogram.png
centroid_clustering_best_layer/layer_<N>_centroid_pca.png
```

This analysis asks which first-language/country groups are close in the model's representation space.

## Run Everything

From the project root:

```bash
cd /home/scur0061/interp-project/interp-32
bash jobs/submit_first_language_country_experiment_grid.sh
```

This submits:

```text
4 datasets x 2 models = 8 probe jobs
```

and schedules comparison plots and centroid-clustering jobs with Slurm dependencies.

## Run Individual Variants

Min5 all-speaker:

```bash
base_job=$(sbatch --parsable jobs/run_probes_first_language_country_min5_pairs_base.sh)
finetuned_job=$(sbatch --parsable jobs/run_probes_first_language_country_min5_pairs_base_960h.sh)
sbatch --dependency=afterok:$base_job:$finetuned_job jobs/plot_probe_comparison_first_language_country_min5_pairs.sh
sbatch --dependency=afterok:$base_job jobs/cluster_centroids_first_language_country_min5_pairs_base.sh
sbatch --dependency=afterok:$finetuned_job jobs/cluster_centroids_first_language_country_min5_pairs_base_960h.sh
```

Min5 balanced-sex:

```bash
base_job=$(sbatch --parsable jobs/run_probes_first_language_country_min5_pairs_balanced_base.sh)
finetuned_job=$(sbatch --parsable jobs/run_probes_first_language_country_min5_pairs_balanced_base_960h.sh)
sbatch --dependency=afterok:$base_job:$finetuned_job jobs/plot_probe_comparison_first_language_country_min5_pairs_balanced.sh
sbatch --dependency=afterok:$base_job jobs/cluster_centroids_first_language_country_min5_pairs_balanced_base.sh
sbatch --dependency=afterok:$finetuned_job jobs/cluster_centroids_first_language_country_min5_pairs_balanced_base_960h.sh
```

Min15 all-speaker:

```bash
base_job=$(sbatch --parsable jobs/run_probes_first_language_country_min15_pairs_base.sh)
finetuned_job=$(sbatch --parsable jobs/run_probes_first_language_country_min15_pairs_base_960h.sh)
sbatch --dependency=afterok:$base_job:$finetuned_job jobs/plot_probe_comparison_first_language_country_min15_pairs.sh
sbatch --dependency=afterok:$base_job jobs/cluster_centroids_first_language_country_min15_pairs_base.sh
sbatch --dependency=afterok:$finetuned_job jobs/cluster_centroids_first_language_country_min15_pairs_base_960h.sh
```

Min15 balanced-sex:

```bash
base_job=$(sbatch --parsable jobs/run_probes_first_language_country_min15_pairs_balanced_base.sh)
finetuned_job=$(sbatch --parsable jobs/run_probes_first_language_country_min15_pairs_balanced_base_960h.sh)
sbatch --dependency=afterok:$base_job:$finetuned_job jobs/plot_probe_comparison_first_language_country_min15_pairs_balanced.sh
sbatch --dependency=afterok:$base_job jobs/cluster_centroids_first_language_country_min15_pairs_balanced_base.sh
sbatch --dependency=afterok:$finetuned_job jobs/cluster_centroids_first_language_country_min15_pairs_balanced_base_960h.sh
```

## Why The Old Folders Did Not Change

The folders:

```text
outputs/analysis/accent_country_min5_pairs/
outputs/analysis/accent_country_min15_pairs/
```

belong to older broad-label experiments. They are not updated by the current standardized grid.

The current outputs are:

```text
outputs/analysis/first_language_country_min5_pairs/
outputs/analysis/first_language_country_min5_pairs_balanced_sex/
outputs/analysis/first_language_country_min15_pairs/
outputs/analysis/first_language_country_min15_pairs_balanced_sex/
```

## Poster Story

The clean story is:

1. We use Speech Accent Archive recordings of a shared passage.
2. We correct broad language labels into first-language labels, especially for Chinese/Mandarin/Cantonese.
3. We compare frozen Wav2Vec2 base and English-ASR-fine-tuned Wav2Vec2.
4. We probe each layer with logistic regression to locate where first-language/country information is linearly decodable.
5. We cluster group centroids at the best layer to visualize which groups are close in embedding space.
6. We compare all-speaker and balanced-sex variants to test robustness to class-size and sex imbalance.

Suggested poster figures:

1. Dataset and pipeline diagram.
2. Layer-wise macro F1 for min5 and min15 all-speaker.
3. Layer-wise macro F1 for min15 balanced-sex.
4. Best-layer confusion matrix.
5. Centroid distance heatmap or dendrogram.

## Limitations

- Country of origin and first language are imperfect proxies for accent.
- Some metadata remains coarse or self-reported.
- Mean pooling discards temporal phonetic structure.
- Balanced-sex runs reduce data and can be noisy.
- Linear probes measure decodability, not causal model use.

## Interpretation Rule Of Thumb

The strongest results are patterns that appear across:

```text
min5 all-speaker
min15 all-speaker
min15 balanced-sex
```

Treat min5 balanced-sex cautiously because it has only 10 examples per label.

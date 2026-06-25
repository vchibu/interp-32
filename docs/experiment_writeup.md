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

## Standardized Dataset Grid

The current grid is:

| Dataset | Inclusion threshold | Speakers used | Output directory |
|---|---:|---|---|
| min5 all-speaker | at least 5 female/male pairs | all speakers in each included label | `outputs/analysis/first_language_country_min5_pairs/` |
| min5 balanced-sex | at least 5 female/male pairs | exactly 5F + 5M per label | `outputs/analysis/first_language_country_min5_pairs_balanced_sex/` |
| min10 all-speaker | at least 10 female/male pairs | all speakers in each included label | `outputs/analysis/first_language_country_min10_pairs/` |
| min10 balanced-sex | at least 10 female/male pairs | exactly 10F + 10M per label | `outputs/analysis/first_language_country_min10_pairs_balanced_sex/` |
| min15 all-speaker | at least 15 female/male pairs | all speakers in each included label | `outputs/analysis/first_language_country_min15_pairs/` |
| min15 balanced-sex | at least 15 female/male pairs | exactly 15F + 15M per label | `outputs/analysis/first_language_country_min15_pairs_balanced_sex/` |

All datasets use the same code path, label definition, English-residence filter, models, metrics, plotting, and centroid clustering.

The min5 datasets include more labels but have smaller classes. The min10 datasets are the middle setting: fewer groups than min5, but less data-starved than the 5F/5M balanced run. The min15 datasets include fewer labels but provide more reliable estimates. The min5 balanced-sex dataset has only 10 examples per class, so it is mainly a strict stress test. The min10 balanced-sex dataset has 20 examples per class. The min15 balanced-sex dataset has 30 examples per class and is the preferred high-confidence balanced robustness check.

## Exact Dataset Construction And Splitting

This section describes precisely how every dataset version is constructed and how examples are split for probing. The important distinction is:

```text
dataset construction = which speakers are eligible and selected
probe split = how the selected speakers are divided into train/test folds
```

The same construction pipeline is used for all dataset versions until the final all-speaker versus balanced-sex selection step.

### Step 1: Start From The Model Feature Index

Each model has its own feature index:

```text
data/manifests/feature_index_facebook_wav2vec2_base.csv
data/manifests/feature_index_facebook_wav2vec2_base_960h.csv
```

These files contain one row per recording/speaker with:

- `speaker_id`
- `sample_name`
- `feature_path`
- `native_language`
- `all_native_languages`
- `country_of_origin`
- `sex`
- `length_of_english_residence`
- other metadata copied from the manifest

The feature paths point to cached `.npz` files containing:

```text
pooled_hidden_states: shape (13, 768)
```

The same speaker inclusion logic is applied separately to each model's feature index, but the metadata and selected speaker identities should match across models because both feature indexes come from the same normalized manifest.

### Step 2: Derive `first_language`

The code derives a new column:

```text
first_language
```

from `native_language` and `all_native_languages`.

The rule is:

1. Normalize labels to lowercase and strip whitespace.
2. Split `all_native_languages` on semicolons.
3. If `native_language` is a broad label such as `chinese` and a more specific label is present, use the more specific label.
4. Otherwise keep the cleaned `native_language` when it is already specific.
5. If only one usable non-broad label exists, use that.
6. Otherwise fall back to the first label in `all_native_languages`.

Examples:

```text
native_language=chinese, all_native_languages=chinese;mandarin -> first_language=mandarin
native_language=cantonese, all_native_languages=cantonese;chinese -> first_language=cantonese
native_language=chinese, all_native_languages=chinese;wu -> first_language=wu
native_language=english, all_native_languages=english -> first_language=english
```

This is the key reason the current experiment folders are named `first_language_country_*`.

### Step 3: Derive The Probe Label

The final class label is:

```text
accent_country_label = first_language + "__" + country_of_origin
```

Examples:

```text
english__usa
mandarin__china
cantonese__china
arabic__saudi arabia
korean__south korea
```

Rows without a usable first-language label, country-of-origin label, feature path, or binary sex label are removed before probing.

### Step 4: Apply English-Residence Filtering

For speakers born outside the configured English-speaking birth countries, the code excludes speakers with:

```text
length_of_english_residence > 2 years
```

The default English-speaking birth-country list is:

```text
usa
uk
canada
australia
ireland
new zealand
south africa
```

So:

- `english__usa` speakers are not excluded for high English residence.
- `french__canada` speakers are not excluded for high English residence because Canada is in the English-speaking birth-country list.
- `mandarin__china` speakers with more than two years English residence are excluded.
- `portuguese__brazil` speakers with more than two years English residence are excluded.

The parsed numeric residence duration is written as:

```text
english_residence_years
```

The filter marker columns include:

```text
is_english_speaking_birth_country
excluded_for_english_residence
eligible_for_accent_country_probe
```

### Step 5: Count Female/Male Availability Per Label

After the residence filter, the code counts speakers per label:

```text
n_female
n_male
n_balanced_female_male_pairs = min(n_female, n_male)
```

This count determines whether a label is eligible for the min5 or min15 datasets.

The count table is saved as:

```text
probe_balanced_pair_counts.csv
```

### Step 6: Dataset-Specific Speaker Selection

The four dataset versions differ only here.

#### 1. `first_language_country_min5_pairs`

Inclusion rule:

```text
n_balanced_female_male_pairs >= 5
```

Speaker usage:

```text
use all speakers from every included label
```

This means labels are eligible if they have at least 5 female and 5 male speakers, but the actual probe uses all speakers in those labels. For example, if a label has 73 female and 33 male speakers, all 106 speakers are kept.

Purpose:

```text
maximum label coverage while still requiring basic female/male support
```

Output folder:

```text
outputs/analysis/first_language_country_min5_pairs/
```

#### 2. `first_language_country_min5_pairs_balanced_sex`

Inclusion rule:

```text
n_balanced_female_male_pairs >= 5
```

Speaker usage:

```text
sample exactly 5 female + 5 male speakers per included label
```

Sampling is random but reproducible using the configured random seed. Speakers not selected by the balanced sample are not used in that run.

Purpose:

```text
strict sex-balanced robustness check with broad label coverage
```

Important caveat:

```text
only 10 speakers per label, so probe results can be noisy and data-starved
```

Output folder:

```text
outputs/analysis/first_language_country_min5_pairs_balanced_sex/
```

Additional output:

```text
probe_balanced_sample_counts.csv
```

#### 3. `first_language_country_min10_pairs`

Inclusion rule:

```text
n_balanced_female_male_pairs >= 10
```

Speaker usage:

```text
use all speakers from every included label
```

This is the middle all-speaker setting. It keeps fewer labels than min5 but more labels than min15. Because all speakers are retained, it still uses natural class sizes, with stratified train/test splits and balanced logistic-regression class weights.

Purpose:

```text
middle-coverage all-speaker probe with stronger class support than min5
```

Output folder:

```text
outputs/analysis/first_language_country_min10_pairs/
```

#### 4. `first_language_country_min10_pairs_balanced_sex`

Inclusion rule:

```text
n_balanced_female_male_pairs >= 10
```

Speaker usage:

```text
sample exactly 10 female + 10 male speakers per included label
```

Sampling is random but reproducible using the configured random seed.

Purpose:

```text
middle balanced-sex robustness check with 20 speakers per label
```

Output folder:

```text
outputs/analysis/first_language_country_min10_pairs_balanced_sex/
```

Additional output:

```text
probe_balanced_sample_counts.csv
```

#### 5. `first_language_country_min15_pairs`

Inclusion rule:

```text
n_balanced_female_male_pairs >= 15
```

Speaker usage:

```text
use all speakers from every included label
```

This is stricter than min5, so it includes fewer labels but each included label has stronger female/male support.

Purpose:

```text
higher-confidence all-speaker probe with fewer, larger classes
```

Output folder:

```text
outputs/analysis/first_language_country_min15_pairs/
```

#### 6. `first_language_country_min15_pairs_balanced_sex`

Inclusion rule:

```text
n_balanced_female_male_pairs >= 15
```

Speaker usage:

```text
sample exactly 15 female + 15 male speakers per included label
```

Sampling is random but reproducible using the configured random seed.

Purpose:

```text
preferred balanced-sex robustness check because it has 30 speakers per label
```

Output folder:

```text
outputs/analysis/first_language_country_min15_pairs_balanced_sex/
```

Additional output:

```text
probe_balanced_sample_counts.csv
```

### Step 7: Final Probe Feature Index

After all filtering and optional balanced sampling, the exact selected rows are written to:

```text
probe_feature_index_filtered.csv
```

This file is the definitive record of which speakers were used in a specific model/dataset probe run. It includes the selected labels, metadata, feature paths, and filter columns.

For any result, the corresponding `probe_feature_index_filtered.csv` is the file to inspect when asking:

```text
Which speakers were actually included?
Which labels were included?
How many examples per label were used?
Was this an all-speaker or balanced-sex dataset?
```

### Step 8: Train/Test Splitting Inside The Probe

Once the final dataset has been constructed, the probe code performs repeated train/test splits.

The split procedure is:

```text
StratifiedShuffleSplit
```

with:

```text
test_size = 0.2
n_splits = 5
random_seed = 42
```

unless overridden in the CLI/config.

Stratification means every train/test split tries to preserve label proportions. For example, if the final dataset has 8 labels, every split attempts to place examples from all 8 labels into both train and test.

Important:

```text
The train/test split happens after speaker filtering and after balanced-sex sampling.
```

So for min15 balanced-sex:

1. First select exactly 15 female and 15 male speakers per label.
2. Then split those 30 speakers per label into train/test folds.

With `test_size=0.2`, approximately:

```text
24 train speakers per label
6 test speakers per label
```

For min5 balanced-sex:

1. First select exactly 5 female and 5 male speakers per label.
2. Then split those 10 speakers per label into train/test folds.

With `test_size=0.2`, approximately:

```text
8 train speakers per label
2 test speakers per label
```

This is why min5 balanced-sex is likely too small for stable probe training.

For min10 balanced-sex:

1. First select exactly 10 female and 10 male speakers per label.
2. Then split those 20 speakers per label into train/test folds.

With `test_size=0.2`, approximately:

```text
16 train speakers per label
4 test speakers per label
```

This is the middle balanced setting: still smaller than the all-speaker datasets, but less data-starved than min5 balanced-sex.

### Step 9: Layer-Wise Probe Training

For each split and each layer:

1. Load `pooled_hidden_states[layer]` for every selected speaker.
2. Fit a `StandardScaler` on the train embeddings only.
3. Train logistic regression on the scaled train embeddings.
4. Evaluate on the held-out test embeddings.

The logistic regression uses class weighting by default:

```text
class_weight = balanced
```

This helps with label imbalance in the all-speaker datasets.

### Step 10: Baselines

Every probe run also evaluates:

```text
majority baseline
stratified random baseline
```

These are written into:

```text
layerwise_probe_split_results.csv
layerwise_probe_results.csv
```

The probe is meaningful only if it exceeds these baselines, especially on macro F1 and balanced accuracy.

### Step 11: Best Layer Selection

The best layer is selected by:

```text
highest macro_f1_mean among linear_probe rows
```

Ties are resolved by balanced accuracy and then by lower layer index.

The best layer is used for:

```text
best_layer_confusion_matrix.csv
centroid_clustering_best_layer/
```

### Step 12: Centroid Clustering Dataset

Centroid clustering uses exactly the same speakers as the corresponding probe run:

```text
probe_feature_index_filtered.csv
```

It does not rerun dataset filtering. It reads the final selected feature index and computes one centroid per label at the best probe layer.

Therefore:

- min5 all-speaker clustering uses all selected min5 speakers.
- min5 balanced-sex clustering uses exactly 5F + 5M per label.
- min15 all-speaker clustering uses all selected min15 speakers.
- min15 balanced-sex clustering uses exactly 15F + 15M per label.

This guarantees that probe results and centroid plots correspond to the same dataset version.

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
6 datasets x 2 models = 12 probe jobs
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

Min10 all-speaker:

```bash
base_job=$(sbatch --parsable jobs/run_probes_first_language_country_min10_pairs_base.sh)
finetuned_job=$(sbatch --parsable jobs/run_probes_first_language_country_min10_pairs_base_960h.sh)
sbatch --dependency=afterok:$base_job:$finetuned_job jobs/plot_probe_comparison_first_language_country_min10_pairs.sh
sbatch --dependency=afterok:$base_job jobs/cluster_centroids_first_language_country_min10_pairs_base.sh
sbatch --dependency=afterok:$finetuned_job jobs/cluster_centroids_first_language_country_min10_pairs_base_960h.sh
```

Min10 balanced-sex:

```bash
base_job=$(sbatch --parsable jobs/run_probes_first_language_country_min10_pairs_balanced_base.sh)
finetuned_job=$(sbatch --parsable jobs/run_probes_first_language_country_min10_pairs_balanced_base_960h.sh)
sbatch --dependency=afterok:$base_job:$finetuned_job jobs/plot_probe_comparison_first_language_country_min10_pairs_balanced.sh
sbatch --dependency=afterok:$base_job jobs/cluster_centroids_first_language_country_min10_pairs_balanced_base.sh
sbatch --dependency=afterok:$finetuned_job jobs/cluster_centroids_first_language_country_min10_pairs_balanced_base_960h.sh
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
outputs/analysis/first_language_country_min10_pairs/
outputs/analysis/first_language_country_min10_pairs_balanced_sex/
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
2. Layer-wise macro F1 for min5, min10, and min15 all-speaker.
3. Layer-wise macro F1 for min10 and min15 balanced-sex.
4. Best-layer confusion matrix.
5. Centroid distance heatmap or dendrogram.
6. Multilingual comparison: layer-wise macro F1 for wav2vec2-large-lv60-self (English) vs XLS-R (multilingual) on min10 balanced-sex.
7. Centroid dendrogram comparison between English-only and multilingual models at their respective best layers.

## Multilingual Pretraining Extension

The main grid uses two English-only wav2vec2 checkpoints. The multilingual extension asks whether pretraining on many languages changes where accent information is represented.

### Research Questions

1. Is first-language/country information more linearly decodable from a multilingual model than from an English-only model?
2. Does the best probe layer shift between English-only and multilingual pretraining?
3. Do centroid distance structures (language-family clustering) differ between the two models?

### Model Pair

Both models are self-supervised, same architecture (wav2vec2-large, 24 Transformer layers, hidden dim 1024), same objective. This isolates the effect of pretraining language coverage.

| Model | Params | Pretraining data | Languages |
|---|---:|---|---|
| `facebook/wav2vec2-large-lv60-self` | ~316M | LibriSpeech-LV-60k + 960h (~960h) | English only |
| `facebook/wav2vec2-xls-r-300m` | ~300M | ~436k hours | 128 languages |

Feature shape per recording (both large models):

```text
pooled_hidden_states: (25, 1024)
```

Layer 0 is the feature-projection output; layers 1-24 are Transformer layers.

### Dataset

The multilingual extension uses the min10 balanced-sex dataset:

```text
first_language_country_min10_pairs_balanced_sex
```

- Inclusion: labels with at least 10 female/male pairs.
- Sampling: exactly 10 female + 10 male speakers per label (20 total).
- Split: StratifiedShuffleSplit, test_size=0.2, n_splits=5, seed=42.

This gives 16 train + 4 test speakers per label per fold. It is the middle-ground robustness check: more stable than min5 balanced-sex (10 per label) but still sex-balanced.

### Output Directory

```text
outputs/analysis/first_language_country_min10_pairs_balanced_sex/
```

Per-model subdirectories:

```text
facebook_wav2vec2_large_lv60_self/
facebook_wav2vec2_xls_r_300m/
```

Each contains the same outputs as the main grid:

```text
layerwise_probe_results.csv
layerwise_probe_split_results.csv
probe_class_counts.csv
probe_balanced_pair_counts.csv
probe_balanced_sample_counts.csv
probe_feature_index_filtered.csv
best_layer_confusion_matrix.csv
layerwise_probe_macro_f1.png
centroid_clustering_best_layer/
```

The comparison plot is written to:

```text
outputs/analysis/first_language_country_min10_pairs_balanced_sex/model_comparison_macro_f1_multilingual.png
```

### Confounds And Limitations

- XLS-R was pretrained on more total audio (~436k h vs 960h). This is unavoidable because multilingual models need more data. The comparison therefore conflates language coverage with data volume.
- Parameter count differs slightly (316M vs 300M, ~5%). This is within noise for probing.
- Accent is not the same as language. Multilingual pretraining may help language identification but accent is intra-/cross-language phonetic variation, so the prediction is genuinely uncertain.

### Citations

- Babu et al. (2021). XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale. arXiv:2111.09296.
- Conneau et al. (2021). Unsupervised Cross-lingual Representation Learning for Speech Recognition. arXiv:2006.13979.

### Run The Multilingual Extension

First extract hidden states for both large models (one-time cost per model):

```bash
sbatch jobs/extract_hidden_states_large_lv60_self_h100.sh
sbatch jobs/extract_hidden_states_xls_r_300m_h100.sh
```

Then submit the probe + comparison + clustering grid:

```bash
bash jobs/submit_multilingual_min10_balanced_experiment.sh
```

This submits:

```text
2 probe jobs (one per model)
1 comparison plot (dependency on both probes)
2 centroid clustering jobs (one per model, dependency on respective probe)
```

Or run individual variants:

```bash
lv60_job=$(sbatch --parsable jobs/run_probes_first_language_country_min10_pairs_balanced_large_lv60_self.sh)
xlsr_job=$(sbatch --parsable jobs/run_probes_first_language_country_min10_pairs_balanced_xls_r_300m.sh)
sbatch --dependency=afterok:$lv60_job:$xlsr_job jobs/plot_probe_comparison_first_language_country_min10_pairs_balanced_multilingual.sh
sbatch --dependency=afterok:$lv60_job jobs/cluster_centroids_first_language_country_min10_pairs_balanced_large_lv60_self.sh
sbatch --dependency=afterok:$xlsr_job jobs/cluster_centroids_first_language_country_min10_pairs_balanced_xls_r_300m.sh
```

## Limitations

- Country of origin and first language are imperfect proxies for accent.
- Some metadata remains coarse or self-reported.
- Mean pooling discards temporal phonetic structure.
- Balanced-sex runs reduce data and can be noisy.
- Linear probes measure decodability, not causal model use.
- Multilingual vs English-only comparison conflates language coverage with total pretraining hours (see Multilingual Pretraining Extension).

## Interpretation Rule Of Thumb

The strongest results are patterns that appear across:

```text
min5 all-speaker
min10 all-speaker
min10 balanced-sex
min15 all-speaker
min15 balanced-sex
```

Treat min5 balanced-sex cautiously because it has only 10 examples per label. The min10 balanced-sex run is the middle-ground robustness check with 20 examples per label.

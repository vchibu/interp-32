#!/usr/bin/env bash
# Multilingual comparison experiment: min15 balanced-sex dataset.
#
# Compares two self-supervised wav2vec2-large models:
#   - facebook/wav2vec2-large-lv60       (English-only pretraining, ~316M)
#   - facebook/wav2vec2-xls-r-300m        (multilingual pretraining, 128 langs, ~300M)
#
# Dataset: first_language_country_min15_pairs_balanced_sex
#   - labels with >= 15 female/male pairs
#   - exactly 15 female + 15 male speakers per label (30 total)
#
# Missing feature indexes are submitted automatically before the probes.
set -euo pipefail

cd /home/scur0061/interp-project/interp-32

echo "Submitting multilingual min15 balanced-sex experiment..."

lv60_feature_index=data/manifests/feature_index_facebook_wav2vec2_large_lv60_self.csv
xlsr_feature_index=data/manifests/feature_index_facebook_wav2vec2_xls_r_300m.csv

lv60_dependency=()
if [[ ! -s "$lv60_feature_index" ]]; then
  lv60_extract_job=$(sbatch --parsable jobs/extract_hidden_states_large_lv60_self_h100.sh)
  lv60_dependency=(--dependency=afterok:$lv60_extract_job)
else
  lv60_extract_job="existing $lv60_feature_index"
fi

xlsr_dependency=()
if [[ ! -s "$xlsr_feature_index" ]]; then
  xlsr_extract_job=$(sbatch --parsable jobs/extract_hidden_states_xls_r_300m_h100.sh)
  xlsr_dependency=(--dependency=afterok:$xlsr_extract_job)
else
  xlsr_extract_job="existing $xlsr_feature_index"
fi

lv60_job=$(sbatch --parsable "${lv60_dependency[@]}" jobs/run_probes_first_language_country_min15_pairs_balanced_large_lv60_self.sh)
xlsr_job=$(sbatch --parsable "${xlsr_dependency[@]}" jobs/run_probes_first_language_country_min15_pairs_balanced_xls_r_300m.sh)

sbatch --dependency=afterok:$lv60_job:$xlsr_job jobs/plot_probe_comparison_first_language_country_min15_pairs_balanced_multilingual.sh
sbatch --dependency=afterok:$lv60_job jobs/cluster_centroids_first_language_country_min15_pairs_balanced_large_lv60_self.sh
sbatch --dependency=afterok:$xlsr_job jobs/cluster_centroids_first_language_country_min15_pairs_balanced_xls_r_300m.sh

cat <<EOF
Submitted multilingual min15 balanced-sex probe jobs:
  wav2vec2-large-lv60 feature index:      $lv60_extract_job
  wav2vec2-xls-r-300m feature index:      $xlsr_extract_job
  wav2vec2-large-lv60 (English):          $lv60_job
  wav2vec2-xls-r-300m (multilingual):     $xlsr_job

Comparison plot and centroid clustering scheduled with Slurm dependencies.

Check progress:
  squeue -u $USER
EOF

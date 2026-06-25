#!/usr/bin/env bash
# Multilingual comparison experiment: min10 balanced-sex dataset.
#
# Compares two self-supervised wav2vec2-large models:
#   - facebook/wav2vec2-large-lv60-self  (English-only pretraining, ~316M)
#   - facebook/wav2vec2-xls-r-300m        (multilingual pretraining, 128 langs, ~300M)
#
# Both models are the same architecture (wav2vec2-large, 24 layers, dim 1024)
# and the same self-supervised objective, isolating the effect of pretraining
# language coverage. The only remaining confound is total pretraining hours
# (960h vs ~436k h), which is unavoidable for multilingual models.
#
# Dataset: first_language_country_min10_pairs_balanced_sex
#   - labels with >= 10 female/male pairs
#   - exactly 10 female + 10 male speakers per label (20 total)
#
# Prerequisite: feature extraction jobs must have completed and produced
#   data/manifests/feature_index_facebook_wav2vec2_large_lv60_self.csv
#   data/manifests/feature_index_facebook_wav2vec2_xls_r_300m.csv
# Run those first:
#   sbatch jobs/extract_hidden_states_large_lv60_self_h100.sh
#   sbatch jobs/extract_hidden_states_xls_r_300m_h100.sh
set -euo pipefail

cd /home/scur0061/interp-project/interp-32

echo "Submitting multilingual min10 balanced-sex experiment..."

lv60_job=$(sbatch --parsable jobs/run_probes_first_language_country_min10_pairs_balanced_large_lv60_self.sh)
xlsr_job=$(sbatch --parsable jobs/run_probes_first_language_country_min10_pairs_balanced_xls_r_300m.sh)

sbatch --dependency=afterok:$lv60_job:$xlsr_job jobs/plot_probe_comparison_first_language_country_min10_pairs_balanced_multilingual.sh
sbatch --dependency=afterok:$lv60_job jobs/cluster_centroids_first_language_country_min10_pairs_balanced_large_lv60_self.sh
sbatch --dependency=afterok:$xlsr_job jobs/cluster_centroids_first_language_country_min10_pairs_balanced_xls_r_300m.sh

cat <<EOF
Submitted multilingual min10 balanced-sex probe jobs:
  wav2vec2-large-lv60-self (English):     \$lv60_job
  wav2vec2-xls-r-300m (multilingual):     \$xlsr_job

Comparison plot and centroid clustering scheduled with Slurm dependencies.

Check progress:
  squeue -u \$USER
EOF

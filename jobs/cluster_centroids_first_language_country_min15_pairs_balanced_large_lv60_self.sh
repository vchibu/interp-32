#!/usr/bin/env bash
#SBATCH --job-name=cluster-flc-lv60-p15bal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd /home/scur0061/interp-project/interp-32
mkdir -p logs

source .venv/bin/activate
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/matplotlib-$USER

python -m src.cli cluster-centroids \
  --feature-index-path outputs/analysis/first_language_country_min15_pairs_balanced_sex/facebook_wav2vec2_large_lv60_self/probe_feature_index_filtered.csv \
  --probe-results-path outputs/analysis/first_language_country_min15_pairs_balanced_sex/facebook_wav2vec2_large_lv60_self/layerwise_probe_results.csv \
  --output-dir outputs/analysis/first_language_country_min15_pairs_balanced_sex/facebook_wav2vec2_large_lv60_self/centroid_clustering_best_layer \
  --label-field accent_country_label \
  --layer best

#!/usr/bin/env bash
#SBATCH --job-name=plot-flc-p10bal-ml
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd /home/scur0061/interp-project/interp-32
mkdir -p logs

source .venv/bin/activate
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/matplotlib-$USER

python -m src.cli plot-probe-comparison \
  --output-path outputs/analysis/first_language_country_min10_pairs_balanced_sex/model_comparison_macro_f1_multilingual.png \
  --summary "wav2vec2-large-lv60 (English)" outputs/analysis/first_language_country_min10_pairs_balanced_sex/facebook_wav2vec2_large_lv60_self/layerwise_probe_results.csv \
  --summary "wav2vec2-xls-r-300m (multilingual)" outputs/analysis/first_language_country_min10_pairs_balanced_sex/facebook_wav2vec2_xls_r_300m/layerwise_probe_results.csv

#!/usr/bin/env bash
#SBATCH --job-name=plot-flc-p5
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
  --output-path outputs/analysis/first_language_country_min5_pairs/model_comparison_macro_f1.png \
  --summary base outputs/analysis/first_language_country_min5_pairs/facebook_wav2vec2_base/layerwise_probe_results.csv \
  --summary base_960h outputs/analysis/first_language_country_min5_pairs/facebook_wav2vec2_base_960h/layerwise_probe_results.csv

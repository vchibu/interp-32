#!/usr/bin/env bash
#SBATCH --job-name=probe-ac-960h-p15
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd /home/scur0061/interp-project/interp-32
mkdir -p logs

source .venv/bin/activate
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/matplotlib-$USER

python -m src.cli run-probes \
  --model-name facebook/wav2vec2-base-960h \
  --analysis-dir outputs/analysis/accent_country_min15_pairs/facebook_wav2vec2_base_960h \
  --accent-country-filter \
  --min-balanced-sex-pairs 15 \
  --min-examples-per-label 1

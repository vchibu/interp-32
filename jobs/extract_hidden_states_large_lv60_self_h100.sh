#!/usr/bin/env bash
#SBATCH --job-name=accent-feat-lv60
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd /home/scur0061/interp-project/interp-32
mkdir -p logs

source .venv/bin/activate
export PYTHONUNBUFFERED=1

python -m src.cli extract-hidden-states \
  --model-name facebook/wav2vec2-large-lv60 \
  --feature-index-path data/manifests/feature_index_facebook_wav2vec2_large_lv60_self.csv \
  --feature-dir outputs/features/facebook_wav2vec2_large_lv60_self

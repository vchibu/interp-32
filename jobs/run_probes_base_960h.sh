#!/usr/bin/env bash
#SBATCH --job-name=accent-probe-960h
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

python -m src.cli run-probes \
  --model-name facebook/wav2vec2-base-960h

#!/usr/bin/env bash
#SBATCH --job-name=accent-normalize
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd /home/scur0061/interp-project/interp-32
mkdir -p logs

source .venv/bin/activate
export PYTHONUNBUFFERED=1

python -m src.cli normalize-audio --num-workers "${SLURM_CPUS_PER_TASK}"

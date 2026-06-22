#!/usr/bin/env bash
#SBATCH --job-name=accent-summary
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd /home/scur0061/interp-project/interp-32
mkdir -p logs

source .venv/bin/activate
export PYTHONUNBUFFERED=1

python -u - <<'PY'
from pathlib import Path

import pandas as pd

from src.analysis.dataset_summary import save_summary, summarize_top_accent_groups

metadata_path = Path("data/raw/accent_archive_metadata.csv")
output_path = Path("outputs/analysis/accent_groups_min_10.csv")

metadata = pd.read_csv(metadata_path)
summary = summarize_top_accent_groups(metadata=metadata, min_total_speakers=10, top_n=100000)
save_summary(summary, output_path)
print(summary.to_string(index=False))
print(f"Summary written to {output_path}")
PY
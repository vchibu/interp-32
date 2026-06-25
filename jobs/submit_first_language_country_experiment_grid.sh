#!/usr/bin/env bash
set -euo pipefail

cd /home/scur0061/interp-project/interp-32

echo "Submitting first-language/country experiment grid..."

min5_base=$(sbatch --parsable jobs/run_probes_first_language_country_min5_pairs_base.sh)
min5_960h=$(sbatch --parsable jobs/run_probes_first_language_country_min5_pairs_base_960h.sh)
sbatch --dependency=afterok:$min5_base:$min5_960h jobs/plot_probe_comparison_first_language_country_min5_pairs.sh
sbatch --dependency=afterok:$min5_base jobs/cluster_centroids_first_language_country_min5_pairs_base.sh
sbatch --dependency=afterok:$min5_960h jobs/cluster_centroids_first_language_country_min5_pairs_base_960h.sh

min5_bal_base=$(sbatch --parsable jobs/run_probes_first_language_country_min5_pairs_balanced_base.sh)
min5_bal_960h=$(sbatch --parsable jobs/run_probes_first_language_country_min5_pairs_balanced_base_960h.sh)
sbatch --dependency=afterok:$min5_bal_base:$min5_bal_960h jobs/plot_probe_comparison_first_language_country_min5_pairs_balanced.sh
sbatch --dependency=afterok:$min5_bal_base jobs/cluster_centroids_first_language_country_min5_pairs_balanced_base.sh
sbatch --dependency=afterok:$min5_bal_960h jobs/cluster_centroids_first_language_country_min5_pairs_balanced_base_960h.sh

min10_base=$(sbatch --parsable jobs/run_probes_first_language_country_min10_pairs_base.sh)
min10_960h=$(sbatch --parsable jobs/run_probes_first_language_country_min10_pairs_base_960h.sh)
sbatch --dependency=afterok:$min10_base:$min10_960h jobs/plot_probe_comparison_first_language_country_min10_pairs.sh
sbatch --dependency=afterok:$min10_base jobs/cluster_centroids_first_language_country_min10_pairs_base.sh
sbatch --dependency=afterok:$min10_960h jobs/cluster_centroids_first_language_country_min10_pairs_base_960h.sh

min10_bal_base=$(sbatch --parsable jobs/run_probes_first_language_country_min10_pairs_balanced_base.sh)
min10_bal_960h=$(sbatch --parsable jobs/run_probes_first_language_country_min10_pairs_balanced_base_960h.sh)
sbatch --dependency=afterok:$min10_bal_base:$min10_bal_960h jobs/plot_probe_comparison_first_language_country_min10_pairs_balanced.sh
sbatch --dependency=afterok:$min10_bal_base jobs/cluster_centroids_first_language_country_min10_pairs_balanced_base.sh
sbatch --dependency=afterok:$min10_bal_960h jobs/cluster_centroids_first_language_country_min10_pairs_balanced_base_960h.sh

min15_base=$(sbatch --parsable jobs/run_probes_first_language_country_min15_pairs_base.sh)
min15_960h=$(sbatch --parsable jobs/run_probes_first_language_country_min15_pairs_base_960h.sh)
sbatch --dependency=afterok:$min15_base:$min15_960h jobs/plot_probe_comparison_first_language_country_min15_pairs.sh
sbatch --dependency=afterok:$min15_base jobs/cluster_centroids_first_language_country_min15_pairs_base.sh
sbatch --dependency=afterok:$min15_960h jobs/cluster_centroids_first_language_country_min15_pairs_base_960h.sh

min15_bal_base=$(sbatch --parsable jobs/run_probes_first_language_country_min15_pairs_balanced_base.sh)
min15_bal_960h=$(sbatch --parsable jobs/run_probes_first_language_country_min15_pairs_balanced_base_960h.sh)
sbatch --dependency=afterok:$min15_bal_base:$min15_bal_960h jobs/plot_probe_comparison_first_language_country_min15_pairs_balanced.sh
sbatch --dependency=afterok:$min15_bal_base jobs/cluster_centroids_first_language_country_min15_pairs_balanced_base.sh
sbatch --dependency=afterok:$min15_bal_960h jobs/cluster_centroids_first_language_country_min15_pairs_balanced_base_960h.sh

cat <<EOF
Submitted probe jobs:
  min5 all-speaker:       $min5_base, $min5_960h
  min5 balanced-sex:      $min5_bal_base, $min5_bal_960h
  min10 all-speaker:      $min10_base, $min10_960h
  min10 balanced-sex:     $min10_bal_base, $min10_bal_960h
  min15 all-speaker:      $min15_base, $min15_960h
  min15 balanced-sex:     $min15_bal_base, $min15_bal_960h

Check progress:
  squeue -u \$USER
EOF

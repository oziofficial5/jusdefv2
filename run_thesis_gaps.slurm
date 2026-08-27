#!/bin/bash
#SBATCH --job-name=jusdef-gaps
#SBATCH --partition=gpu          # TODO: confirm partition with `sinfo` on Milan cluster
#SBATCH --gres=gpu:a100:1        # or gpu:ampere:1 depending on cluster
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/gaps-%j.out
#SBATCH --error=logs/gaps-%j.err

# Two outstanding thesis experiments:
#   Part 1  sentence-count stratification of the LEDGAR 10-20% regime
#           (inference only, ~30 min)
#   Part 2  v1 under the corrected protocol + per-correction ablations
#           (6 EUR-Lex trainings at ~2.7 h each on A100, ~16 h)
# Walltime is set to 24 h with headroom. Both parts are sentinel-gated, so a
# re-submission after a timeout resumes rather than restarting.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs outputs/checkpoints outputs/logs outputs/sentinels

bash ./scripts/run_thesis_gaps.sh

#!/bin/bash
#SBATCH --job-name=jusdef-lbretune
#SBATCH --partition=gpu          # TODO: confirm partition with `sinfo`
#SBATCH --gres=gpu:a100:1        # or gpu:ampere:1 depending on cluster
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/lbretune-%j.out
#SBATCH --error=logs/lbretune-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs outputs/checkpoints outputs/logs outputs/sentinels

bash ./scripts/run_legalbert_retune.sh

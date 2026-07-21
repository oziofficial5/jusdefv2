#!/bin/bash
#SBATCH --job-name=jusdef-ft10
#SBATCH --partition=gpu          # TODO: confirm partition with `sinfo` on Milan cluster
#SBATCH --gres=gpu:a100:1        # or gpu:ampere:1 depending on cluster
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=logs/ft10-%j.out
#SBATCH --error=logs/ft10-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs outputs/checkpoints outputs/logs outputs/sentinels

bash ./scripts/run_ft10_perclass.sh

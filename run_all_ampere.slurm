#!/bin/bash
#SBATCH --job-name=jusdef-all
#SBATCH --partition=gpu          # TODO: confirm partition with `sinfo` on Milan cluster
#SBATCH --gres=gpu:a100:1        # or gpu:ampere:1 depending on cluster
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=logs/run_all-%j.out
#SBATCH --error=logs/run_all-%j.err

set -euo pipefail
cd ~/jusdefv2
mkdir -p logs outputs/checkpoints outputs/logs

bash ./scripts/run_all_ampere.sh

#!/bin/bash
# Generic SLURM wrapper:  sbatch [--account=<allocation>] scripts/slurm_job.sh <script.py> [args...]
#   e.g.  sbatch scripts/slurm_job.sh train.py --data_dir /path/to/RadioMapSeer/
# Set VENV=/path/to/venv to activate an environment with requirements.txt installed.
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
set -e
if [ -n "${VENV:-}" ]; then source "$VENV/bin/activate"; fi
cd "${SLURM_SUBMIT_DIR:-.}"
echo "node=$(hostname)"
nvidia-smi -L || echo "WARNING: no GPU visible on this node"
python "$@"

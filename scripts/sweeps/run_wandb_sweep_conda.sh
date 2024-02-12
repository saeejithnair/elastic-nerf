#!/bin/bash

# Usage:
# ./scripts/sweeps/run_wandb_sweep_conda.sh 0 guoz0t2b [conda_env_name] [wandb_project_name]

# Can optionally pass in a conda environment name as the third argument.
CONDA_ENV_NAME=${3:-elastic_nerf}
WANDB_PROJ_NAME=${4:-"elastic-nerf"}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

set -o allexport
source "${SCRIPT_DIR}/../../.env"
set +o allexport

# Initialize Conda in bash script
source /home/$USER/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV_NAME"


export CUDA_VISIBLE_DEVICES=$1
wandb agent $WANDB_ENTITY/$WANDB_PROJ_NAME/$2

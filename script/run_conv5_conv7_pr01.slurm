#!/bin/bash
#SBATCH -J fld_c5c7
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 00:30:00
#SBATCH -o script/logs/conv5_conv7_pr01_%j.out
#SBATCH -e script/logs/conv5_conv7_pr01_%j.err
set -euxo pipefail

# Use the env's interpreter directly — avoids brittle `conda activate` under set -e.
ENV_PY="$DATA/conda/envs/folding-yolo/bin/python"

cd /data/fs201221/dw23251/project/master_thesis_folding

"$ENV_PY" -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
"$ENV_PY" script/run_conv5_conv7_pr01.py

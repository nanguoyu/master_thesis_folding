#!/bin/bash
#SBATCH -J fld_evlsw
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00
#SBATCH -o script/logs/eval_conv5_conv7_sweep_%j.out
#SBATCH -e script/logs/eval_conv5_conv7_sweep_%j.err
set -euxo pipefail
cd /data/fs201221/dw23251/project/master_thesis_folding
$DATA/conda/envs/folding-yolo/bin/python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
$DATA/conda/envs/folding-yolo/bin/python script/eval_conv5_conv7_sweep.py

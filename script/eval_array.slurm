#!/bin/bash
#SBATCH -J fld_evlarr
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00
#SBATCH --array=0-1
#SBATCH -o script/logs/eval_array_%A_%a.out
#SBATCH -e script/logs/eval_array_%A_%a.err
set -euxo pipefail
cd /data/fs201221/dw23251/project/master_thesis_folding

CONFIGS=(yolo_conv5_conv7 yolo_conv4_to_conv8)
CFG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
echo "Eval array task $SLURM_ARRAY_TASK_ID -> config=$CFG"

$DATA/conda/envs/folding-yolo/bin/python script/eval_one_config.py "$CFG"

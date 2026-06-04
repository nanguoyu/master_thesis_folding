#!/bin/bash
#SBATCH -J fld_arr
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00
#SBATCH --array=0-7
#SBATCH -o script/logs/sweep_array_%A_%a.out
#SBATCH -e script/logs/sweep_array_%A_%a.err
set -euxo pipefail
cd /data/fs201221/dw23251/project/master_thesis_folding

# Task ID -> (config, PR). 0-3 = conv5_conv7, 4-7 = conv4_to_conv8.
CONFIGS=(yolo_conv5_conv7 yolo_conv5_conv7 yolo_conv5_conv7 yolo_conv5_conv7 \
         yolo_conv4_to_conv8 yolo_conv4_to_conv8 yolo_conv4_to_conv8 yolo_conv4_to_conv8)
PRS=(0.1 0.3 0.5 0.7 0.1 0.3 0.5 0.7)

CFG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
PR=${PRS[$SLURM_ARRAY_TASK_ID]}
echo "Array task $SLURM_ARRAY_TASK_ID -> config=$CFG PR=$PR"

$DATA/conda/envs/folding-yolo/bin/python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
$DATA/conda/envs/folding-yolo/bin/python script/run_one_pr.py "$CFG" "$PR"

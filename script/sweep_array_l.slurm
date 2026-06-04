#!/bin/bash
#SBATCH -J fld_arr_l
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00
#SBATCH --array=0-3
#SBATCH -o script/logs/sweep_array_l_%A_%a.out
#SBATCH -e script/logs/sweep_array_l_%A_%a.err
set -euxo pipefail
cd /data/fs201221/dw23251/project/master_thesis_folding

# 4-task array: yolov8l x {conv4_to_conv8} x PR={0.1, 0.3, 0.5, 0.7}
CONFIGS=(yolo_conv4_to_conv8_l yolo_conv4_to_conv8_l yolo_conv4_to_conv8_l yolo_conv4_to_conv8_l)
PRS=(0.1 0.3 0.5 0.7)
WEIGHTS=weights/yolov8l.pt

CFG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
PR=${PRS[$SLURM_ARRAY_TASK_ID]}
echo "Array task $SLURM_ARRAY_TASK_ID -> config=$CFG PR=$PR weights=$WEIGHTS"

$DATA/conda/envs/folding-yolo/bin/python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
$DATA/conda/envs/folding-yolo/bin/python script/run_one_pr.py "$CFG" "$PR" "$WEIGHTS"

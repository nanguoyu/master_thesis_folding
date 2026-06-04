#!/bin/bash
#SBATCH -J fld_evl_l
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00
#SBATCH -o script/logs/eval_array_l_%j.out
#SBATCH -e script/logs/eval_array_l_%j.err
set -euxo pipefail
cd /data/fs201221/dw23251/project/master_thesis_folding

# Single eval task for the yolov8l conv4_to_conv8 sweep.
echo "Eval yolo_conv4_to_conv8_l (baseline=weights/yolov8l.pt)"

$DATA/conda/envs/folding-yolo/bin/python script/eval_one_config.py yolo_conv4_to_conv8_l weights/yolov8l.pt

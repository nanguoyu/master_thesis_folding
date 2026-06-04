"""
PR sweep for yolo_conv5_conv7.json on musica_lnz.

Runs run_folding_experiment for PR in {0.3, 0.5, 0.7} sequentially.
Each PR produces both a no-repair and a forward-pass-repair checkpoint
under weights/{without_repair_fix,forward_pass_repair_fix}/<PR>/.
PR=0.1 was already done by run_conv5_conv7_pr01.py — not re-run here.
"""
import os
import sys

# Repo root on sys.path so `import folding_main` works when invoked from script/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import folding_main

PRS = [0.3, 0.5, 0.7]
CONFIG = "config_folding/yolo_conv5_conv7.json"
CALIB_DS = "coco/images/val2017"
N_CALIB = 1000

for pr in PRS:
    folding_main.u_cache = {}  # match folding_main.main()'s pattern
    print(f"\n\n========== PR = {pr} ==========\n", flush=True)
    folding_main.run_folding_experiment(
        weights_path="weights/yolov8m.pt",
        config_path=CONFIG,
        pairing_rate=pr,
        number_calib_images=N_CALIB,
        do_repair=True,
        calib_ds=CALIB_DS,
    )
    print(f"\n===== PR={pr} done =====\n", flush=True)

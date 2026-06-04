"""
Single-experiment driver: fold one config at one PR.

Invoked by sweep_array.slurm via:  python script/run_one_pr.py <config_name> <PR> [<weights_path>]
Where <config_name> is e.g. "yolo_conv5_conv7" (no path, no .json).
The optional <weights_path> defaults to "weights/yolov8m.pt" for back-compat
with existing m-model sweeps; pass "weights/yolov8l.pt" for the l sweep.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import folding_main

if len(sys.argv) not in (3, 4):
    raise SystemExit("Usage: run_one_pr.py <config_name> <PR> [<weights_path>]")

config_name = sys.argv[1]
pr = float(sys.argv[2])
weights_path = sys.argv[3] if len(sys.argv) > 3 else "weights/yolov8m.pt"

print(f"\n========== {config_name}  PR={pr}  weights={weights_path} ==========\n", flush=True)
folding_main.u_cache = {}  # match folding_main.main() pattern
folding_main.run_folding_experiment(
    weights_path=weights_path,
    config_path=f"config_folding/{config_name}.json",
    pairing_rate=pr,
    number_calib_images=1000,
    do_repair=True,
    calib_ds="coco/images/val2017",
)
print(f"\n===== {config_name} PR={pr} done =====\n", flush=True)

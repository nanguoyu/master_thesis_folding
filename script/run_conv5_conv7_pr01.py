"""Driver for the first cluster folding run.

Calls run_folding_experiment once with the simple conv5/conv7 plan at
PR=0.1, 1000 calibration images, repair enabled. Calibration uses COCO
val2017 (no train2017 available on the cluster; COCOImageFolder is
label-free, so any image directory works).
"""
import os
import sys

# Make the repo root importable when run as `python script/run_conv5_conv7_pr01.py`.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from folding_main import run_folding_experiment


def main() -> None:
    run_folding_experiment(
        weights_path="weights/yolov8m.pt",
        config_path="config_folding/yolo_conv5_conv7.json",
        pairing_rate=0.1,
        number_calib_images=1000,
        do_repair=True,
        calib_ds="coco/images/val2017",
    )


if __name__ == "__main__":
    main()

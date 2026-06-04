"""Eval driver: compare baseline vs two folded YOLOv8m variants on COCO val2017.

Runs `model.val()` on three checkpoints (baseline, folded-no-repair,
folded-forward-pass-repair) and dumps a small json summary plus a printed
table. The folded checkpoints come from the conv5/conv7 PR=0.1 folding job
(slurm 53398).

Standalone script — not part of the folding pipeline. Companion slurm file:
`script/eval_conv5_conv7_pr01.slurm`.
"""
import json
import os
import sys

# Make the repo root importable when run as `python script/eval_conv5_conv7_pr01.py`.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ultralytics import YOLO

DATA_YAML = os.path.join(REPO_ROOT, "coco", "coco.yaml")
IMGSZ = 640

MODELS = [
    ("baseline",         "weights/yolov8m.pt"),
    ("folded_no_repair", "weights/without_repair_fix/0.1/yolo_conv5_conv7_folded_without_repair_fix.pt"),
    ("folded_fp_repair", "weights/forward_pass_repair_fix/0.1/yolo_conv5_conv7_folded_forward_pass_repair_fix_calib1000.pt"),
]


def main() -> None:
    results = []
    for tag, rel_path in MODELS:
        abs_path = os.path.join(REPO_ROOT, rel_path)
        print(f"\n=== {tag}: {rel_path} ===", flush=True)
        model = YOLO(abs_path)
        r = model.val(
            data=DATA_YAML,
            split="val",
            imgsz=IMGSZ,
            verbose=False,
            save_json=False,
            plots=False,
        )
        mp, mr = float(r.box.mp), float(r.box.mr)
        f1 = 2.0 * mp * mr / (mp + mr + 1e-6)
        n_params = sum(p.numel() for p in model.model.parameters())
        results.append({
            "tag": tag,
            "path": rel_path,
            "params": n_params,
            "mp": mp,
            "mr": mr,
            "f1": f1,
            "map50":   float(r.box.map50),
            "map5095": float(r.box.map),
        })

    out_dir = os.path.join(REPO_ROOT, "results_save", "save_statistics_eval")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "conv5_conv7_pr01.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n=== Saved: {out_path} ===")

    # Comparison table.
    print(f"\n{'tag':<22}{'params':>14}{'F1':>8}{'mAP50':>8}{'mAP50-95':>10}")
    for r in results:
        print(f"{r['tag']:<22}{r['params']:>14,}{r['f1']:>8.4f}{r['map50']:>8.4f}{r['map5095']:>10.4f}")


if __name__ == "__main__":
    main()

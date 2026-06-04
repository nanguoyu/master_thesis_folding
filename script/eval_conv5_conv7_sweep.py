"""
Eval the full PR sweep for conv5/conv7 folding against COCO val2017.

Compares the baseline yolov8m to folded-no-repair and folded-forward-pass-repair
checkpoints at PR in {0.1, 0.3, 0.5, 0.7}. Skips any model whose .pt is missing
(useful if a PR hasn't been folded yet).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO

REPO = "/data/fs201221/dw23251/project/master_thesis_folding"
DATA_YAML = f"{REPO}/coco/coco.yaml"
IMGSZ = 640

PRS = ["0.1", "0.3", "0.5", "0.7"]
MODELS = [("baseline", "weights/yolov8m.pt")]
for pr in PRS:
    MODELS.append((
        f"folded_no_repair_pr{pr}",
        f"weights/without_repair_fix/{pr}/yolo_conv5_conv7_folded_without_repair_fix.pt",
    ))
    MODELS.append((
        f"folded_fp_repair_pr{pr}",
        f"weights/forward_pass_repair_fix/{pr}/yolo_conv5_conv7_folded_forward_pass_repair_fix_calib1000.pt",
    ))


def main():
    results = []
    for tag, rel_path in MODELS:
        path = os.path.join(REPO, rel_path)
        if not os.path.exists(path):
            print(f"\n=== SKIP {tag}: {rel_path} (missing) ===", flush=True)
            continue
        print(f"\n=== {tag}: {rel_path} ===", flush=True)
        model = YOLO(path)
        r = model.val(
            data=DATA_YAML, split="val", imgsz=IMGSZ,
            verbose=False, save_json=False, plots=False,
        )
        mp, mr = float(r.box.mp), float(r.box.mr)
        f1 = 2 * (mp * mr) / (mp + mr + 1e-6)
        n_params = sum(p.numel() for p in model.model.parameters())
        results.append({
            "tag": tag,
            "path": rel_path,
            "params": n_params,
            "mp": mp, "mr": mr, "f1": f1,
            "map50": float(r.box.map50),
            "map5095": float(r.box.map),
        })

    out_dir = f"{REPO}/results_save/save_statistics_eval"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/conv5_conv7_pr_sweep.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n=== Saved: {out_path} ===")

    print(f"\n{'tag':<30}{'params':>14}{'F1':>8}{'mAP50':>8}{'mAP50-95':>10}")
    for r in results:
        print(f"{r['tag']:<30}{r['params']:>14,}{r['f1']:>8.4f}{r['map50']:>8.4f}{r['map5095']:>10.4f}")


if __name__ == "__main__":
    main()

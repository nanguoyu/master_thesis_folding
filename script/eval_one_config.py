"""
Eval the full PR sweep for one config against COCO val2017.

Invoked by eval_array.slurm via:  python script/eval_one_config.py <config_name> [<baseline_weights>]
Writes results_save/save_statistics_eval/<config_name>_pr_sweep_b2.json
(_b2 suffix marks the run as post-B2-fix, so it doesn't clobber pre-fix snapshots.)
The optional <baseline_weights> defaults to "weights/yolov8m.pt" for back-compat;
pass "weights/yolov8l.pt" for the l sweep. The folded checkpoint paths already
key off config_name, so they route correctly when config_name = yolo_conv4_to_conv8_l.
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


def main():
    if len(sys.argv) not in (2, 3):
        raise SystemExit("Usage: eval_one_config.py <config_name> [<baseline_weights>]")
    config_name = sys.argv[1]
    baseline_weights = sys.argv[2] if len(sys.argv) > 2 else "weights/yolov8m.pt"

    models = [("baseline", baseline_weights)]
    for pr in PRS:
        models.append((
            f"folded_no_repair_pr{pr}",
            f"weights/without_repair_fix/{pr}/{config_name}_folded_without_repair_fix.pt",
        ))
        models.append((
            f"folded_fp_repair_pr{pr}",
            f"weights/forward_pass_repair_fix/{pr}/{config_name}_folded_forward_pass_repair_fix_calib1000.pt",
        ))

    results = []
    for tag, rel in models:
        path = os.path.join(REPO, rel)
        if not os.path.exists(path):
            print(f"\n=== SKIP {tag}: {rel} (missing) ===", flush=True)
            continue
        print(f"\n=== {tag}: {rel} ===", flush=True)
        model = YOLO(path)
        r = model.val(data=DATA_YAML, split="val", imgsz=IMGSZ,
                      verbose=False, save_json=False, plots=False)
        mp, mr = float(r.box.mp), float(r.box.mr)
        f1 = 2 * (mp * mr) / (mp + mr + 1e-6)
        n_params = sum(p.numel() for p in model.model.parameters())
        results.append({
            "tag": tag, "path": rel,
            "params": n_params,
            "mp": mp, "mr": mr, "f1": f1,
            "map50": float(r.box.map50),
            "map5095": float(r.box.map),
        })

    out_dir = f"{REPO}/results_save/save_statistics_eval"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{config_name}_pr_sweep_b2.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n=== Saved: {out_path} ===")

    print(f"\n{'tag':<30}{'params':>14}{'F1':>8}{'mAP50':>8}{'mAP50-95':>10}")
    for r in results:
        print(f"{r['tag']:<30}{r['params']:>14,}{r['f1']:>8.4f}"
              f"{r['map50']:>8.4f}{r['map5095']:>10.4f}")


if __name__ == "__main__":
    main()

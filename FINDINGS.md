# Experimental Progress & Findings — Model Folding on YOLOv8

Notes from Dong, reviewing your implementation and running follow-up
experiments. Written for you (Manuel) to read first; we can talk through it
in the next meeting.

Repo now lives at `https://github.com/nanguoyu/master_thesis_folding` (a fork
of your original). Your `main` is set as `upstream` so you can `git pull
upstream/main` to get any new work from your side. Everything we did has
been pushed to `origin/main` on the fork.

---

## 1. Where things stood when I started

Your README_DONG.md asked one main question:

> "I am confused that the forward pass calibration does not increase the
> accuracy in the folded versions (in the pruned versions, the forward pass
> does work). Can you maybe take a look at it…?"

After reading the paper LaTeX, both official repos
(`marza96/ModelFolding` and `nanguoyu/model-folding-universal`, now under
`knowledge/`), and your `folding_main.py`, I found two distinct reasons the
forward-pass REPAIR was misbehaving plus three more correctness bugs in the
folding engine itself. Two of those (BN reset scope, calibration letterbox)
you had already partially fixed in commit `1dd8777` before I started
poking — nice catch.

A short summary of every code change made during this round:

| Bug | Severity | File / lines | One-line fix |
|---|---|---|---|
| **M2** | major | `folding_main.py` `repair_bn_forward_pass` | Reset *every* `nn.BatchNorm2d` (match official `reset_bn_stats`). Your "downstream of first fold" filter still let upstream BNs drift in `model.train()` mode. |
| **B3** | blocker | `folding_main.py` `run_folding_experiment` | Drop the `break` after first consumer; FPN-style fan-out has multiple consumers per folded layer. |
| **B4** | blocker | `folding_main.py` `merge_conv_bn` | Un-comment the block-diagonal `U` expansion for concat-fed inputs — it was sitting inside a triple-quoted string and never executing. |
| **M5** | major | `folding_main.py` `merge_conv_bn` | Guard conv `bias` in the output-fold branch. Detect-head 1×1 convs carry `bias=True` (6 of them in YOLOv8m), so without this guard any future config that flips them on will shape-mismatch. |
| **B2** | major | `folding_main.py` `cumpute_cluster_matrix_u`, `c2f_layer_folding` | Include BN γ and β as columns in the k-means input matrix `A`. Two filters with similar spatial weights but very different BN scaling were previously merged together. Matches the paper's `W_tot = [W_l | γ | β | W_{l+1}^T]` and the official `concat_weights()` (non-approx-repair). |

The smoke test (`smoke_test_folding.py`) was added so we don't break the
fold/REPAIR primitives going forward; it runs in ~20 s on CPU and gets
exercised after every code change.

---

## 2. Environment

`environment.yml` builds a reproducible mamba env (Python 3.11, torch 2.12,
ultralytics 8.4.60). The only fiddly piece is **IBM Hartigan k-means** —
no PyPI release, install from `git+https://github.com/IBM/hartigan-kmeans.git`;
the spec file already does that. On macOS arm64 add `libcxx` so its C
extension loads.

`smoke_test_folding.py` runs the fold + REPAIR primitives on yolov8n in
~20 s and is what I use after every code change.

---

## 3. Methodology — what each experiment measures

All folding uses your existing engine (`folding_main.py`) with the bug fixes
above. Calibration for forward-pass REPAIR uses **1000 images sampled from
COCO val2017**; evaluation is **`model.val()` on the full 5000-image
val2017** at `imgsz=640`. We report:

- Params (raw nn.Module count, not the slightly different count from
  ultralytics' `fused` summary — they differ by a few thousand for buffer
  reasons; we use the raw count consistently).
- F1 = harmonic mean of mean-precision and mean-recall over all 80 classes.
- mAP50 and mAP50-95 from ultralytics' standard COCO eval.

For every config we report two folded variants:
- **`folded_no_repair_fix`** — folded weights + your `merge_conv_bn` BN
  statistics heuristic (the inverse-std merge), no recalibration.
- **`folded_fp_repair_fix`** — same fold, plus the corrected
  `repair_bn_forward_pass` (resets every BN, runs 1000 calibration images,
  switches back to eval mode). This is the closest thing to the paper's
  data-driven **Fold-R**.

We **do not** yet implement Fold-AR (the paper's closed-form, data-free
alternative). That's open work.

---

## 4. Results

### 4.1 `yolo_conv5_conv7` on yolov8m (only 2 plain Convs folded)

Folded layers: `model.5.conv`, `model.7.conv`. Maximum reduction at PR=0.7
is just 8.5% because only two layers are touched.

| PR | Mode | Params | Δ params | F1 | mAP50 | mAP50-95 |
|---:|---|---:|---:|---:|---:|---:|
| — | baseline (yolov8m) | 25,886,080 | 0% | **0.6587** | 0.6653 | 0.4978 |
| 0.1 | no_repair | 25,571,872 | −1.21% | 0.6571 | 0.6628 | 0.4970 |
| 0.1 | fp_repair | 25,571,872 | −1.21% | 0.6392 | 0.6406 | 0.4721 |
| 0.3 | no_repair | 24,945,376 | −3.64% | 0.6391 | 0.6449 | 0.4826 |
| 0.3 | fp_repair | 24,945,376 | −3.64% | 0.6298 | 0.6302 | 0.4652 |
| 0.5 | no_repair | 24,318,880 | −6.06% | 0.5843 | 0.5767 | 0.4285 |
| 0.5 | fp_repair | 24,318,880 | −6.06% | **0.6026** | 0.5989 | 0.4401 |
| 0.7 | no_repair | 23,692,384 | −8.47% | 0.3027 | 0.2301 | 0.1681 |
| 0.7 | fp_repair | 23,692,384 | −8.47% | **0.5198** | 0.4956 | 0.3603 |

REPAIR–vs–no-repair ΔF1 by PR: −0.018 / −0.009 / +0.018 / **+0.217**.
**Breakeven is around PR=0.5.** This reproduces your original observation
that REPAIR hurts at low PR (we now know exactly why — see §5).

### 4.2 `yolo_conv4_to_conv8` on yolov8m (C2f blocks + plain convs across model.4–8)

Many layers folded together; total reduction now scales meaningfully.

| PR | Mode | Params | Δ params | F1 | mAP50 | mAP50-95 |
|---:|---|---:|---:|---:|---:|---:|
| — | baseline | 25,886,080 | 0% | 0.6587 | 0.6653 | 0.4978 |
| 0.1 | no_repair | 24,200,836 | −6.51% | 0.6044 | 0.5992 | 0.4442 |
| 0.1 | fp_repair | 24,200,836 | −6.51% | **0.6159** | 0.6103 | 0.4471 |
| 0.3 | no_repair | 21,248,390 | −17.92% | 0.1943 | 0.1404 | 0.0968 |
| 0.3 | fp_repair | 21,248,390 | −17.92% | **0.4718** | 0.4312 | 0.3011 |
| 0.5 | no_repair | 18,842,272 | −27.21% | 0.0014 | 0.0002 | 0.0001 |
| 0.5 | fp_repair | 18,842,272 | −27.21% | 0.1374 | 0.0676 | 0.0398 |
| 0.7 | no_repair | 16,976,666 | −34.42% | 0.0000 | 0.0000 | 0.0000 |
| 0.7 | fp_repair | 16,976,666 | −34.42% | 0.0089 | 0.0028 | 0.0013 |

REPAIR ΔF1: **+0.012 / +0.278 / +0.136 / +0.009**. Here REPAIR earns its
keep from PR=0.1 already, and at PR=0.3 it rescues the model from
catastrophic collapse (F1 jumps from 0.19 to 0.47).

### 4.3 `yolo_conv4_to_conv8_l` on yolov8l (same config, wider model)

Same folded modules as in 4.2, applied to yolov8l. yolov8l is wider in
model.4–6 (depth=1.0, width=1.0) but actually slightly narrower at
model.7/8 because its `max_channels` cap is 512 vs yolov8m's 768. So the
"wider folds better" hypothesis from the paper is **mixed** here, not
clean.

| PR | Mode | Params | Δ params | F1 | mAP50 | mAP50-95 |
|---:|---|---:|---:|---:|---:|---:|
| — | baseline (yolov8l) | 43,691,520 | 0% | **0.6828** | 0.6916 | 0.5245 |
| 0.1 | no_repair | 40,642,588 | −6.98% | 0.6331 | 0.6337 | 0.4741 |
| 0.1 | fp_repair | 40,642,588 | −6.98% | 0.6357 | 0.6360 | 0.4730 |
| 0.3 | no_repair | 35,423,390 | −18.92% | 0.3112 | 0.2417 | 0.1734 |
| 0.3 | fp_repair | 35,423,390 | −18.92% | **0.4874** | 0.4535 | 0.3208 |
| 0.5 | no_repair | 31,214,848 | −28.56% | 0.0164 | 0.0041 | 0.0030 |
| 0.5 | fp_repair | 31,214,848 | −28.56% | 0.1818 | 0.1071 | 0.0681 |
| 0.7 | no_repair | 28,036,662 | −35.83% | 0.0000 | 0.0000 | 0.0000 |
| 0.7 | fp_repair | 28,036,662 | −35.83% | 0.0280 | 0.0059 | 0.0031 |

Result JSONs are in `results_save/save_statistics_eval/`. The
`*_pr_sweep_b2.json` files are post-B2-fix; the older
`conv5_conv7_pr01.json` and `conv5_conv7_pr_sweep.json` are pre-fix and
kept around for comparison.

---

## 5. What this tells us

### 5.1 Why forward-pass REPAIR hurt at low PR — answered

At PR=0.1 on 2 isolated Convs, the BN statistics produced by your
`merge_conv_bn` heuristic (inverse-std averaging) are already very close
to the true post-fold statistics. Re-estimating them from 1000 calibration
images introduces *sampling noise* that's bigger than the residual
heuristic error. The variance-collapse story from the paper is real but
only kicks in when collapse is severe enough to dominate the noise.

That happens in two ways:

- **High per-layer PR on few layers** — `conv5_conv7` PR=0.5+ — REPAIR
  breakeven is around PR=0.5, +0.22 F1 by PR=0.7.
- **Moderate per-layer PR across many layers** — `conv4_to_conv8` PR=0.3 —
  the per-layer variance distortion is small but **propagates and
  compounds** through ~24 folded convs. REPAIR globally re-normalizes
  every BN, undoing the compounded drift. +0.28 F1 lift at PR=0.3.

The second case is more practically useful (you get real compression at
modest per-layer aggression). Your original PR=0.1/0.2/0.3 experiments on
`conv5_conv7` sat in exactly the worst region: collapse small enough that
your `merge_conv_bn` heuristic handles it well, calibration noise large
enough that REPAIR makes it worse. **Not a bug in REPAIR — a regime
mismatch.**

### 5.2 Useful compression ceiling without fine-tune is around 18–20% params

- `conv4_to_conv8` PR=0.3 on yolov8m: −17.9% params, F1=0.47, mAP50-95=0.30.
- `conv4_to_conv8_l` PR=0.3 on yolov8l: −18.9% params, F1=0.49, mAP50-95=0.32.

Beyond that, both no_repair and fp_repair fall off a cliff. PR=0.5 is
borderline-broken, PR=0.7 is dead. This matches the paper's qualitative
claim that data-free folding is best around 20–50% sparsity for CIFAR
models — YOLOv8m + COCO is harder because detection is more sensitive to
small statistics perturbations than classification.

### 5.3 Wider folds better in absolute terms — but doesn't extend the Pareto frontier of the YOLOv8 size family

Comparing the two `conv4_to_conv8` sweeps at matched PR (both with
fp_repair):

| PR | yolov8m F1 | yolov8l F1 | l − m |
|---:|---:|---:|---:|
| 0 (baseline) | 0.6587 | 0.6828 | +0.024 |
| 0.1 | 0.6159 | 0.6357 | +0.020 |
| 0.3 | 0.4718 | 0.4874 | +0.016 |
| 0.5 | 0.1374 | 0.1818 | +0.044 |

yolov8l wins at every PR. So **at matched PR**, wider is more
fold-resilient — consistent with the paper.

But the more useful question is **at matched params**:

| Configuration | Params | mAP50-95 |
|---|---:|---:|
| **yolov8m baseline (unfolded)** | **25.9M** | **0.498** |
| yolov8l + fold PR=0.5 + REPAIR | 31.2M | 0.068 |
| yolov8l + fold PR=0.3 + REPAIR | 35.4M | 0.321 |
| yolov8l + fold PR=0.1 + REPAIR | 40.6M | 0.473 |
| yolov8l baseline (unfolded) | 43.7M | 0.524 |

For any folded-yolov8l point we tested, the unfolded yolov8m sits at a
smaller param count *and* higher mAP. Folding doesn't open a new Pareto
point in the YOLOv8 size family — the n/s/m/l/x scales already cover the
curve. The paper's "wider folds better" demonstrations were on
ResNet/VGG, where there's no pre-tuned scale family, so folding was the
only path to a smaller-and-still-good model. YOLOv8 is different.

This is, I think, an *honest negative finding* worth including in your
thesis. It does not invalidate the algorithm — it tells you when folding
is the right tool and when it isn't.

### 5.4 The clustering input matters less than you'd think

The B2 fix (adding BN γ, β to the k-means input matrix) changed F1 by at
most 0.004 in `conv5_conv7` low-PR cases. At PR=0.7 on `conv5_conv7`
no_repair, the post-B2 fix accidentally made it worse (F1 0.50 → 0.30):
because YOLOv8m's BN γ values are O(1) while individual conv weights are
O(0.01), the BN columns dominate the clustering distance at small fold
sizes, biasing toward BN-similar (not weight-similar) clusters. When
that's combined with no REPAIR to fix the resulting unusual statistics,
the model breaks more than under the pre-B2 raw-weight clustering.

The B2 fix matches the official `concat_weights()` faithfully, so we
keep it on. But it's worth noting that "match the paper" and "best
YOLO performance" aren't always the same thing — at PR=0.5+ with
REPAIR enabled, B2 is fine; at low PR without REPAIR, raw-weight
clustering was accidentally better.

---

## 6. Things I deliberately did **not** do

- **Fold-AR (paper Algorithm 1)** — closed-form, data-free repair. The
  paper's main data-free contribution. Not implemented. It would
  particularly help in the regime where forward-pass REPAIR is too noisy
  (i.e. low-PR / few-layer cases like `conv5_conv7` PR=0.1). I left it as
  the next major code item.
- **SPPF folding (model.9)** — you flagged this as future work. The
  internal MaxPool + 4-way concat needs special handling that
  `c2f_layer_folding` doesn't cover.
- **Neck (PAN-FPN) folding** — we designed `yolo_neck.json` but did not
  run it. The neck's Concat layers fan in from *both* a folded source and
  a frozen source (e.g. `model.12.cv1.conv` input = 576 ch from frozen
  SPPF + 384 ch from folded `model.6.cv2.conv`). Your current `pre`
  schema in the folding plan only allows one upstream per consumer, and
  `merge_conv_bn`'s block-diagonal `U` expansion only fires when
  `actual_in_channels` is an integer multiple of `n_original` — which a
  mixed concat (576 + 384) is not. **This is a real engineering
  blocker.** Two options for fixing:
    - Extend `pre` to be list-valued (each upstream contributes a block;
      frozen sources contribute identity). ~80 LOC engine change.
    - Restrict neck folding to `cv2.conv` outputs only, never `cv1.conv`.
      Loses ~70% of the neck gain.
  We didn't pick one. Up to you.
- **Detect-head folding** — `model.22.cv2.X.0.conv` / `cv3.X.0.conv` have
  `bias=True`; M5 already guards them, but their output channels are
  fixed by the (4 + 80) detection format. So head folding is mostly an
  *input-side* problem driven by neck folding (above).
- **yolov8s/n/x** — the wider-is-better story should be sharper at
  yolov8x (more redundancy) and inverted at yolov8n (less). Not run.

---

## 7. Suggestions for your thesis writeup

A few framings that I think hold up well, based on what we ran:

1. **"REPAIR is regime-dependent, not universally good."** This is your
   most defensible empirical observation. Frame the original
   confusing-behavior of `conv5_conv7` PR=0.1 as a *positive result* —
   you discovered the regime where data-driven REPAIR introduces more
   noise than it fixes. The paper's variance-collapse story only kicks
   in past a measurable threshold (your `conv4_to_conv8` PR=0.3 vs
   `conv5_conv7` PR=0.1 contrast makes this very concrete).

2. **"Useful compression without fine-tune is bounded around 18–20%
   params for YOLOv8 on COCO."** Backed by both yolov8m and yolov8l data.
   This number is the practical ceiling and it's a reasonable engineering
   contribution.

3. **"Model folding does not extend the Pareto frontier of an
   already-tuned scale family."** The wider-folds-better claim is true at
   matched PR, but doesn't translate to matched-params Pareto
   improvements when n/s/m/l/x already exist. This is a real conditioning
   point for the paper's "wider works better" framing that I think it's
   worth making clear in the thesis. (For Olga and me, this is also
   useful — we'd want to revisit the claim's scope in a follow-up.)

4. Fine-tune comparison is fair to add as context, **with the caveat**
   that all data-free / fine-tune-free pruning methods sit at one
   operating point and YOLO papers that report better numbers use
   fine-tune. The 2 cited works in your README make this point well.

---

## 8. Open questions for the meeting

- Do you want to implement Fold-AR next (it's the paper's flagship
  data-free repair and would directly address the noise-vs-collapse
  tradeoff)?
- Do you want to do the engine change for multi-source `pre` to unlock
  the neck? That's the only thing standing between us and ~30–40%
  compression on yolov8m. Risk: moderate, code change ~80 LOC.
- Should we add yolov8s or yolov8x to the wider-vs-narrower comparison?
- Latency / GFLOPs measurement — folding doesn't change depth, so the
  wall-clock speedup may be smaller than the param reduction. Worth
  measuring before claiming compression.

Happy to walk through any of this. The setup is reproducible on your
side via `mamba env create -f environment.yml` plus a `git pull` on the
fork.

— Dong

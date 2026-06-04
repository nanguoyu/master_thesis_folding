# Notes on the folding code and experiments

Manuel,

I took a longer look at the code and ran a few experiments to dig into
what your `README_DONG.md` asked about. Wrote it down so we can go through
it together.

Everything is on a fork at `nanguoyu/folding-on-yolo`. Your repo is set
as `upstream` here; I haven't opened a pull request back to you, the work
just sits on the fork. Pull from there if you want any of the changes
locally.

## What I changed in the code

Five places. Two of them — the BN reset scope and the calibration
letterboxing — you had already partly fixed in `1dd8777`, nice catch on
both.

| Bug | What was wrong | Fix |
|---|---|---|
| **M2** | `repair_bn_forward_pass` only reset the BN layers downstream of the first folded conv. Then `model.train()` puts the whole model in train mode and the other ~50 BNs silently drift their running stats under default momentum. | Reset every `BatchNorm2d` in the model, like both official repos do. |
| **B3** | `run_folding_experiment` looked for the first consumer of a folded layer and broke. YOLO's FPN has cases where one source feeds multiple consumers (e.g. a C2f's `cv2.conv` output goes to both the next stride-2 Conv and a head Concat). | Drop the `break`; apply input-fold to every consumer. |
| **B4** | The block-diagonal `U` expansion in `merge_conv_bn` was sitting inside a `"""..."""` block and never executing. It's needed for concat-fed inputs where one source feeds multiple Conv outputs. | Un-comment the block. |
| **M5** | Output-side fold rewrote `conv.weight` but never touched `conv.bias`. YOLOv8m's Detect head has six 1×1 convs with `bias=True`. | One-line addition: fold bias through the same projection matrix. |
| **B2** | k-means input matrix `A` was `[W_l \| W_{l+1}^T]`. Paper and official `concat_weights` also include `γ` and `β`. Two filters with similar spatial weights but very different BN scaling were being clustered together. | Pass the BN module into `cumpute_cluster_matrix_u` and `c2f_layer_folding`; add γ and β as columns of `A`. |

There's a `smoke_test_folding.py` at the repo root now. It runs the
cluster + fold + REPAIR primitives on yolov8n in about 20 seconds on
CPU. I ran it after every code change. If you mess with `folding_main.py`,
it'll catch the obvious breakage.

## Environment

`environment.yml` builds a reproducible mamba env. The one fiddly thing
is the IBM Hartigan k-means: no PyPI release, so install it from
`git+https://github.com/IBM/hartigan-kmeans.git`. The spec file does
this for you. On macOS arm64 you also need `libcxx` so the C extension
links. Linux is smoother.

## What I ran

Three configurations, four PR values each (0.1, 0.3, 0.5, 0.7), with
and without forward-pass REPAIR. Calibration is 1000 random images from
COCO val2017; evaluation is `model.val()` on the full 5000-image val2017
at `imgsz=640`. The JSON results live in `results_save/save_statistics_eval/`.

### `yolo_conv5_conv7` on yolov8m

Only model.5.conv and model.7.conv get folded. Even at PR=0.7 the total
reduction is just 8.5%.

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

ΔF1 of REPAIR vs no-repair, by PR: −0.018, −0.009, +0.018, **+0.217**.
Breakeven is around PR=0.5. This reproduces your original observation
that REPAIR hurts at low PR.

### `yolo_conv4_to_conv8` on yolov8m

Folds the whole model.4–8 stretch, including the three backbone C2f
blocks. Now we get meaningful compression.

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

ΔF1 of REPAIR vs no-repair: +0.012, **+0.278**, +0.136, +0.009. REPAIR
earns its keep starting from PR=0.1, and at PR=0.3 it rescues the model
from catastrophic collapse (F1 jumps from 0.19 to 0.47).

### Same config on yolov8l

Just point the driver at `weights/yolov8l.pt`. The engine reads channel
counts and bottleneck counts from the live module list, so yolov8l's
extra bottlenecks (n=6 vs n=4 in model.4/6, n=3 vs n=2 in model.8) get
folded automatically; no JSON edits needed. I made a copy
`yolo_conv4_to_conv8_l.json` with the right `num_channels` for
documentation, but the engine doesn't read that field.

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

## What this tells us

Three findings I think are worth talking about.

**1. Forward-pass REPAIR is regime-dependent.**

Your observation in `README_DONG.md` — that REPAIR hurt the folded
model — is correct, *in the regime you were testing*. When folding is
shallow and per-layer aggressive (PR=0.1 on two convs only), the merged
BN statistics from your `merge_conv_bn` heuristic are already very
close to the true post-fold values. Recomputing them from 1000
calibration images introduces sampling noise that's larger than the
heuristic's residual error.

REPAIR actually helps in two cases:

- *High per-layer PR on few layers* (`conv5_conv7` PR=0.5+): collapse
  becomes severe enough to dominate the noise.
- *Moderate per-layer PR across many layers* (`conv4_to_conv8` PR=0.3):
  per-layer collapse is small but compounds through ~24 folded convs;
  REPAIR re-normalizes the whole network globally.

The second case is the practically useful one. Your PR=0.1/0.2/0.3
experiments on `conv5_conv7` were sitting in the worst region for
REPAIR: not enough collapse to need it, and enough calibration noise
to make it hurt. Not a bug in REPAIR and not a bug in your code, just
a regime mismatch.

**2. The compression ceiling without fine-tune is around 18–20% params.**

Both `conv4_to_conv8` runs land in the same place at PR=0.3 with REPAIR:

| | yolov8m | yolov8l |
|---|---:|---:|
| Reduction | −17.9% | −18.9% |
| mAP50-95 | 0.301 | 0.321 |

Beyond that both no-repair and fp_repair fall off a cliff. PR=0.5 is
borderline-broken, PR=0.7 is dead. The paper claims data-free folding
works up to 50–70% sparsity on CIFAR classifiers; YOLOv8m + COCO is
harder because detection is more sensitive to per-channel statistics
than classification. Worth saying explicitly in the thesis.

**3. Wider folds better in absolute terms, but folding doesn't extend
YOLOv8's Pareto frontier.**

At matched PR, yolov8l wins at every point we tested:

| PR | yolov8m F1 (fp_repair) | yolov8l F1 (fp_repair) |
|---:|---:|---:|
| 0 | 0.6587 | 0.6828 |
| 0.1 | 0.6159 | 0.6357 |
| 0.3 | 0.4718 | 0.4874 |

This part is consistent with what the paper says about wider models.
The interesting question, though, is at matched *params*. Every
folded-l configuration we ran sits above unfolded yolov8m in params
and below it in mAP:

| Configuration | Params | mAP50-95 |
|---|---:|---:|
| yolov8m baseline | 25.9M | 0.498 |
| yolov8l fold PR=0.5 + REPAIR | 31.2M | 0.068 |
| yolov8l fold PR=0.3 + REPAIR | 35.4M | 0.321 |
| yolov8l fold PR=0.1 + REPAIR | 40.6M | 0.473 |
| yolov8l baseline | 43.7M | 0.524 |

For any folded-l point we tested, just using unfolded yolov8m would
have been smaller and more accurate. YOLOv8's n/s/m/l/x family already
covers this part of the curve well; folding doesn't open a new Pareto
point in this family.

That's not a knock on the algorithm. It's a knock on the use case.
The paper's ResNet18→ResNet50 and VGG width-sweep results sit in
domains where there's no manually-tuned scale family. Here there is
one and it's well-tuned. I think this is honest and worth saying in
the thesis writeup — it doesn't invalidate folding, it tells us when
folding is the right tool.

## A side note about the B2 clustering fix

When I added γ and β to the k-means input matrix to match the official
code, the F1 change was at most 0.004 on `conv5_conv7` at low PR. But
on PR=0.7 + no_repair, it made things noticeably worse (F1 went
from 0.50 to 0.30). Reason: YOLOv8m's γ values are O(1) while
individual conv weight entries are O(0.01), so once you add γ as a
column it dominates the clustering distance at small fold sizes.
Without REPAIR to clean up afterwards, the BN-biased clusters are
worse than the original raw-weight ones.

The fix is the right one — it matches the paper and both official
repos — but "match the paper" and "best YOLO performance" aren't
always the same thing. Probably worth a footnote in the thesis if
you compare pre-B2 and post-B2 numbers (the pre-B2 conv5_conv7 JSONs
are kept around in `results_save/save_statistics_eval/` for this).

## What I didn't get to

- **Fold-AR (paper Algorithm 1).** The closed-form, data-free repair.
  The paper's main contribution and still not implemented. It would
  specifically help in the regime where forward-pass REPAIR is too
  noisy — exactly the `conv5_conv7` low-PR cases you started with.
  Needs no calibration data so it sidesteps both the bugs you and I
  have been chasing. This is the next major code item.
- **Neck (PAN-FPN) folding.** I sketched `yolo_neck.json` but didn't
  run it. The blocker is real: Concats in the neck fan in from both a
  folded source and a frozen source. `model.12.cv1.conv`'s input is
  960 channels = 576 from the frozen SPPF plus 384 from the folded
  `model.6.cv2.conv`. The `pre` field in your JSON only allows one
  upstream per consumer, and `merge_conv_bn`'s block-diagonal U
  expansion needs `actual_in_channels % n_original == 0`. 960 / 384
  isn't integer. The clean fix is making `pre` list-valued and
  building the consumer's U as a horizontal concat of (each source's
  U, or identity for frozen sources). Roughly 80 LOC of engine
  change. If we want meaningful compression past 20%, this is the
  thing to do.
- **SPPF folding.** You already flagged this. The internal MaxPool
  plus 4-way concat needs special handling that `c2f_layer_folding`
  doesn't cover.
- **yolov8s/x.** The wider-folds-better story would be sharper at
  yolov8x; yolov8n would close the loop on the negative direction.
- **Latency / GFLOPs measurement.** Folding doesn't change depth, so
  the wall-clock speedup is probably smaller than the param reduction
  suggests. Worth measuring before claiming compression.

## Stuff to talk about

- Fold-AR next, or the neck engine extension first? Fold-AR is
  paper-faithful but only helps in regimes we now understand. The
  neck engine extension is more work but unlocks the actual
  compression ceiling. I'd lean toward neck, but it's a 4–6 hour
  change.
- Whether to test yolov8x for the thesis. Probably a few hours of
  cluster time.
- The "doesn't extend Pareto" framing — happy to discuss whether and
  how to put that in the writeup.

Talk soon,
— Dong

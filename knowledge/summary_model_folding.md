# Forget the Data and Fine-tuning! Just Fold the Network to Compress

Paper: arXiv:2502.10216 (Wang, Šikić, Thiele, Saukh). LaTeX source in
`knowledge/latex_2502.10216/`. Official code: `knowledge/ModelFolding`
(marza96, CV reference) and `knowledge/model-folding-universal` (nanguoyu, CV + LLM).

## TL;DR

Model folding is a **data-free, fine-tuning-free** structured compression method.
It merges structurally-similar channels *within a single network* by running
$k$-means clustering on weight tensors, replacing each cluster by its centroid.
The only hard part is repairing the data statistics that channel-merging
destroys (variance collapse / explosion); the paper contributes two data-free
repair schemes (Fold-AR, Fold-DIR) that match data-driven REPAIR.

## Problem & motivation

Pruning/quantization degrade accuracy and need fine-tuning on the original
training data to recover — infeasible when data is private/unavailable. Prior
data-free method IFM (Chen et al. 2023) merges channels but fails to preserve
data statistics, so it collapses at high sparsity. Goal: structured
compression with **no data and no fine-tuning** that survives high sparsity.

## Method

Model folding runs layer-by-layer in three phases: **(1) cluster, (2) merge,
(3) repair data statistics.**

### 1. Channel clustering (data-free)

For a layer with weight matrix $\mathbf{W}_l \in \mathbb{R}^{n\times m}$, reduce
$n$ output rows to $k<n$ centroids:
$$\mathbf{W}_l \approx \mathbf{U}\mathbf{M} = \mathbf{C}\mathbf{W}_l,\quad
\mathbf{C} = \mathbf{U}(\mathbf{U}^T\mathbf{U})^{-1}\mathbf{U}^T,\quad
\mathbf{M} = (\mathbf{U}^T\mathbf{U})^{-1}\mathbf{U}^T\mathbf{W}_l$$
where $\mathbf{U}\in\{0,1\}^{n\times k}$ is the cluster-assignment matrix.
Minimizing $J=\|\mathbf{W}_l-\mathbf{U}\mathbf{M}\|_F^2$ over $\mathbf{U},\mathbf{M}$
is exactly $k$-means; the optimal $\mathbf{M}$ is the cluster mean.

**Interdependence between layers.** Folding output channels of layer $l$ also
changes the input of layer $l{+}1$. The combined cost is minimized by clustering
the *concatenated* matrix:
$$J_{l,l+1} = \|\mathbf{W}_{l,l+1} - \mathbf{C}\mathbf{W}_{l,l+1}\|_F^2,\quad
\mathbf{W}_{l,l+1} = [\,\mathbf{W}_l \mid \mathbf{W}_{l+1}^T\,]$$
The folded forward pass is
$\tilde{\mathbf{y}}_{l+1} = \sigma((\mathbf{W}_{l+1}\mathbf{C}^T)\,\sigma((\mathbf{C}\mathbf{W}_l)\mathbf{x}_l))$.
In practice (Algorithm 1): the folded layer becomes $\mathbf{M}\mathbf{W}_l$
(output channels merged by **mean**) and the next layer becomes
$\mathbf{U}^T\mathbf{W}_{l+1}^T$ (input channels merged by **sum**).
$k$-means beats spectral, agglomerative and iterative-greedy (IFM-style) clustering.

### 2. Merging — simple mean of cluster members (the centroid $\mathbf{M}$).

### 3. Repairing data statistics

Averaging correlated-but-not-identical channels shrinks variance
(**variance collapse**); naive statistics handling (IFM) instead **overshoots**.
The *variance ratio* of layer $l$ must stay near 1:
$$\mu\!\left[\tfrac{\mathrm{Var}(\tilde{\mathbf{x}}_l)}{\mathrm{Var}(\mathbf{x}_l)}\right]
= \tfrac{1}{|\mathbf{x}_l|}\sum_k \tfrac{\mathrm{Var}(\tilde{\mathbf{x}}_{l,k})}{\mathrm{Var}(\mathbf{x}_{l,k})}$$

Three repair variants:

- **Fold-R** (data-driven REPAIR baseline, Jordan et al. 2022): recompute
  BatchNorm statistics with a forward pass over real data.
- **Fold-AR** (approximate REPAIR, data-free). The centroid of a cluster $c$ of
  size $N_c$ of unit-variance normalized activations has variance
  $\mathrm{Var}(\hat z_l(c)) = \tfrac{1}{N_c^2}[N_c + (N_c^2-N_c)E[c]]$,
  where $E[c]$ is the mean intra-cluster correlation. To restore unit variance,
  rescale every centroid (folded into the BN scale $\Sigma_s$):
  $$\hat z_l(c) \leftarrow \hat z_l(c)\,\frac{N_c}{\sqrt{N_c + (N_c^2-N_c)E[c]}}$$
  With no data, $E[c]$ is estimated from *normalized weights* assuming
  uncorrelated inputs:
  $E[c]=\tfrac{1}{N_c^2-N_c}\sum_{i\ne j\in I_c}
  \tfrac{\tilde{\mathbf{w}}_l(i)\tilde{\mathbf{w}}_l^T(j)}
  {\sqrt{(\tilde{\mathbf{w}}_l(i)\tilde{\mathbf{w}}_l^T(i))(\tilde{\mathbf{w}}_l(j)\tilde{\mathbf{w}}_l^T(j))}}$.
  Fold-AR clusters $\mathbf{W}_\text{tot}=[\mathbf{W}_{l+1}^T\mid\hat{\mathbf{W}}_l\mid\mathrm{diag}(\Sigma_s)]$
  with $\hat{\mathbf{W}}_l=\Sigma_n\mathbf{W}_l$; it is a **closed-form analytic
  correction, no forward pass**.
- **Fold-DIR** (Deep-Inversion REPAIR, data-free). Synthesize one batch of
  images by Deep Inversion (optimize noise to match BN running stats), then run
  a single forward pass to update BN statistics — same as Fold-R but with
  synthetic data.

### Structural details

- **Residual blocks:** the shortcut and the block's last conv must share the
  same clustering matrix ($\mathbf{U}_{l,s}=\mathbf{U}_l^{(2)}$); cost is
  $\|\mathbf{W}_\text{tot}-\mathbf{C}_l^{(2)}\mathbf{W}_\text{tot}\|_F^2$ with
  $\mathbf{W}_\text{tot}=[\mathbf{W}_{l,s}\mid\mathbf{W}_l^{(2)}]$.
- **BatchNorm:** scale $\Sigma_s$, normalize $\Sigma_n$, conv $\mathbf{W}_l$
  must all use the same $\mathbf{U}$ ($\mathbf{U}_s=\mathbf{U}_n=\mathbf{U}_l$).
- **Conv layers:** each filter $\mathcal{W}_{c_o,c_i,:,:}$ is flattened; rows of
  $\mathbf{W}_l$ are output channels, columns of $\mathbf{W}_{l+1}$ input channels.
- **LLMs:** cluster attention heads / MLP channels; no BN, so no REPAIR.

## Experiments

- **CV:** ResNet18/50, VGG11-BN on CIFAR10/100 and ImageNet. Fold-AR/Fold-DIR
  track data-driven Fold-R and beat IFM and structured magnitude pruning,
  with the gap widening at high sparsity (ResNet18/CIFAR10 keeps >80% acc at
  70% sparsity while pruning is near chance). Fold-DIR ≥ Fold-AR.
- **LLMs:** LLaMA-7B / LLaMA2-7B, 20% folding, no data/fine-tuning, comparable
  to calibration-based FLAP/Wanda_sp; fine-tuning only layernorms recovers more.
- **Compression in seconds** vs tens of hours for data-free KD (Table 12).
- *Not compared against:* fine-tuned pruning baselines (by design — data-free
  setting); per-layer adaptive sparsity is left to future work.

## Limitations & open questions

- Stated: ineffective on low-redundancy networks; uses uniform per-layer
  sparsity, no per-layer sensitivity allocation.
- Inferred: Fold-AR's $E[c]$ assumes uncorrelated layer inputs — optimistic in
  deep nets with strong feature correlation; all CV evidence is BN-based
  architectures; detection/segmentation heads are untested.

## Connections — to this project (`master_thesis_folding`)

This repo is a master-student attempt to port model folding to **YOLOv8m**
(object detection on COCO), `folding_main.py`. Findings from cross-checking the
paper + both official repos:

- **Clustering/merging is essentially correct.** `cumpute_cluster_matrix_u`
  builds $\mathbf{A}=[\mathbf{W}_l\mid\mathbf{W}_{l+1}^T]$ and $k$-means it —
  matches `concat_weights`/`compress_weight_clustering`. `merge_conv_bn` merges
  output channels by $\mathbf{M}$ (mean) and input channels by $\mathbf{U}^T$
  (sum) — matches `merge_channel_clustering` (axis 0 → $\mathbf{M}$, axis 1 → sum).
- **The forward-pass REPAIR is mis-scoped — the core bug.** Both official repos
  (`ModelFolding/.../test_merge` and `model-folding-universal/core/repair.py::
  reset_bn_stats`) reset **every** `BatchNorm2d` before the forward pass
  ("resetting stats ... is necessary for stability"). The student's
  `repair_bn_forward_pass` resets **only the BN of folded convs**. Folding also
  corrupts the *input-folded* next layer (summed input → variance explosion)
  and everything downstream; those BNs are never reset. `model.train()` then
  silently mutates the ~50 un-reset BNs too. This is why forward-pass REPAIR
  helps pruning (which calls it with `config_path=None` → resets all BN) but
  hurts folding.
- **Fold-AR is not implemented.** The paper's main data-free contribution
  (closed-form $\Sigma_s$ rescaling by $N_c/\sqrt{N_c+(N_c^2-N_c)E[c]}$,
  see `merge_channel_clustering_approx_repair`) is absent. Porting it is the
  highest-value next step and needs no calibration data.
- The student clusters **raw** conv weights, not BN-normalized weights
  $\hat{\mathbf{W}}_l=\Sigma_n\mathbf{W}_l$ as Fold-AR's Algorithm 1 specifies.

# LATTE
**L**atent-aware **A**utoregressive **T**ransformer for **T**oken **E**mbeddings — a **structure-informed protein Encoder**.

> LATTE learns a **structure-aligned** 256-d latent space by matching reconstruction embeddings to ESMS/ESM-2 with a **cosine + MSE** perceptual loss, while keeping the KL **active near 0.05** to prevent collapse. It achieves **97.17%** reconstruction on UniRef50 (held-out), yields **0.987** (5-fold) FP vs non-FP accuracy and **2.70/3.80 nm** RMSE for λ_abs/λ_em with simple GP models, and provides a broader, heavier-tailed geometry than ESM-2 that improves **latent prefilter recall** for **Deep BLAST**.
<p align="center
  <a href="docs/LATTE.pdf"><img src="https://img.shields.io/badge/Paper-LATTE%20(manuscript)-green.svg?style=flat-square" alt="paper"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Ahnd6474/LATTE?style=flat-square" alt="license"></a>
  <a><img src="https://img.shields.io/badge/python-3.9%2B-blue.svg?style=flat-square"></a>
  <a><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square"></a>
</p>

---

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Highlights](#highlights)  
3. [Method](#method)  
4. [Architecture](#architecture)  
5. [Dataset & Provided Latents](#dataset--provided-latents)  
6. [Installation](#installation)  
7. [Quick Start](#quick-start)
8. [Benchmarks & Results](#benchmarks--results)
9. [Embedding Geometry vs ESM-2](#embedding-geometry-vs-esm-2)
10. [Deep BLAST (Latent → Alignment)](#deep-blast-latent--alignment)
11. [Latent Tree Query API](#latent-tree-query-api)
12. [Reproducing Paper Results](#reproducing-paper-results)
13. [Known Limitations](#known-limitations)
14. [Citation](#citation)
15. [License](#license)
16. [Contact](#contact)

---

## Project Overview
LATTE is a compact (~**5.5 M** params) transformer encoder for protein sequences. Reconstructions are aligned to pretrained ESMS/ESM-2 embeddings using a **perceptual loss** (COS + MSE) in addition to CE and KL. This keeps the latent **informative** and avoids posterior collapse; the selected checkpoint (**epoch 380**) balances **Val CE = 0.072** and **KL ≈ 0.048**.

This repo also includes **1M LATTE encoder latents** for a random UniRef50 subset to enable fast kNN/ANN lookup and to serve as a **prefilter** for **Deep BLAST**.

---

## Highlights
- **Structure-aware training.** Perceptual loss aligns reconstructions to ESMS/ESM-2 (**COS + MSE**) with schedule  
  \(L_1 = \lambda(L_\text{COS}+L_\text{MSE}) + \alpha L_\text{CE} + \beta L_\text{KL}\),  
  **λ = 5**, **α** decays **30 → 0.1**, **β** warms **0 → 0.1** (first **100 epochs**).  
- **Active latent space.** Mean KL/dim ≈ **0.04998** (RMSE **0.07027**); avoids collapse seen in the ablation without structural loss.  
- **Compact.** ~**5.5 M** params (4-layer encoder, d=256, 4 heads, FFN=512, dropout=0.3).  
- **Reconstruction.** **97.17%** on held-out UniRef50 (epoch 380).  
- **Downstream FP tasks.** 5-fold accuracy **0.987** (FP vs non-FP); λ_abs / λ_em RMSE **2.70 / 3.80 nm** with simple Gaussian processes.  
- **Geometry.** Matched-subset pairwise cosine distances are **broader/heavier-tailed** vs ESM-2 (LATTE mean **0.1694**, SD **0.3428**; ESM-2 mean **0.0381**, SD **0.0822**), with high rank concordance (Spearman **ρ = 0.761**).  
- **Deep BLAST.** Latent kNN prefilter (256-d, cosine) → BLAST only on shortlist; reduces fan-out by **10–100×** while preserving alignment-level interpretability.

---

## Method
We train under teacher forcing with:
\[
L_1 = \lambda(L_\text{COS}+L_\text{MSE}) + \alpha L_\text{CE} + \beta L_\text{KL},
\]
**λ = 5**, **α: 30→0.1**, **β: 0→0.1** over **100 epochs**. The cosine term tolerates plausible substitutions in embedding space; MSE penalizes larger deviations. This prevents KL collapse and keeps latents predictive.

---

## Architecture
- **Encoder:** 4× Transformer (d_model=256, heads=4, FFN=512, dropout=0.3)  
- **Surrogate:** lightweight transformer that maps latent **z** 
- **Total params:** ~**5.5 M**  
<img src= "https://github.com/Ahnd6474/LATTE/blob/main/img/figure1.jpg"></img>
---

## Dataset & Provided Latents
- **Training:** random UniRef50 subsample; **epoch 380** chosen (KL near active threshold, lowest Val CE).  
- **Provided latents:** **1,000,000** UniRef50 sequences encoded by LATTE (256-d vectors + metadata) for FAISS/ANN retrieval and downstream analytics.

---

## Installation
```bash
# 1) Clone
git clone https://github.com/Ahnd6474/LATTE.git
cd LATTE

# 2) (Optional) Conda
conda create -n latte python=3.9 -y
conda activate latte

# 3) Python deps
pip install -r requirements.txt

# 4) (Optional) Git LFS for checkpoints
git lfs install && git lfs pull
```

---

## Quick Start
```python
from vae_module import Tokenizer, Config, load_vae, encode, decode

cfg = Config(model_path="models/latte_epoch380.pt", max_len=512)
tok = Tokenizer.from_esm()
model = load_vae(cfg, len(tok.vocab), tok.pad_idx, tok.bos_idx)

seq = "MKTFFVLLLACTIVCLLA"
z = encode(model, seq, tok, cfg.max_len)
new_seq = decode(model, z, tok, cfg.max_len)  # surrogate-assisted free run
print(new_seq)
```


## Benchmarks & Results
| Task                     | Dataset       | Metric             | LATTE (this work) |
|--------------------------|---------------|--------------------|-------------------|
| Reconstruction           | UniRef50      | % accurate         | **97.17**         |
| FP vs non-FP (5-fold)    | FPbase        | Accuracy           | **0.987**         |
| λ_abs                    | FPbase        | RMSE (nm)          | **2.70**          |
| λ_em                     | FPbase        | RMSE (nm)          | **3.80**          |


---

## Embedding Geometry vs ESM-2
<img src="https://raw.githubusercontent.com/Ahnd6474/LATTE/main/img/figure4.png" width="400">

- **Pairwise distances (matched subset):** LATTE mean **0.1694**, SD **0.3428**, p50 **0.0141**, p90 **0.9006** vs ESM-2 mean **0.0381**, SD **0.0822**, p50 **0.00953**, p90 **0.1133**  
- **Direct comparison:** OLS **slope = 0.125**, **intercept = 0.017**, **Spearman ρ = 0.761** → preserved neighbor ordering but expanded dynamic range (useful for recall in prefiltering)  
- **Clustering:** k = 3 cosine-silhouette **0.9431** (LATTE) vs **0.9022** (ESM-2); cross-partition agreement shows strong consistency (e.g., AMI/ARI/FMI as in Table 8)


## Deep BLAST (Latent → Alignment)
<img src="https://github.com/Ahnd6474/LATTE/blob/main/img/elbow_adaptive_full_870k.png?raw=true" width="400">
<img src="https://github.com/Ahnd6474/LATTE/blob/main/img/dendrogram.png" width="1000">

1. **Retrieve** top-K neighbors by **cosine** in 256-d LATTE latent space (FAISS/ANN).  
2. **Align** only that shortlist with **BLAST** for alignment-level interpretability.  
3. **Tune** K (or radius) for recall/cost; **fallback** to global BLAST when latent similarity is low.

This shifts BLAST from broad discovery to **precise refinement**, often cutting search fan-out by **10–100×** while enriching biologically coherent hits.

---
## Latent Tree Query API

We now ship the centroid tree index that was previously available only in
`notebooks/vec-treeing.ipynb`. The new `LatentTreeIndex` helper performs cosine
filtering over the hierarchical clustering to shortlist promising clusters (and
optionally return the underlying sequences) without scanning the entire 1M
latent catalogue.

```python
from vae_module import Config, Tokenizer, load_vae, LatentTreeIndex

cfg = Config(model_path="models/latte_epoch380.pt", max_len=512)
tok = Tokenizer.from_esm()
model = load_vae(cfg, len(tok.vocab), tok.pad_idx, tok.bos_idx)

# Load the tree exported by notebooks/vec-treeing.ipynb
index = LatentTreeIndex.from_directory(
    "hclust/index",
    members_path="clusters_438/members.parquet",  # requires git-lfs pull
    members_columns=["sequence", "uniref_id"],    # optional extra columns
)

query = "MKTFFVLLLACTIVCLLA"
hits = index.query_sequence(
    query,
    model,
    tok,
    cfg.max_len,
    top_k=5,
    max_distance=0.25,
    fetch_members=False,  # set True to return the sequences
)

for h in hits:
    print(h.cluster_id, h.distance, h.size)
```

> **Note**
> * Run `git lfs pull` (or download the full archives) to obtain
>   `clusters_438/members.parquet` if you plan to fetch the sequences behind a
>   cluster.
> * Sequence retrieval uses `pyarrow.dataset`. Install `pyarrow` if it is not
>   already available (`pip install pyarrow`).

The same API works for batches by encoding sequences yourself and passing the
latent vectors to `LatentTreeIndex.query_latent`.

---

## Reproducing Paper Results
```bash
# Train LATTE on a UniRef50 subsample
python train_latte.py   --data data/uniref50_subsample.fasta   --epochs 380   --save models/latte_epoch380.pt

# GFP pipelines (classification + spectral regression)
python fp_pipeline.py   --weights models/latte_epoch380.pt   --fp_csv data/fpbase_curated.csv

# Geometry: ESM-2 vs LATTE comparison
python geometry_compare.py   --esm_cache data/esm2_embeddings.npy   --latte_cache data/latte_latents.npy
```
(See the manuscript for exact settings and supplementary figures.)

---

## Known Limitations
- Extremely remote homology can still benefit from larger PLMs/MSA-based features; use LATTE as a **recall-boosting front-end**, not a wholesale replacement for alignment.

---

## Citation
If you use LATTE, please cite the manuscript:

```bibtex
@article{ahn2025latte,
  author    = {Danny Ahn},
  title     = {{LATTE}: A Structure-Informed Latent Model for Protein Sequence Embedding},
  journal   = {Bioinformatics},
  year      = {2025},
  url       = {https://github.com/Ahnd6474/LATTE},
  note      = {In press; preprint available at https://github.com/Ahnd6474/LATTE}
}
```

---

## License
Code and models are released under **MIT**; third-party components retain their original licenses.

---

## Contact
Danny Ahn — <ahnd6474@gmail.com>

# LATTE
**L**atent-conditioned **A**utoregressive **T**ransformer for **T**oken **E**mbeddings — a **structure‑informed protein VAE** with a latent‑conditioned decoder.

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Paper-LATTE%20(Bioinformatics%2C%20preprint)-green.svg?style=flat-square" alt="paper"></a>
  <a href="https://github.com/Ahnd6474/LATTE/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Ahnd6474/LATTE?style=flat-square" alt="license"></a>
  <a href="#"><img src="https://img.shields.io/badge/python-3.9%2B-blue.svg?style=flat-square"></a>
  <a href="#"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square"></a>
</p>

> **LATTE** is a **structure‑aware protein VAE** that aligns reconstructions to pretrained **ESMS/ESM‑2 embeddings** via a perceptual loss (**COS + MSE**). This keeps the latent space **active (target KL ≈ 0.05)** and informative for downstream tasks. LATTE reaches **97.17%** reconstruction on UniRef50 and supports **Deep BLAST**: latent‑space retrieval followed by classical alignment.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Highlights](#highlights)  
3. [Method](#method)  
4. [Architecture](#architecture)  
5. [Dataset & Latent Embeddings](#dataset--latent-embeddings)  
6. [Installation](#installation)  
7. [Quick Start](#quick-start)  
8. [Benchmarks & Results](#benchmarks--results)  
9. [Reproducing Paper Results](#reproducing-paper-results)  
10. [Deep BLAST (Latent→Alignment)](#deep-blast-latentalignment)  
11. [Known Limitations](#known-limitations)  
12. [Citation](#citation)  
13. [License](#license)  
14. [Contact](#contact)

---

## Project Overview

LATTE is a compact transformer VAE for protein sequences. It learns a **structure‑aligned latent space** by matching reconstruction embeddings to **ESMS/ESM‑2** through a **cosine + MSE perceptual loss**, while keeping an **active KL** to prevent posterior collapse. The latent is then used to condition an autoregressive decoder (teacher‑forced during training; surrogate‑assisted for free run).

This repository also provides **latent embeddings for a 1M random subset of UniRef50**, produced by the LATTE encoder. These latents enable **fast nearest‑neighbor queries** and serve as the prefilter for **Deep BLAST** (retrieve by cosine in latent space → run BLAST only on the shortlist).

---

## Highlights

- **Structure‑aware training** — token‑wise **cosine + MSE** alignment to ESMS/ESM‑2 embeddings (perceptual loss).  
- **Active latent space** — KL kept near **0.05**, avoiding collapse while remaining informative.  
- **Compact** — ~**5.5M** parameters (4‑layer encoder, 4‑layer decoder; *d*=256, 4 heads, FFN=512).  
- **Reconstruction** — **97.17%** on held‑out UniRef50.  
- **Downstream FP tasks** — FP vs non‑FP **0.987** (5‑fold); **2.70/3.80 nm** RMSE for λ_abs/λ_em with GP regressors.  
- **Embedding geometry** — broader, heavier‑tailed pairwise distances than ESM‑2; preserves neighbor ranking (high ρ), improving **latent prefilter recall**.  
- **Deep BLAST** — kNN in 256‑d latent space reduces fan‑out by orders of magnitude, then classic BLAST restores alignment‑level interpretability.

---

## Method

We optimize a structure‑aware ELBO variant under teacher forcing:

\[
\mathcal{L}_1 = \lambda (L_{\mathrm{COS}} + L_{\mathrm{MSE}}) + \alpha L_{\mathrm{CE}} + \beta L_{\mathrm{KL}},
\]

with **λ = 5**; **α** decays **30 → 0.1** and **β** warms **0 → 0.1** over the first **100 epochs**. The cosine term tolerates plausible substitutions; MSE penalizes larger embedding deviations. Together they keep latents informative and discourage collapse. A small **Transformer surrogate** learns to predict decoder memory from *z* for free‑run generation without encoder memory.

---

## Architecture

- Encoder: 4× Transformer layers (d_model=256, heads=4, FFN=512, Dropout=0.3)  
- Decoder: 4× Transformer layers (teacher‑forced training)  
- Surrogate: 2× layers, 4 heads (predicts decoder memory from latent *z*)  
- Total params ≈ **5.5M**

---

## Dataset & Latent Embeddings

- **Training:** random subsample of **UniRef50**. The selected checkpoint (**epoch 380**) balances **active KL ≈ 0.048** and low CE.  
- **Provided latents:** **1,000,000** sequences randomly sampled from **UniRef50**, encoded by the **LATTE encoder**. Each item includes: sequence ID, sequence length, 256‑d latent vector (and optional metadata).  
- **Intended use:** FAISS/ANN indexing for fast retrieval; **Deep BLAST** prefilter; downstream clustering, tagging, and property modeling.

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
from latte import Tokenizer, Config, load_vae, encode, decode

cfg = Config(model_path="models/latte_epoch380.pt")
tok = Tokenizer.from_esm()
model = load_vae(cfg, len(tok.vocab), tok.pad_idx, tok.bos_idx)

seq = "MKTFFVLLLACTIVCLLA"
z = encode(model, seq, tok, cfg.max_len)
# Teacher-forced or surrogate-assisted decoding
new_seq = decode(model, z, tok, cfg.max_len)
print(new_seq)
```

---

## Benchmarks & Results

| Task               | Dataset                 | Metric                 | LATTE (this work) | Notes                                  |
|--------------------|-------------------------|------------------------|-------------------|----------------------------------------|
| Reconstruction     | UniRef50 (held‑out)     | % accurate             | **97.17**         | Epoch 380 checkpoint                   |
| Mutational effect  | ProteinGym              | Spearman ρ (≤512 / all)| **0.7779 / 0.689**| 3‑layer MLP on latents                 |
| FP vs non‑FP       | FPbase                  | 5‑fold Accuracy        | **0.987**         | GP classifier                          |
| λ_abs              | FPbase                  | RMSE (nm)              | **2.70**          | GP regressor                           |
| λ_em               | FPbase                  | RMSE (nm)              | **3.80**          | GP regressor                           |

**Embedding geometry.** Pairwise cosine distances over matched subsets are **broader/heavier‑tailed** in LATTE than ESM‑2; direct LATTE↔ESM comparisons show strong rank concordance, so latent kNN preserves neighbor ordering while expanding dynamic range—useful for **latent prefiltering** before alignment.

---

## Reproducing Paper Results

```bash
# Train on a UniRef50 subsample
python train_latte.py --data data/uniref50_subsample.fasta                       --epochs 380                       --save models/latte_epoch380.pt

# ProteinGym evaluation (mutational effects)
python protein_gym_evaluate.py --weights models/latte_epoch380.pt
```

---

## Deep BLAST (Latent→Alignment)

1. **Retrieve** top‑K neighbors by **cosine** in 256‑d LATTE latent space (FAISS/ANN).  
2. **Align** only this shortlist with classical **BLAST** to recover alignment‑level interpretability.  
3. **Tune** K (or radius) to trade recall vs. cost; fall back to global BLAST if latent similarity is too low.  

This pipeline reduces fan‑out by **10–100×** while enriching for structurally/functional coherent hits, improving top‑rank precision without sacrificing interpretability.

---

## Known Limitations

- Free‑run decoding can drift on very long sequences; the surrogate mitigates but does not eliminate this.  
- Extremely remote homology still benefits from larger PLMs or MSA‑based features; use the latent prefilter as a **recall‑boosting front‑end**, not a replacement for alignment.

---

## Citation

If you use LATTE, please cite:

```bibtex
@article{ahn2025latte,
  title={LATTE: A Structure-Informed Latent Model for Protein Sequence Embedding},
  author={Ahn, Danny and Lee, Minjae and Moon, Sihyeon and Jung, Jooyoung},
  journal={Bioinformatics},
  year={2025},
  doi={10.1093/bioinformatics/btzXXX}
}
```

---

## License

Code and models are released under the **Business Source License 1.1 (BSL‑1.1)**. Third‑party components retain their respective licenses.

---

## Contact

Danny Ahn — <ahnd6474@gmail.com>

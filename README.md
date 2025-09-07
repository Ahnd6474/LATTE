# Latent GPT

<p align="center">
  <a href="https://doi.org/10.1093/bioinformatics/btzXXX"><img src="https://img.shields.io/badge/Paper-Bioinformatics(TMD)-green.svg?style=flat-square" alt="paper"></a>
  <a href="https://github.com/Ahnd6474/Latent-GPT/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Ahnd6474/Latent-GPT?style=flat-square" alt="license"></a>
  <a href="#"><img src="https://img.shields.io/badge/python-3.9%2B-blue.svg?style=flat-square"></a>
  <a href="#"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square"></a>
</p>

> **Latent GPT** (Latent Generative Pretrained Transformer) is a **structure-aware protein VAE** that aligns reconstructions to **pretrained ESM2/ESMS embeddings** via a perceptual loss (COS + MSE), keeping the latent space active (KL ≈ 0.05) and informative. It reaches **97.17%** reconstruction on UniRef50 and **ProteinGym Spearman’s ρ = 0.7779 (≤512 aa) / 0.689 (all 217)**. Downstream FP tasks: **0.987** (5-fold accuracy) and **2.7/3.8 nm** RMSE for absorption/emission.

> **Note on math rendering:** Use `$$ ... $$` for display and `$ ... $` for inline LaTeX on GitHub.

---

**Table of Contents**

1. [Features](#features)
2. [Method](#method)
3. [Architecture Diagram](#architecture-diagram)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Repository Structure](#repository-structure)
7. [Pre-trained Models](#pre-trained-models)
8. [Reproducing Paper Results](#reproducing-paper-results)
9. [Benchmarks](#benchmarks)
10. [Citation](#citation)
11. [Availability and Implementation](#availability-and-implementation)
12. [License](#license)
13. [Contact](#contact)

---

## Features

- **Structure-aware training** — token-wise **cosine + MSE** alignment to **ESM2/ESMS** embeddings (perceptual loss) to encode structural/functional cues and avoid KL collapse.
- **Lightweight** — ~5.5M parameters; transformer encoder–decoder (4 layers each, d=256, 4 heads).
- **Active latent space** — mean KL near **0.05**, reducing posterior collapse risk.
- **Generalization** — **97.17%** reconstruction on UniRef50; **ProteinGym ρ = 0.7779 (≤512 aa) / 0.689 (all 217)** with a simple 3-layer MLP on latents.
- **Downstream utility** — FP vs non-FP **0.987** (5-fold), wavelength RMSE **2.7/3.8 nm**.

---

## Method

Two-phase objective with perceptual supervision:

**Phase-1 (teacher-forced)**

$$
L_1 \;=\; \lambda\,(L_{\text{COS}} + L_{\text{MSE}}) \;+\; \alpha\,L_{\text{CE}} \;+\; \beta\,L_{\text{KL}} \,,
$$

computed on pretrained **ESM2/ESMS** embeddings to enforce structural consistency.

**Phase-2 (Train surrogate)**

$$
L_2 \;=\; L_{\text{CE}}(\tilde{x}) 
$$

warming up \( \lambda \) to stabilize latent-conditioned rollouts (K≈64–256).

---

## Architecture Diagram

![Latent GPT overview](https://github.com/Ahnd6474/Latent-GPT/blob/main/img/ML_architecture-1.png)

---

## Installation

```bash
# 1) Clone
git clone https://github.com/Ahnd6474/Latent-GPT.git
cd Latent-GPT

# 2) (Optional) Conda env
conda create -n latent-gpt python=3.9 -y
conda activate latent-gpt

# 3) Python deps
pip install -r requirements.txt
```

---

## Quick Start

```python
from vae_module import Tokenizer, Config, load_vae, encode, decode

cfg = Config(model_path="models/vae_epoch380.pt")
tok = Tokenizer.from_esm()

model = load_vae(cfg,
                 vocab_size=len(tok.vocab),
                 pad_idx=tok.pad_idx,
                 bos_idx=tok.bos_idx)

seq = "MKTFFVLLLACTIVCLLA"
z   = encode(model, seq, tok, cfg.max_len)
new_seq = decode(model, z, tok)
print(new_seq)
```

> Check `notebooks/` for end-to-end training/evaluation examples.

---

## Repository Structure

- `notebooks/latent-gpt-training.ipynb` — end-to-end training/evaluation.
- `notebooks/fp-cluster.ipynb` — K-means clustering + consensus decoding.
- `notebooks/fp-regressor.ipynb` — GP/MLP regressors on latent features.
- `models/vae_epoch380.pt` — main checkpoint used in the paper.
- `docs/Latent_GPT.pdf` — paper/preprint (short form).

---

## Pre-trained Models

| File              | Epoch | KL     | Rec. Acc.  | Notes                                  |
|-------------------|:-----:|:------:|:----------:|----------------------------------------|
| `vae_epoch380.pt` |  380  | 0.048  | **97.17%** | Paper model (used in all experiments)  |
| `vae_epoch500.pt` |  500  | 0.002  | 99.98%     | Very low KL (risk of collapse)         |
| `vae_sur.pt`      |  380  | 0.048  | **97.17%** | VAE with 2-layer surrogate memory net  |

> We use **Git LFS** for checkpoints. Run `git lfs pull` after cloning.

The `vae_sur.pt` bundle packages a small transformer surrogate (`Z2MemorySurrogate`) that predicts decoder memory directly from the latent vector. This enables faster, teacher-free generation while keeping reconstruction quality.

### Usage

```python
from vae_module import Tokenizer, Config, load_vae, encode, decode

cfg = Config(model_path="models/vae_sur.pt")
tok = Tokenizer.from_esm()
model = load_vae(cfg, len(tok.vocab), tok.pad_idx, tok.bos_idx)

seq = "MKTFFVLLLACTIVCLLA"
z = encode(model, seq, tok, cfg.max_len)
new_seq = decode(model, z, tok)  # surrogate supplies decoder memory
print(new_seq)
```

---

## Reproducing Paper Results

```bash
# Training on a UniRef50 subsample
python train_baseline.py --data data/uniref50_subsample.fasta                          --epochs 380                          --save models/vae_epoch380.pt

# ProteinGym evaluation (all sets)
python protein_gym_evaluate.py --weights models/vae_epoch380.pt
```

> Reference training was run on **Kaggle T4 sessions** (see paper).

---

## Benchmarks

| Task               | Dataset                   | Metric       | Latent GPT     | Notes                                  |
|--------------------|---------------------------|--------------|----------------|----------------------------------------|
| Reconstruction     | UniRef50 (held-out)       | % accurate   | **97.17**      |                                        |
| Mutational effect  | ProteinGym (≤512 / all)   | Spearman ρ   | **0.7779 / 0.689** | 3-layer MLP on latents(on full dataset of all mutations)         |
| FP vs non-FP       | FPbase                     | 5-fold Acc   | **0.987**      | GP classifier                          |
| λ_abs              | FPbase                     | RMSE (nm)    | **2.70**       | GP regressor                           |
| λ_em               | FPbase                     | RMSE (nm)    | **3.80**       | GP regressor                           |

---

## Citation

If you use this code, please cite:

```bibtex
@article{ahn2025latentgpt,
  title={Latent GPT: A Structure-Informed Variational Autoencoder for Sequence Embedding and De Novo Protein Generation},
  author={Ahn, Danny and Lee, Minjae and Moon, Sihyeon and Jung, Jooyoung},
  journal={Bioinformatics},
  year={2025},
  doi={10.1093/bioinformatics/btzXXX}
}
```

---

## Availability and Implementation

Code, pretrained weights, and datasets: **https://github.com/Ahnd6474/Latent-GPT**.

---

## License

Code and models are released under the [Business Source License 1.1](LICENSE).  
Third-party components retain their respective licenses (Biopython, MIT, BSD-3-Clause, MPL-2.0).

---

## Contact

Contact: <ahnd6474@gmail.com>

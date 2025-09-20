# ZART
**Z**-**A**utoregressive **R**econstruction with **T**ransformer — a **structure‑informed protein VAE** with a latent‑conditioned autoregressive decoder.

<p align="center">
  <a href="https://doi.org/10.1093/bioinformatics/btzXXX"><img src="https://img.shields.io/badge/Paper-ZART%20(Bioinformatics%2C%20preprint)-green.svg?style=flat-square" alt="paper"></a>
  <a href="https://github.com/Ahnd6474/Latent-GPT/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Ahnd6474/Latent-GPT?style=flat-square" alt="license"></a>
  <a href="#"><img src="https://img.shields.io/badge/python-3.9%2B-blue.svg?style=flat-square"></a>
  <a href="#"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square"></a>
</p>

> **ZART**  is a **structure‑aware protein VAE** that aligns reconstructions to **pretrained ESMS/ESM2 embeddings** via a perceptual loss (**COS + MSE**). This keeps the latent space **active (KL ≈ 0.05)** and informative for downstream tasks. ZART reaches **97.17%** reconstruction on UniRef50 and **ProteinGym Spearman’s ρ = 0.7779 (≤512 aa, 162 sets) / 0.689 (all 217)**. On fluorescent‑protein (FP) tasks, it achieves **0.987** (5‑fold accuracy) and **2.70 / 3.80 nm** RMSE for absorption/emission prediction.

---

**Table of Contents**

1. [Features](#features)  
2. [Method](#method)  
3. [Architecture](#architecture)  
4. [Installation](#installation)  
5. [Quick Start](#quick-start)  
6. [Pre‑trained Models](#pre-trained-models)  
7. [Benchmarks & Results](#benchmarks--results)  
8. [Reproducing Paper Results](#reproducing-paper-results)  
9. [Data & Availability](#data--availability)  
10. [Known Limitations](#known-limitations)  
11. [Citation](#citation)  
12. [License](#license)  
13. [Contact](#contact)

---

## Features

- **Structure‑aware training** — token‑wise **cosine + MSE** alignment to **ESMS/ESM2** embeddings (perceptual loss) to encode structural/functional cues and avoid KL collapse.
- **Lightweight** — ~**5.5M** parameters; Transformer **encoder/decoder 4 layers each**, *d*=256, **4 heads**, FFN=512.
- **Active latent space** — mean **KL ≈ 0.05**, mitigating posterior collapse.
- **Generalization** — **97.17%** reconstruction on UniRef50.
- **Mutational effects** — **ProteinGym ρ = 0.7779 (≤512 aa, n=162) / 0.689 (all 217)** with a simple 3‑layer MLP on latents.
- **Downstream utility** — FP vs non‑FP **0.987** (5‑fold CV), wavelength RMSE **2.70 / 3.80 nm**.

---

## Method

### Loss & Training Phases

**Phase‑1 (teacher‑forced)** optimizes a structure‑aware ELBO variant:

$$
L_1 = \lambda\ (L_{\mathrm{COS}} + L_{\mathrm{MSE}}) + \alpha\ L_{\mathrm{CE}} + \beta L_{\mathrm{KL}} .
$$

where **λ = 5**, **α** is linearly decayed **30 → 0.1** and **β** is warmed **0 → 0.1** over the first **100 epochs**. The cosine term tolerates plausible substitutions while MSE penalizes larger deviations, keeping latents informative (target **KL ≈ 0.05**).

**Phase‑2 (surrogate for free‑run)** trains a small **Transformer surrogate** to predict decoder memory from *z*, enabling latent‑conditioned autoregressive decoding without encoder memory.

### Hyperparameters

| Parameter    | Value |
|--------------|:-----:|
| Vocab Size   | 33    |
| d_model      | 256   |
| Latent Dim   | 256   |
| n_heads      | 4     |
| Feed Forward | 512   |
| Dropout      | 0.3   |

---

## Architecture

ZART is a compact Transformer VAE with **4‑layer encoders/decoders** and a **2‑layer, 4‑head surrogate** for free‑run. (~**5.5M** params total).

<p align="center">
  <img src="https://github.com/Ahnd6474/Latent-GPT/blob/main/img/ML_architecture-1.png" alt="ZART overview" width="70%"/>
</p>

---

## Installation

```bash
# 1) Clone
git clone https://github.com/Ahnd6474/ZART.git
cd ZART

# 2) (Optional) Conda env
conda create -n zart python=3.9 -y
conda activate zart

# 3) Python deps
pip install -r requirements.txt

# 4) Fetch pre-trained weights (uses Git LFS)
git lfs install
git lfs pull
```

---

## Quick Start

```python
from vae_module import Tokenizer, Config, load_vae, encode, decode

cfg = Config(model_path="models/vae_epoch380.pt")
tok = Tokenizer.from_esm()
model = load_vae(cfg, len(tok.vocab), tok.pad_idx, tok.bos_idx)

seq = "MKTFFVLLLACTIVCLLA"
z = encode(model, seq, tok, cfg.max_len)
new_seq = decode(model, z, tok, cfg.max_len)  # teacher-forced or surrogate-assisted
print(new_seq)
```

Check `notebooks/` for end‑to‑end training/evaluation examples.

---

## Pre‑trained Models

| File              | Epoch | KL     | Rec. Acc. | Notes                                  |
|-------------------|:-----:|:------:|:---------:|----------------------------------------|
| `vae_epoch380.pt` |  380  | 0.048  | **97.17%**| Paper model (used in all experiments)  |
| `vae_epoch500.pt` |  500  | 0.002  | 99.98%    | Very low KL (risk of collapse)         |
| `vae_sur.pt`      |  380  | 0.048  | **97.17%**| VAE + 2‑layer surrogate memory for free‑run |

> We use **Git LFS** for checkpoints. Run `git lfs pull` after cloning.

---

## Benchmarks & Results

| Task               | Dataset                 | Metric          | ZART (Latent GPT) | Notes                                 |
|--------------------|-------------------------|-----------------|-------------------|---------------------------------------|
| Reconstruction     | UniRef50 (held‑out)     | % accurate      | **97.17**         |                                       |
| Mutational effect  | ProteinGym (≤512 / all) | Spearman ρ      | **0.7779 / 0.689**| 3‑layer MLP on latent embeddings      |
| FP vs non‑FP       | FPbase                  | 5‑fold Accuracy | **0.987**         | GP classifier                         |
| λ_abs              | FPbase                  | RMSE (nm)       | **2.70**          | GP regressor                          |
| λ_em               | FPbase                  | RMSE (nm)       | **3.80**          | GP regressor                          |

**Notes.** Noisy‑latent reconstruction improves with noise scale; sampled *z* near active KL produce **novel sequences** with low training‑set identity; FP k‑means + consensus decoding yields GFP‑like folds (see paper & supplement).

---

## Reproducing Paper Results

```bash
# Training on a UniRef50 subsample
python train_baseline.py --data data/uniref50_subsample.fasta \
                         --epochs 380 \
                         --save models/vae_epoch380.pt

# ProteinGym evaluation
python protein_gym_evaluate.py --weights models/vae_epoch380.pt
```

Reference training used **Kaggle T4 sessions**.

---

## Data & Availability

- **Paper:** Bioinformatics (preprint), DOI pending  
- **Code & Models:** https://github.com/Ahnd6474/Latent-GPT  
- **FP dataset:** FPbase / Kaggle (see paper for links)

---

## Known Limitations

- During **free‑run decoding**, attention mismatches can degrade sequence quality. The surrogate helps, but long‑horizon rollouts may still underperform teacher‑forced reconstruction; further stabilization is ongoing.

---

## Citation

If you use this work, please cite:

```bibtex
@article{ahn2025zart,
  title={ZART: A Structure-Informed Variational
Autoencoder for Sequence Embedding and De
Novo Protein Generation},
  author={Ahn, Danny and Lee, Minjae and Moon, Sihyeon and Jung, Jooyoung},
  journal={Bioinformatics},
  year={2025},
  doi={}
}
```

---

## License

Code and models are released under the **Business Source License 1.1 (BSL‑1.1)**.  
Third‑party components retain their respective licenses.

---

## Contact

Danny Ahn — <ahnd6474@gmail.com>

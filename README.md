
# ESMS‑VAE

<p align="center">
  <a href="https://doi.org/10.1093/bioinformatics/btzXXX"><img src="https://img.shields.io/badge/Paper-Bioinformatics(TMD)-green.svg?style=flat-square" alt="paper"></a>
  <a href="https://github.com/Ahnd6474/ESMS-VAE/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Ahnd6474/ESMS-VAE?style=flat-square" alt="license"></a>
  <a href="#"><img src="https://img.shields.io/badge/python-3.9%2B-blue.svg?style=flat-square"></a>
  <a href="#"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square"></a>
</p>

> **ESMS‑VAE** (*Evolutionary Scale Modeling Student VAE*) is a 5.5 M‑parameter transformer VAE that learns structure‑aware latent representations of proteins through a novel **structural loss**.  It reaches **97.17 %** reconstruction accuracy on UniRef50 sequences and surpasses prior VAEs on the ProteinGym benchmark(Supervised) (*ρ = 0.689*).  Downstream tasks such as fluorescent‑protein classification (F1 = 0.99) and wavelength regression (RMSE ≈ 3 nm) confirm its practical utility.

---

**Table of Contents**

1. [Features](#features)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Pre‑trained Models](#pre-trained-models)
5. [Reproducing Paper Results](#reproducing-paper-results)
6. [Benchmarks](#benchmarks)
7. [Citation](#citation)
8. [License](#license)

---

### Features

* **Structure‑aware learning** via cosine + MSE loss on ESMS embeddings.  
* **Lightweight** – 5.5 M parameters; trains on a single T4 in ≤ 6 h.  
* **High fidelity** – 97 % reconstruction on test sequences.  
* **Robust latent space** – active KL (~0.05) prevents posterior collapse.  
* **Strong generalisation** – outperforms Kermut on 217 ProteinGym DMS sets.  
* **Plug‑and‑play embeddings** for GP/NN models in downstream prediction tasks.

![Architecture Diagram](img/struct.png)

---

### Installation

```bash
# 1. Clone
git clone https://github.com/Ahnd6474/ESMS-VAE.git
cd ESMS-VAE

# 2. Create env (optional)
conda create -n esms-vae python=3.9 -y
conda activate esms-vae

# 3. Install Python requirements
pip install -r requirements.txt
```

> **GPU:** A single NVIDIA T4/RX‑A5000 (~16 GB) is sufficient for both training and inference.

---

### Quick Start

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

---

### Pre‑trained Models

| File | Epoch | KL | Rec. Acc. | Notes |
|------|------:|----:|----------:|-------|
| `vae_epoch380.pt` | 380 | 0.048 | **97.17 %** | Paper model (used in all experiments) |
| `vae_epoch500.pt` | 500 | 0.002 | 99.98 % | High accuracy but suffers KL vanishing |

Download from the [**Releases**](../../releases) page or via `git lfs pull`.

---

### Reproducing Paper Results

```bash
# Training on UniRef50 subset
python train_baseline.py --data data/uniref50_subsample.fasta                          --epochs 380                          --save models/vae_epoch380.pt

# ProteinGym inference (takes ≈3 h)
python protein_gym_evaluate.py --weights models/vae_epoch380.pt
```

The scripts will output a CSV matching Table S2 of the paper.

---

### Benchmarks

| Task | Dataset | Metric | ESMS‑VAE | Previous SOTA |
|------|---------|--------|---------:|--------------:|
| Reconstruction | UniRef50 test | % accurate | **97.17** |-|
| Mutational effect | ProteinGym (162/217) | Spearman ρ | **0.7779**/**0.689** | 0.698/0.657 (Kermut) |
| FP vs non‑FP | FPbase | 5‑fold Acc | **0.987** |-|
| λabs | FPbase | RMSE (nm) | **2.70** |-|
| λem | FPbase | RMSE (nm) | **3.80** |-|

---

### Citation

If you use this code, please cite:

```bibtex
@article{ahn2025esmsvae,
  title={ESMS VAE: A Structure-Informed Variational Autoencoder for Protein Engineering},
  author={Ahn, Danny and Lee, Minjae and Moon, Shihyun and Jung, Jooyoung},
  journal={Bioinformatics},
  year={2025},
  doi={10.1093/bioinformatics/btzXXX}
}
```

---

### License

Non‑commercial research use only.  See [LICENSE](LICENSE) for details.

---

> © 2025 Danny Ahn et al. 본 리포지토리는 비영리 연구 목적에 한해 사용을 허가합니다.

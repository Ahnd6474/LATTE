# ESMSVAE Baseline

This baseline uses the pretrained ESMS-VAE model and a small MLP regression head to score ProteinGym supervised datasets.

## Usage

```bash
python baselines/ESMSVAE/scoring.py \
    --checkpoint models/vae_epoch380.pt \
    --assay path/to/OTU7A_HUMAN_Tsuboyama_2023_2L2D.csv
```

The script prints a line of the form:

```
<assay_name>\tSpearman: 0.8421
```

Ensure that the dependencies listed in `requirements.txt` are installed before running.

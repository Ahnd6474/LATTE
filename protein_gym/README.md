# ProteinGym Leaderboard Submission

This directory contains helper scripts and documentation for generating
prediction files that can be submitted to the ProteinGym leaderboard.
The workflow relies on the pretrained **ESMS VAE** model provided in
`models/vae_epoch380.pt` and trains a small MLP head on each ProteinGym
DMS substitution dataset.

## Required Data

1. Download the `DMS_ProteinGym_substitutions` archive from Kaggle or
the official ProteinGym website.
2. Place all CSV files into `protein_gym/data/`. Each file should
   contain a column with the mutated sequence (named either
   `sequence` or `mutated_sequence`) and a `DMS_score` column.

## Generating Predictions

Run `generate_submission.py` with the folder containing the CSV files
and an output directory:

```bash
python protein_gym/generate_submission.py \
    --data-dir protein_gym/data \
    --output-dir protein_gym/predictions \
    --weights models/vae_epoch380.pt
```

For every CSV file, a new file `<name>_pred.csv` will be written to the
output directory containing the original sequence and the predicted
score. These files can be zipped and uploaded to the leaderboard.

## Supervised Models

If you trained additional supervised models for the benchmark, place
their checkpoints inside `protein_gym/supervised_models/`.
A short description of each model (architecture, training procedure,
metrics) can be added to `supervised_models/README.md`.

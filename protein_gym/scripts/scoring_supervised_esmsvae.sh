#!/bin/bash
#SBATCH --job-name=prot_esmsvae
#SBATCH --gpus=1
#SBATCH --time=02:00:00

CHECKPOINT_URL="https://your.storage/esmsvae.pt"
CHECKPOINT_PATH="esmsvae.pt"

wget -O ${CHECKPOINT_PATH} ${CHECKPOINT_URL}

pip install -r protein_gym/baselines/ESMSVAE/requirements.txt

ASSAYS=(
  OTU7A_HUMAN_Tsuboyama_2023_2L2D
  SDA_BACSU_Tsuboyama_2023_1PV0
  # add the remaining assay IDs
)

for assay in "${ASSAYS[@]}"; do
  python protein_gym/baselines/ESMSVAE/scoring.py \
    --checkpoint ${CHECKPOINT_PATH} \
    --assay data/${assay}.csv
done

#!/bin/bash

DATASET="../Data/ILM/compound_dataset"

for SPLIT in train valid
do
python create_ilm_examples.py \
  ${SPLIT} \
  data/char_masks/${DATASET} \
  --data_dir ${DATASET} \
  --seed 0 \
  --data_name custom \
  --data_split ${SPLIT}
done

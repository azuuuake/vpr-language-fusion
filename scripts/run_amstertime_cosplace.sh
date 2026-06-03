#!/bin/bash

set -e

cd /content || exit

if [ ! -d "auto_VPR" ]; then
  git clone --recursive https://github.com/gmberton/auto_VPR
fi

cd auto_VPR
git submodule update --init --recursive

pip install -r requirements.txt -q
pip install faiss-cpu -q

printf "y\n" | python main.py \
  --method=cosplace \
  --backbone=ResNet18 \
  --descriptors_dimension=512 \
  --database_folder="/content/VPR-datasets-downloader/datasets/amstertime/images/test/database" \
  --queries_folder="/content/VPR-datasets-downloader/datasets/amstertime/images/test/queries" \
  --device=cpu \
  --image_size 320 320 \
  --batch_size 4 \
  --num_workers 2 \
  --recall_values 1 5 10 \
  --save_descriptors \
  --log_dir amstertime_cosplace_test

#!/bin/bash

set -e

cd /content || exit

if [ ! -d "VPR-datasets-downloader" ]; then
  git clone https://github.com/gmberton/VPR-datasets-downloader
fi

cd VPR-datasets-downloader
pip install -r requirements.txt -q
python download_amstertime.py

echo "AmsterTime downloaded to:"
echo "/content/VPR-datasets-downloader/datasets/amstertime"

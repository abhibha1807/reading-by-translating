#!/usr/bin/env bash
nvidia-smi
pwd
source activate /abhibha-volume/envs
conda info
which python
conda list
# pip install torchtext
# pip install tensorboard
# pip install utils
# pip install tokenizers
python /abhibha-volume/reading-by-translating/main.py

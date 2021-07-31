#!/usr/bin/env bash
nvidia-smi
pwd
source activate /abhibha-volume/envs
conda info
which python
python Model.py
python /abhibha-volume/reading-by-translating/main_.py
#python trainTokenizer.py

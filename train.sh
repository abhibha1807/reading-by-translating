#!/usr/bin/env bash
nvidia-smi
conda env create -f environment.yml
conda update -n base -c defaults conda
source activate yxy
python main_.py
#!/usr/bin/env bash
nvidia-smi
conda env create -f environment.yml
conda update -n base -c defaults conda
source activate yxy
# conda install -c huggingface transformers
# conda install -c conda-forge tensorboard
# conda install -c pytorch torchtext
python main_.py
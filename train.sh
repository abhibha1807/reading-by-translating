#!/usr/bin/env bash
nvidia-smi
# conda env create -f environment.yml
# conda update -n yxy -c defaults conda
# conda install -n base -c conda-forge tensorboard
# conda install -n base -c conda-forge huggingface transformers
# source activate rbt
# conda init bash
# conda activate rbt
# conda install -c huggingface transformers
# conda install -c conda-forge tensorboard
# conda install -c pytorch torchtext
export PATH="$PATH:/abhibha-volume/rbt/bin/python"
set -e 
source //abhibha-volume/rbt/bin/activate
echo $VIRTUAL_ENV
which python
pip list --local
# echo $PYTHONPATH
# export PYTHONPATH="$PWD"
# echo $PYTHONPATH
# alias python=python3.6
pwd
/abhibha-volume/rbt/bin/python --version
/abhibha-volume/rbt/bin/python -c "import sys; print('\n'.join(sys.path))"
/abhibha-volume/rbt/bin/python main_.py
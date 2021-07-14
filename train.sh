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
source ../rbt/bin/activate
echo $VIRTUAL_ENV
echo $PYTHONPATH
export PYTHONPATH="$PWD/reading-by-translating"
echo $PYTHONPATH
# alias python=python3.6
pwd
../rbt/bin/python3.6 --version
../rbt/bin/python3.6 -c "import sys; print('\n'.join(sys.path))"
../rbt/bin/python3.6 main_.py
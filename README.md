# Reading By Translating

This code attmpts to implement the idea outlined in Reading by translating.

### Dataset
The WMT English to German dataset is used. Link: https://nlp.stanford.edu/projects/nmt/
The percentage of dataset to be used as the unlabled dataset can be specified in the config file.

### Dependencies
  - cudatoolkit=10.2
  - pytorch=1.8.0
  - tensorboard=1.15.0
  - transformers=4.8.2
  - torchtext=0.9
  - tokenizers=0.10.1

### Files
**`Model.py`**
MT model class to load pretrained models from the 'models' directory and performs 
training according to the  pipeline described in the RBT paper. Training occurs in 3 steps,
1) Train first MT model using matrix A to calculate loss.
2) Train second MT model on a dataset created by first MT model on the unlabeled dataset.
3) Estimate A by reducing the validation loss of second MT model on validation set of MT dataset.

**`losses.py`**

Implements loss functions for step 1,2 and 3 of the pipeline.
compute_loss1-> calculates batch loss for step1 by multiplying A
compute_loss2-> calculates batch loss for step 2 and 3. 

**`main.py`**
Responsible for loading data, tokenizers, optimizers, schedulers, criterion and executing the main training loop.

**`Traintokenizer.py`**
Load training data and train a BertWordPieceTokenizer with the specified params and saves it.

**`utils.py`**
Implements suppplementary functions
calc_bleu: Calculates the bleu score for a batch
createBatchesA: dataloader class for A
loadTokenizer: loads trained BertWordPieceTokenizer.



## Steps to run
Run `train.sh`
Running `train.sh` performs the following tasks,
1. It sets up the environment using `environment.yml`
2. It runs `trainTokenizer.py` that trains and saves the tokenizers in the 'tokenizers' directory.
3. The training takes place when `main.py` is run.



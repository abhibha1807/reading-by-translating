import torch
from torch.utils.data import Dataset
import torchtext
from torchtext.data.metrics import bleu_score
from transformers import BertTokenizerFast
import os

'''
Implements suppplementary functions
calc_bleu: calculate the bleu score for a batch
createBatchesA: dataloader class for A
loadTokenizer: loads trained BertWordPieceTokenizer.
'''

def calc_bleu(en_input, lm_labels, model, tokenizer):
  gen_op=model.generate(input_ids=en_input, decoder_start_token_id=0) 
  print(gen_op)
  candidate=(tokenizer.batch_decode(gen_op))
  reference=(tokenizer.batch_decode(lm_labels))

  print(candidate)
  print(reference)
  for i in range(len(candidate)):
    score=0
    can = candidate[i].split(' ')
    print(can)
    ref = reference[i].split(' ')
    print(ref)
    while (len(can)<len(ref)):
      can.append('[PAD]')
    print(len(can), len(ref))
    score+=(bleu_score(can, ref))
  return(score/en_input.shape[0])

# creates a dataloader for weight matrix A 
class createBatchesA(Dataset):
    def __init__(self, A):
        self.samples = A

    def __len__(self):
      return len(self.samples)

    def __getitem__(self, idx):
      return self.samples[idx]

def _concat(xs, device):
  p=[]
  for x in xs:
    p.append(x.view(-1).to(device))
  return (torch.cat(p).to(device))

def loadTokenizer(train_en_file, encparams, train_de_file, decparams):
  en_tok_path = encparams["tokenizer_path"]
  en_tokenizer = BertTokenizerFast(os.path.join(en_tok_path, "vocab.txt"))
  de_tok_path = decparams["tokenizer_path"]
  de_tokenizer = BertTokenizerFast(os.path.join(de_tok_path, "vocab.txt"))
  return (en_tokenizer, de_tokenizer)
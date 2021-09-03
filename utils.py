import torch
from torch.utils.data import Dataset
import torchtext
from torchtext.data.metrics import bleu_score
from transformers import BertTokenizerFast
import os
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
cc = SmoothingFunction()

'''
Implements suppplementary functions
calc_bleu: calculate the bleu score for a batch
createBatchesA: dataloader class for A
loadTokenizer: loads trained BertWordPieceTokenizer.
'''

def calc_bleu(en_input, lm_labels, model, tokenizer):
  gen_op=model.generate(input_ids=en_input, decoder_start_token_id=0) 
  # print(gen_op)
  candidate=(tokenizer.batch_decode(gen_op))
  reference=(tokenizer.batch_decode(lm_labels))

  # print(candidate)
  # print(reference)
  score=0
  for i in range(len(candidate)):
    can =  candidate[i].split(' ')
    print('can:', can)
    ref = [i for i in reference[i].split(' ')]
    print('ref:', ref)
    print('\n')
    # while (len(can)<len(ref)):
    #   can.append('[PAD]')
    # print(len(can), len(ref))
    # if len(can)==len(ref):
    #   score+=(bleu_score(can, ref))
    # else:
    #   print('invalid lengths' )

    score = score + sentence_bleu(ref, can, smoothing_function=cc.method7, weights = (0.5, 0.5))
    print('score:', score)
    
  return(candidate[i] + '\n' + reference[i], score)

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
    # p.append(x.view(-1))
  return (torch.cat(p).to(device))
  # y=torch.cat(p)
  # print(y)
  # return y

def loadTokenizer(train_en_file, encparams, train_de_file, decparams):
  en_tok_path = encparams["tokenizer_path"]
  en_tokenizer = BertTokenizerFast(os.path.join(en_tok_path, "vocab.txt"),  truncation=True)
  de_tok_path = decparams["tokenizer_path"]
  de_tokenizer = BertTokenizerFast(os.path.join(de_tok_path, "vocab.txt"),  truncation=True)
  return (en_tokenizer, de_tokenizer)
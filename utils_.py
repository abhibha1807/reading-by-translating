from torchtext.data.metrics import bleu_score
from model2 import *

def pad_sentences(sentence):
  s = sentence.split(' ')
  while len(s)<=MAX_LENGTH:
    s.append('[PAD]')
  print(s)
  return s
  

def get_bleu_score(model,test_inputs, tokenizer, vocab):
    predicted = model.generate(test_inputs[0], vocab)
    actual = tokenizer.decode(list(torch.squeeze(test_inputs[0][1], dim=-1)))
    predicted = pad_sentences(predicted)
    actual = pad_sentences(actual)
    print(predicted)
    print(actual)
    bleu_score(predicted, actual)
from torchtext.data.metrics import bleu_score
from model2 import *
import os
import shutil
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pad_sentences(sentence):
  s = sentence.split(' ')
  while len(s)<=MAX_LENGTH:
    s.append('[PAD]')
  print(s)
  return s
  

def get_bleu_score(model,test_inputs, tokenizer, vocab):
    predicted = model.generate(test_inputs[0], tokenizer, vocab)
    actual = tokenizer.decode(list((test_inputs[1])))
    predicted = pad_sentences(predicted)
    actual = pad_sentences(actual)
    print('predicted sentence:', predicted)
    print('actual sentence:', actual)
    return bleu_score(predicted, actual)

def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)
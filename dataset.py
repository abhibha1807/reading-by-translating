from __future__ import unicode_literals, print_function, division

import re
import torch
from io import open
import unicodedata
import torch
from torch.utils.data import TensorDataset

MAX_LENGTH = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "PAD"}
        self.n_words = 3  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


eng_prefixes = (
    
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def pad_sentences(pairs):
  for i in range(len(pairs)):
    n_en = len(pairs[i][0].split(' '))
    n_fr = len(pairs[i][1].split(' '))
    if n_en < MAX_LENGTH:
      for j in range(n_en, MAX_LENGTH):
        pairs[i][0] = pairs[i][0] + ' PAD'
    if n_fr < MAX_LENGTH:
      for j in range(n_fr, MAX_LENGTH):
        pairs[i][1] = pairs[i][1] + ' PAD'
  return (pairs)


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    pairs = pad_sentences(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

#fucnctions to load sentence pairs.
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair, input_lang, output_lang):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return torch.stack((input_tensor, target_tensor))


def get_train_dataset(pairs, tokenizer):
  attn_idx = torch.arange(len(pairs))
  #print(attn_idx)
  tensor_pairs = []
  for pair in pairs:
    source = torch.unsqueeze(torch.tensor(tokenizer.encode(pair[0]).ids), dim=-1)
    target = torch.unsqueeze(torch.tensor(tokenizer.encode(pair[1]).ids), dim=-1)
    #print(pair[0], pair[1])
    #print(source, target)
    tensor_pairs.append(torch.stack([source, target]))
  #print(tensor_pairs)
  #print(torch.stack((tensor_pairs)).size())
  train_data = TensorDataset(torch.stack((tensor_pairs)), attn_idx)
  #print(train_data)
  return train_data

def get_un_dataset(pairs, tokenizer):
  tensor_pairs = []
  for pair in pairs:
    source = torch.unsqueeze(torch.tensor(tokenizer.encode(pair[0]).ids), dim=-1)
    target = torch.unsqueeze(torch.tensor(tokenizer.encode(pair[1]).ids), dim=-1)
    #print(pair[0], pair[1])
    #print(source, target)
    tensor_pairs.append(torch.stack([source, target]))
  #print(tensor_pairs)
  un_data = TensorDataset(torch.stack((tensor_pairs)))
  #print(un_data)
  return un_data

def get_valid_dataset(pairs, tokenizer):
  tensor_pairs = []
  for pair in pairs:
    source = torch.unsqueeze(torch.tensor(tokenizer.encode(pair[0]).ids), dim=-1)
    target = torch.unsqueeze(torch.tensor(tokenizer.encode(pair[1]).ids), dim=-1)
    #print(pair[0], pair[1])
    #print(source, target)
    tensor_pairs.append(torch.stack([source, target]))
  #print(tensor_pairs)
  valid_data = TensorDataset(torch.stack((tensor_pairs)))
  #print(valid_data)
  return valid_data
  



'''
Sample output:
Reading lines...
Read 135842 sentence pairs
Trimmed to 10599 sentence pairs
Counting words...
Counted words:
fra 4347
eng 2805
['je suis content . PAD PAD PAD PAD PAD PAD', 'i m glad . PAD PAD PAD PAD PAD PAD']
'''
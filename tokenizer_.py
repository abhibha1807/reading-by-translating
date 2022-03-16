import sys
import os
import json
from tokenizers import BertWordPieceTokenizer
from tokenizers.processors import BertProcessing
import argparse


def train_tokenizer(dataset, max_length, min_freq, vocabsize, save_location):
    """
    Train a BertWordPieceTokenizer with the specified params and save it
    """

    print('is the tokenizer trained right??')
    print(dataset[0:2])
    # Get tokenization params
    save_location = save_location
    max_length = max_length
    min_freq = min_freq
    vocabsize = vocabsize

    tokenizer = BertWordPieceTokenizer()
    tokenizer.do_lower_case = False
    special_tokens = ["[S]","[PAD]","[/S]","[UNK]","[MASK]", "[SEP]","[CLS]"]
    tokenizer.train_from_iterator(dataset, vocab_size=vocabsize, min_frequency=min_freq, special_tokens = special_tokens)

    tokenizer._tokenizer.post_processor = BertProcessing(("[SEP]", tokenizer.token_to_id("[SEP]")), ("[CLS]", tokenizer.token_to_id("[CLS]")),)
    tokenizer.enable_truncation(max_length=max_length)
    tokenizer.enable_padding(length=max_length, pad_token='[PAD]')


    print("Saving tokenizer ...")
    if not os.path.exists(save_location):
        os.makedirs(save_location)
    tokenizer.save_model(save_location)
    return tokenizer

def get_tokenizer(pairs, max_length, min_freq, vocabsize, save_location):
    print('training tokenizers')    
    # tokenizer = train_tokenizer([i[0] for i in pairs],  max_length, min_freq, vocabsize, save_location)
    # tokenizer = train_tokenizer([i[1] for i in pairs],  max_length, min_freq, vocabsize, save_location)
    tokenizer = train_tokenizer(pairs,  max_length, min_freq, vocabsize, save_location)
    return tokenizer



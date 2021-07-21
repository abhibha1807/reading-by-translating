import sys
import os
import json
from tokenizers import BertWordPieceTokenizer
from tokenizers.processors import BertProcessing
import argparse
parser = argparse.ArgumentParser("rbt")
parser.add_argument('--config', type=str, default='config.json', help='config file')
args = parser.parse_args()

def train_tokenizer(filename, params):
    """
    Train a BertWordPieceTokenizer with the specified params and save it
    """
    # Get tokenization params
    save_location = params["tokenizer_path"]
    max_length = params["max_length"]
    min_freq = params["min_freq"]
    vocabsize = params["vocab_size"]

    tokenizer = BertWordPieceTokenizer()
    tokenizer.do_lower_case = False
    special_tokens = ["[S]","[PAD]","[/S]","[UNK]","[MASK]", "[SEP]","[CLS]"]
    tokenizer.train(files=[filename], vocab_size=vocabsize, min_frequency=min_freq, special_tokens = special_tokens)

    tokenizer._tokenizer.post_processor = BertProcessing(("[SEP]", tokenizer.token_to_id("[SEP]")), ("[CLS]", tokenizer.token_to_id("[CLS]")),)
    tokenizer.enable_truncation(max_length=max_length)

    print("Saving tokenizer ...")
    if not os.path.exists(save_location):
        os.makedirs(save_location)
    tokenizer.save_model(save_location)


if __name__ == '__main__':
    '''
    Load training data and train BertWordPieceTokenizer
    '''
    configfile = args.config
    with open(configfile, "r") as f:
        config = json.load(f)

    dataset = config["dataset"]
    encparams = config["encoder_params"]
    decparams = config["decoder_params"]

    train_en_file = dataset["train_en_file"]
    train_de_file = dataset["train_de_file"]

    train_tokenizer(train_en_file, encparams)
    train_tokenizer(train_de_file, decparams)


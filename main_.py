import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import BertTokenizerFast
from dataClass import TranslationDataset 
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from Model import model
from utils import createBatchesA
# from trainTokenizer import train_tokenizer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.autograd.set_detect_anomaly(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)


def trainTokenizer(train_en_file, encparams, train_de_file, decparams):
    en_tok_path = encparams["tokenizer_path"]
    en_tokenizer = BertTokenizerFast(os.path.join(en_tok_path, "vocab.txt"))
    de_tok_path = decparams["tokenizer_path"]
    de_tokenizer = BertTokenizerFast(os.path.join(de_tok_path, "vocab.txt"))
    return (en_tokenizer, de_tokenizer)

def run():
    configfile = "config.json"
    # Read the params
    with open(configfile, "r") as f:
        config = json.load(f)
    globalparams = config["global_params"]
    encparams = config["encoder_params"]
    decparams = config["decoder_params"]
    model1params = config["model_params"]
    model2params = config["model_params_"]
    enc_maxlength = encparams["max_length"]
    dec_maxlength = decparams["max_length"]
    batch_size = config["batch_size"]
    model1_path = model1params["model_path"]
    model2_path = model2params["model_path"]
    
    # Get the dataset files
    train_en_file = globalparams["train_en_file"]
    train_de_file = globalparams["train_de_file"]
    valid_en_file = globalparams["valid_en_file"]
    valid_de_file = globalparams["valid_de_file"]

    #train Bert tokenizers
    en_tokenizer, de_tokenizer=trainTokenizer(train_en_file, encparams, train_de_file, decparams)

    mdl=model(model1_path, model2_path, device, batch_size)

    optimizer1 = torch.optim.Adam(mdl.model1.parameters(), lr=model1params['lr'])
    optimizer2 = torch.optim.Adam(mdl.model2.parameters(), lr=model2params['lr'])
    criterion = nn.NLLLoss(ignore_index=de_tokenizer.pad_token_id)

    train_dataset = TranslationDataset(train_en_file, train_de_file, en_tokenizer, de_tokenizer, enc_maxlength, dec_maxlength)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, \
                                            drop_last=True, num_workers=1, collate_fn=train_dataset.collate_function)

    valid_dataset = TranslationDataset(valid_en_file, valid_de_file, en_tokenizer, de_tokenizer, enc_maxlength, dec_maxlength)
    valid_dataloader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, \
                                            drop_last=True, num_workers=1, collate_fn=valid_dataset.collate_function)

    A=torch.rand(len(train_dataset), requires_grad=True, device ='cpu')
    optimizer3 = torch.optim.SGD([A], lr=config["learning_rateA"])

    #create batches for A
    torch.multiprocessing.freeze_support()
    A_batch = DataLoader(createBatchesA(A), batch_size=batch_size)
    writer = SummaryWriter()


    #main training loop
    for epoch in range(config["num_epochs"]):
        print('\n')
        print("Starting epoch", epoch+1)
        epoch_loss1 = mdl.train_model1(A_batch, train_dataloader, optimizer1,criterion, A)
        writer.add_scalar('Loss/model1', epoch_loss1, epoch)
        epoch_loss2 = mdl.train_model2(train_dataloader, optimizer2, criterion)# using the same training dataset for now.
        writer.add_scalar('Loss/model2', epoch_loss2, epoch)
        epoch_loss3 = mdl.val_model2( valid_dataloader, optimizer3, criterion, A, model1params['lr'], model1params['lr'] )
        writer.add_scalar('Loss/val', epoch_loss3, epoch)
    writer.close()

if __name__ == '__main__':
    run()
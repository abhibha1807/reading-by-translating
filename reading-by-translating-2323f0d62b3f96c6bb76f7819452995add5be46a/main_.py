import sys
import os
import time
import glob
import json
import torch
import utils
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataClass import TranslationDataset 
from torch.utils.data import DataLoader
from transformers import BertModel, BertForMaskedLM, BertConfig, EncoderDecoderModel

from torch.utils.tensorboard import SummaryWriter
from Model import TranslationModel
from utils import createBatchesA, loadTokenizer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.autograd.set_detect_anomaly(True)
import logging 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

import logging
from time import gmtime, strftime

logging.root.handlers = []
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.FileHandler("debug"+strftime("%Y-%m-%d_%H:%M:%S", gmtime())+".log"),
        logging.StreamHandler()
    ]
)
parser = argparse.ArgumentParser("rbt")
parser.add_argument('--config', type=str, default='config.json', help='config file')
args = parser.parse_args()

# args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
# utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))


def run():
    '''
    Function responsible for loading data, tokenizers, optimizers, schedulers
    and executing the main training loop 
    '''
    configfile = args.config
    with open(configfile, "r") as f:
        config = json.load(f)
    dataset = config["dataset"]
    # unlabeled = config["unlabeled"]
    encparams = config["encoder_params"]
    decparams = config["decoder_params"]
    model1params = config["model1"]
    model2params = config["model2"]
    enc_maxlength = encparams["max_length"]
    dec_maxlength = decparams["max_length"]
    batch_size = config["batch_size"]
    model1_path = model1params["model_path"]
    model2_path = model2params["model_path"]
    inst=4
    batch_size=1
    
    # Get the dataset files
    train_en_file = dataset["train_en_file"]
    train_de_file = dataset["train_de_file"]
    valid_en_file = dataset["valid_en_file"]
    valid_de_file = dataset["valid_de_file"]
    unlabeled_size=config["unlabeled_percent"]
    unlabeled_size=0.5

    #load BertWordPieceTokenizer
    en_tokenizer, de_tokenizer=loadTokenizer(train_en_file, encparams, train_de_file, decparams)

    #initialize models.
    # model1_path='./saved_models/model1'
    # model2_path='./saved_models/model2'
    mdl=TranslationModel(device, batch_size, logging, model1_path, model2_path, config)

    optimizer1 = torch.optim.Adam(mdl.model1.parameters(), lr=model1params['lr'], weight_decay=model1params['weight_decay'])
    optimizer2 = torch.optim.Adam(mdl.model2.parameters(), lr=model2params['lr'],  weight_decay=model2params['weight_decay'])
    criterion = nn.NLLLoss(ignore_index=de_tokenizer.pad_token_id)

    #initialize matrix A
    A=torch.rand(50000, requires_grad=True, device = device)
    optimizer3 = torch.optim.SGD([A], lr=config["learning_rateA"])
    
    torch.multiprocessing.freeze_support()
    A_batch = DataLoader(createBatchesA(A), batch_size=batch_size)

    writer = SummaryWriter()

    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, float(config['num_epochs']), eta_min=model1params["learning_rate_min"])
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, float(config['num_epochs']), eta_min=model2params["learning_rate_min"])
    scheduler3 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer3, float(config['num_epochs']), eta_min=config["learning_rate_min"])



    #training and validation datasets
    # train_dataset = TranslationDataset(train_en_file, train_de_file, en_tokenizer, de_tokenizer, enc_maxlength, dec_maxlength)
    # valid_dataset = TranslationDataset(valid_en_file, valid_de_file, en_tokenizer, de_tokenizer, enc_maxlength, dec_maxlength)

    # print('before train:', len(train_dataset))
    # print('before valid:', len(valid_dataset))
    # unlabeled_amount = int(len(train_dataset) * unlabeled_size)
    # print('len of u:', unlabeled_amount)
    # print('len of dataset:', len(train_dataset))
    
    #splitting the dataset into unlabeled and training datasets
    # train_set, unlabeled_set = torch.utils.data.random_split(train_dataset, [
    #             (len(train_dataset) - unlabeled_amount), 
    #             unlabeled_amount
    # ])
    print(batch_size)
    print(config["num_epochs"])
    epochs=3

    for epoch in range(epochs):
        start=0
        end=start+inst
        a_ind=0
        for i in range(int(50000/inst)):
            print('instances gone:', inst*(i+1))
            print(start, end)
            train_dataset = TranslationDataset(train_en_file, train_de_file, en_tokenizer, de_tokenizer, enc_maxlength, dec_maxlength, start, end, inst)
            valid_dataset = TranslationDataset(valid_en_file, valid_de_file, en_tokenizer, de_tokenizer, enc_maxlength, dec_maxlength, start, end, inst)
            start=start+inst
            end=end+inst

            print('before train:', len(train_dataset))
            print('before valid:', len(valid_dataset))
            unlabeled_amount = int(len(train_dataset) * unlabeled_size)
            print('len of u:', unlabeled_amount)
            print('len of dataset:', len(train_dataset))


            #splitting the dataset into unlabeled and training datasets
            train_set, unlabeled_set = torch.utils.data.random_split(train_dataset, [
                        (len(train_dataset) - unlabeled_amount), 
                        unlabeled_amount
            ])

            if len(train_set)>0 and len(valid_dataset)>0 and len(unlabeled_set)>0:

                train_dataloader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False, \
                                                        drop_last=True, num_workers=1, collate_fn=train_dataset.collate_function)

                unlabeled_dataloader = torch.utils.data.DataLoader(dataset=unlabeled_set, batch_size=batch_size, shuffle=False, \
                                                        drop_last=True, num_workers=1, collate_fn=train_dataset.collate_function)


                valid_dataloader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, \
                                                        drop_last=True, num_workers=1, collate_fn=valid_dataset.collate_function)
                
                print('train:', len(train_set))
                print('unlabeled:', len(unlabeled_set))
                print('valid:', len(valid_dataset))
            
                #main training loop
                
                print('\n')
                t = torch.cuda.get_device_properties(0).total_memory
                r = torch.cuda.memory_reserved(0) 
                al = torch.cuda.memory_allocated(0)
                f = r-al  # free inside reserved
                print('freeeee:', f)
            
                epoch_loss1 = mdl.train_model1(A_batch, train_dataloader, optimizer1, de_tokenizer, criterion, scheduler1)
                writer.add_scalar('Loss/model1', epoch_loss1, epoch)
                t = torch.cuda.get_device_properties(0).total_memory
                r = torch.cuda.memory_reserved(0) 
                al = torch.cuda.memory_allocated(0)
                f = r-al  # free inside reserved
                print('freeeee:', f)
                epoch_loss2 = mdl.train_model2(unlabeled_dataloader, optimizer2, de_tokenizer, criterion, scheduler2)# using the same training dataset for now.
                writer.add_scalar('Loss/model2', epoch_loss2, epoch)
                t = torch.cuda.get_device_properties(0).total_memory
                r = torch.cuda.memory_reserved(0) 
                al = torch.cuda.memory_allocated(0)
                f = r-al  # free inside reserved
                print('freeeee:', f)
                epoch_loss3, a_ind = mdl.val_model2( valid_dataloader, optimizer3, A, A_batch , de_tokenizer, criterion, scheduler3, a_ind)
                writer.add_scalar('Loss/val', epoch_loss3, epoch)
            if (inst*(i+1))%100 == 0:
                print('saving model after'+str((inst*(i+1)))+'instances')
                mdl.save_model(config['model_path'])

                # model1_path=config["model1"]["saved_model_path"]
                # model2_path=config["model2"]["saved_model_path"]
                # model1_path='./saved_models/model1'
                # model2_path='./saved_models/model2'
                # mdl=TranslationModel(device, batch_size, logging, model1_path, model2_path, config)

    writer.close()

if __name__ == '__main__':
    run()
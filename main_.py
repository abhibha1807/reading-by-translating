import sys
import os
import time
import glob
import json
import torch
import utils
import gc
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataClass import TranslationDataset 
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from Model import TranslationModel
from utils import createBatchesA, loadTokenizer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.autograd.set_detect_anomaly(True)
import logging 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
from pynvml import *
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
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

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
    
    # Get the dataset files
    train_en_file = dataset["train_en_file"]
    train_de_file = dataset["train_de_file"]
    valid_en_file = dataset["valid_en_file"]
    valid_de_file = dataset["valid_de_file"]
    unlabeled_size=config["unlabeled_percent"]
    
    #load BertWordPieceTokenizer
    en_tokenizer, de_tokenizer=loadTokenizer(train_en_file, encparams, train_de_file, decparams)

    #initialize models.
    mdl=TranslationModel(device, batch_size, logging, config)

    optimizer1 = torch.optim.Adam(mdl.model1.parameters(), lr=model1params['lr'], weight_decay=model1params['weight_decay'])
    optimizer2 = torch.optim.Adam(mdl.model2.parameters(), lr=model2params['lr'],  weight_decay=model2params['weight_decay'])
    criterion = nn.NLLLoss(ignore_index=de_tokenizer.pad_token_id)

    #training and validation datasets
    train_dataset = TranslationDataset(train_en_file, train_de_file, en_tokenizer, de_tokenizer, enc_maxlength, dec_maxlength)
    valid_dataset = TranslationDataset(valid_en_file, valid_de_file, en_tokenizer, de_tokenizer, enc_maxlength, dec_maxlength)

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

    train_dataloader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False, \
                                            drop_last=True, num_workers=1, collate_fn=train_dataset.collate_function)

    unlabeled_dataloader = torch.utils.data.DataLoader(dataset=unlabeled_set, batch_size=batch_size, shuffle=False, \
                                            drop_last=True, num_workers=1, collate_fn=train_dataset.collate_function)


    valid_dataloader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, \
                                            drop_last=True, num_workers=1, collate_fn=valid_dataset.collate_function)
    
    print('train:', len(train_set))
    print('unlabeled:', len(unlabeled_set))
    print('valid:', len(valid_dataset))

    #initiliaze matrix A
    A=torch.rand(len(train_dataset), requires_grad=True, device=device)
    print('len A:', len(A))
    optimizer3 = torch.optim.SGD([A], lr=config["learning_rateA"])
    torch.multiprocessing.freeze_support()
    A_batch = DataLoader(createBatchesA(A), batch_size=batch_size)
    
    writer = SummaryWriter()
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, float(config['num_epochs']), eta_min=model1params["learning_rate_min"])
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, float(config['num_epochs']), eta_min=model2params["learning_rate_min"])
    scheduler3 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer3, float(config['num_epochs']), eta_min=config["learning_rate_min"])
    

    #main training loop
    for epoch in range(config["num_epochs"]):

        print('\n')
        print("Starting epoch", epoch+1)
      
        epoch_loss1 = mdl.train_model1(A_batch, train_dataloader, optimizer1, de_tokenizer, criterion, scheduler1)
        writer.add_scalar('Loss/model1', epoch_loss1, epoch)

       
        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(h)
        print(f'total    : {info.total}')
        print(f'free     : {info.free}')
        print(f'used     : {info.used}')

        # mdl.model2 =  mdl.model2.cuda()
        epoch_loss2 = mdl.train_model2(unlabeled_dataloader, optimizer2, de_tokenizer, criterion, scheduler2)# using the same training dataset for now.
        writer.add_scalar('Loss/model2', epoch_loss2, epoch)
        
      
        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(h)
        print(f'total    : {info.total}')
        print(f'free     : {info.free}')
        print(f'used     : {info.used}')

        epoch_loss3 = mdl.val_model2(valid_dataloader, optimizer3, A, A_batch , de_tokenizer, criterion, scheduler3)
        writer.add_scalar('Loss/val', epoch_loss3, epoch)
        
        # mdl.save_model(config['model_path'])

    writer.close()

if __name__ == '__main__':
    run()
#!/usr/bin/env python
# coding: utf-8

# In[2]:


# !nvidia-smi


# In[1]:




# In[1]:


import operator, functools
from queue import PriorityQueue
import time
import numpy as np
import torch
import random
from torch.optim import lr_scheduler
from torch import optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, default_data_collator
import datasets
from datasets import *
import os
import sys
import logging
from torch.utils.data import TensorDataset
from sacremoses import MosesPunctNormalizer, MosesTokenizer
from tokenizers import Tokenizer
import nltk
import math
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from accelerate import Accelerator

nltk.download('punkt')


# In[2]:


class parse_args():
    def __init__(self):
        self.dataset_name='wmt14_gigafren'
        self.predict_with_generate=True
        self.dataset_config_name='fr-en'
        self.num_beams=2
        self.max_source_length=50

        self.max_target_length=50
        self.val_max_target_length=15
        self.pad_to_max_length=False
        self.ignore_pad_token_for_loss=True
#         self.source_lang='en_XX'
#         self.target_lang='ro_RO'
        self.preprocessing_num_workers=4
        self.max_length=15
        self.per_device_train_batch_size=50
        self.per_device_eval_batch_size=50
        self.learning_rate=5e-5
        self.weight_decay=0.0
        self.num_train_epochs=5
        self.max_train_steps=10000
        self.gradient_accumulation_steps=1
        self.lr_scheduler_type='linear'
        self.num_warmup_steps=0


# In[3]:


def preprocess_function(examples):
    inputs = [ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    
    english_output = tokenizer_english.encode_batch(inputs,add_special_tokens=True)
    french_output = tokenizer_french.encode_batch(targets,add_special_tokens=True)
    
    model_inputs = {}
    model_inputs["english"] = [e.ids for e in english_output]
    model_inputs["french"] = [f.ids for f in french_output]
    return model_inputs


# In[4]:


log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join('./', 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


# In[5]:


args=parse_args()


# In[6]:


tokenizer_english = Tokenizer.from_file("english_tokenizer_enfr.json")
tokenizer_french = Tokenizer.from_file("french_tokenizer_enfr.json")


# In[7]:


class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward


# In[8]:


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)
        
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        
        #src = [src len, batch size]
        
        embedded = self.dropout(self.embedding(src))
        
        #embedded = [src len, batch size, emb dim]
        
        outputs, hidden = self.rnn(embedded)
                
        #outputs = [src len, batch size, hid dim * num directions]
        #hidden = [n layers * num directions, batch size, hid dim]
        
        #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        #outputs are always from the last layer
        
        #hidden [-2, :, : ] is the last of the forwards RNN 
        #hidden [-1, :, : ] is the last of the backwards RNN
        
        #initial decoder hidden is final hidden state of the forwards and backwards 
        #  encoder RNNs fed through a linear layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        
        #outputs = [src len, batch size, enc hid dim * 2]
        #hidden = [batch size, dec hid dim]
        
        return outputs, hidden


# In[9]:


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        #repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        
        #energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)
        
        #attention= [batch size, src len]
        
        return F.softmax(attention, dim=1)


# In[10]:


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
        
        #input = [batch size]
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        
        input = input.unsqueeze(0)
        
        #input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input))
        
        #embedded = [1, batch size, emb dim]
        
        a = self.attention(hidden, encoder_outputs)
                
        #a = [batch size, src len]
        
        a = a.unsqueeze(1)
#         print(a.shape)
        #a = [batch size, 1, src len]
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
#         print(encoder_outputs.shape)
        weighted = torch.bmm(a, encoder_outputs)
        
        #weighted = [batch size, 1, enc hid dim * 2]
        
        weighted = weighted.permute(1, 0, 2)
        
        #weighted = [1, batch size, enc hid dim * 2]
#         print(weighted.shape)
#         print(embedded.shape)
        
        rnn_input = torch.cat((embedded, weighted), dim = 2)
        
        #rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]
            
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        
        #output = [seq len, batch size, dec hid dim * n directions]
        #hidden = [n layers * n directions, batch size, dec hid dim]
        
        #seq len, n layers and n directions will always be 1 in this decoder, therefore:
        #output = [1, batch size, dec hid dim]
        #hidden = [1, batch size, dec hid dim]
        #this also means that output == hidden
        assert (output == hidden).all()
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden.squeeze(0)


# In[11]:


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
#         src=src.permute(1,0)
#         trg=src.permute(1,0)
        #src = [src_len,batch_size ]
        #trg = [trg_len,batch_size ]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
        
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #encoder_outputs is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src)
                
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        for t in range(1, trg_len):
            #insert input token embedding, previous hidden state and all encoder hidden states
            #receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs


# In[12]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[13]:


tokenizer_english.pad_token_id=3
tokenizer_french.pad_token_id=3


# In[14]:


def collate_batch(batch):
  
    target_list, source_list, = [], []
  
    for b in batch:
        target_list.append(torch.tensor(b['french'],dtype=torch.int64))
        source_list.append(torch.tensor(b['english'],dtype=torch.int64))
  
    target_list = pad_sequence(target_list, batch_first=True, padding_value=tokenizer_french.pad_token_id)
    source_list = pad_sequence(source_list, batch_first=True, padding_value=tokenizer_english.pad_token_id)

    return source_list.to(device),target_list.to(device)


# In[15]:


beam_width = 5
topk = 1  # how many sentence do you want to generate
EOS_token = tokenizer_french.token_to_id('[SEP]')
SOS_token = tokenizer_french.token_to_id('[CLS]')
MAX_LENGTH=100


# In[16]:


class fun1():
    def __init__(self,loss,outputs):
        self.loss= loss
        self.output = outputs 


# In[17]:


class RNN_MODEL(nn.Module):
    def __init__(self,criterion,tokenizer_english,tokenizer_french,MODEL=None):
        
        super(RNN_MODEL, self).__init__()
        INPUT_DIM = tokenizer_english.get_vocab_size()
        OUTPUT_DIM = tokenizer_french.get_vocab_size()
        ENC_EMB_DIM = 256
        DEC_EMB_DIM = 256
        ENC_HID_DIM = 512
        DEC_HID_DIM = 512
        ENC_DROPOUT = 0.5
        DEC_DROPOUT = 0.5
        
        
        self._criterion = criterion
        self.tokenizer_english = tokenizer_english
        self.tokenizer_french = tokenizer_french
        attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
        enc  = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
        dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)
        self.model = Seq2Seq(enc, dec, device).to(device)
        self.model.apply(init_weights)
        
        if MODEL is not None:
            self.model.load_state_dict(torch.load(MODEL))
        
#         self.enc = self.model.encoder
#         self.dec = self.model.decoder
        
    def forward(self,src,trg,teacher_forcing_ratio=0.5):
#         src = src.permute(1,0)
#         trg = trg.permute(1,0)
        
        output = self.model(src, trg, teacher_forcing_ratio)
        batch_size = trg.shape[1]
        output_dim = output.shape[-1]
        
        output = output[1:].view(-1, output_dim)
        
        trg = trg[1:].reshape(-1)
#         print(output.shape)
#         print(trg.shape)
        loss_vec = self._criterion(output,trg)
        
        
        
        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]
        return fun1(loss_vec,output)
    
    def get_loss_vec(self,src,trg,teacher_forcing_ratio=0.5):
#         output = self(src,trg,teacher_forcing_ratio).output
        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]
#         batch_size = trg.shape[1]
#         output_dim = output.shape[-1]
        
#         output = output[1:].view(-1, output_dim)
        
#         trg = trg[1:].reshape(-1)
# #         print(output.shape)
# #         print(trg.shape)
#         loss_vec = self._criterion(output,trg)
        
        
#         loss_vec = loss_vec.view(batch_size, -1).mean(dim = 1)
        loss_vec = self(src,trg,teacher_forcing_ratio).loss
        return loss_vec
        
        
    def generate(self,batch,beam_width=5):
        src = batch[0]
        trg = batch[1]
        src = src.permute(1,0)
        trg = trg.permute(1,0)
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = dec.output_dim

        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(device)

        #encoder_outputs is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.model.encoder(src)
        if beam_width>1:
            decoded_batch = []
            #     :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
            #     :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
            #     :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
            #     :return: decoded_batch
            # decoding goes sentence by sentence
            for idx in range(trg.size(1)):
                decoder_hidden = hidden[idx, :].unsqueeze(0)
                encoder_output = encoder_outputs[:,idx, :].unsqueeze(1)
                # Start with the start of the sentence token
                decoder_input = torch.LongTensor([[SOS_token]]).cuda()
                endnodes = []
                number_required = min((topk + 1), topk - len(endnodes))
                # starting node -  hidden vector, previous node, word id, logp, length
                node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
                nodes = PriorityQueue()
                # start the queue
                nodes.put((-node.eval(), node))
                qsize = 1
                # start beam search
                while True:
                    # give up when decoding takes too long
                    if qsize > 2000: break
                    # fetch the best node
                    score, n = nodes.get()
                    decoder_input = n.wordid
                    decoder_input = decoder_input.squeeze(0)
                    decoder_hidden = n.h
                    if n.wordid.item() == EOS_token and n.prevNode != None:
                        endnodes.append((score, n))
                        # if we reached maximum # of sentences required
                        if len(endnodes) >= number_required:
                            break
                        else:
                            continue
                    # decode for one step using decoder
                    decoder_output, decoder_hidden = self.model.decoder(decoder_input, decoder_hidden, encoder_output)

                    # PUT HERE REAL BEAM SEARCH OF TOP
                    log_prob, indexes = torch.topk(decoder_output, beam_width)
                    nextnodes = []
                    for new_k in range(beam_width):
                        decoded_t = indexes[0][new_k].view(1, -1)
                        log_p = log_prob[0][new_k].item()

                        node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                        score = -node.eval()
                        nextnodes.append((score, node))


                    # put them into queue
                    for i in range(len(nextnodes)):
                        score, nn = nextnodes[i]
                        nodes.put((score, nn))
                        # increase qsize
                    qsize += len(nextnodes) - 1
                # choose nbest paths, back trace them
                if len(endnodes) == 0:
                    endnodes = [nodes.get() for _ in range(topk)]
                utterances = []
                for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                    utterance = []
                    utterance.append(n.wordid)
                    # back trace
                    while n.prevNode != None:
                        n = n.prevNode
                        utterance.append(n.wordid)

                    utterance = utterance[::-1]
                    utterances.append(utterance)

                decoded_batch.append(utterances)

            l= torch.full((batch_size,MAX_LENGTH),3)
            for i in range(len(decoded_batch)):
                l[i,:len(decoded_batch[i][0])]=torch.tensor([r[0].cpu().numpy()[0] for r in decoded_batch[i][0]][:MAX_LENGTH])
            return l
        else:
            decoder_hidden = hidden
            decoded_batch = torch.zeros((batch_size, MAX_LENGTH))
            decoder_input = trg[0,:]
            for t in range(MAX_LENGTH):
                decoder_output, decoder_hidden = self.model.decoder(decoder_input, decoder_hidden, encoder_outputs)

                topv, topi = decoder_output.data.topk(1)  # get candidates
                topi = topi.view(-1)
                decoded_batch[:, t] = topi

                decoder_input = topi.detach().view(-1)
        return decoded_batch
    
    
    # new model for the definitions of gradients in architec.py
    def new(self):
        model_new = RNN_MODEL(self._criterion,self.tokenizer_english,self.tokenizer_french)
        model_new.model.load_state_dict(self.model.state_dict())
        return model_new


# In[18]:


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


# In[19]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# In[20]:


# Not using in current implementation
def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        
        src = batch[0]
        trg = batch[1]
        src = src.permute(1,0)
        trg = trg.permute(1,0)
        
        optimizer.zero_grad()
        
        output = model(src, trg)
        
        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]
        
        output_dim = output.shape[-1]
        
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].reshape(-1)
        
        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]
        
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


# In[21]:


def evaluate(rnn_model, iterator, criterion):
    
    rnn_model.model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src = batch[0]
            trg = batch[1]
            src = src.permute(1,0)
            trg = trg.permute(1,0)

            loss = rnn_model(src, trg, 0).loss #turn off teacher forcing

            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]

#             output_dim = output.shape[-1]
            
#             output = output[1:].view(-1, output_dim)
#             trg = trg[1:].reshape(-1)

#             #trg = [(trg len - 1) * batch size]
#             #output = [(trg len - 1) * batch size, output dim]

#             loss = criterion(output, trg)

            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


# In[22]:


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


# In[23]:


losses_valid = AverageMeter('Loss', ':.4e')
losses_test = AverageMeter('Loss', ':.4e')
losses_train = AverageMeter('Loss', ':.4e')

total_valid=0.0
completed_steps=0

perplexity_valid = AverageMeter('Perplexity', ':.4e')
perplexity_test = AverageMeter('Perplexity', ':.4e')
perplexity_train = AverageMeter('Perplexity', ':.4e')


# ### Train

# In[24]:


dataset = load_dataset("wmt14_gigafren", "fr-en")
# dataset = load_dataset('wmt14','fr-en',cache_dir="/abhisingh-volume/fr_en_dataset")
# prefix = args.source_prefix if args.source_prefix is not None else ""

# Preprocessing the datasets.
# First we tokenize all the texts.
column_names = dataset["train"].column_names
mt = MosesTokenizer(lang='fr')
# DataLoaders creation:
label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
max_target_length = args.max_target_length
padding = "max_length" if args.pad_to_max_length else False
source_lang='en'
target_lang='fr'
tokenizers={}
tokenizers['en']=tokenizer_english
tokenizers['fr']=tokenizer_french
args.preprocessing_num_workers=8
tokenizer_english.enable_truncation(max_length=100)
tokenizer_french.enable_truncation(max_length=100)
processed_datasets = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=args.preprocessing_num_workers,
    remove_columns=column_names,
    desc="Running tokenizer on dataset",
)


# In[25]:


example=processed_datasets['train'][0]['english']


# In[26]:


tokenizer_english.decode_batch([example])


# In[27]:


tokenizer_english.decode(tokenizer_english.encode('SAN FRANCISCO – It has never been easy to have a rational conversation about the value of gold.').ids,skip_special_tokens=True)


# In[28]:


tokenizer_english.decode(processed_datasets['train'][0]['english'])


# In[29]:


tokenizer_french.decode(processed_datasets['train'][0]['french'])


# In[30]:


train_dataset = processed_datasets["train"]
eval_dataset = processed_datasets["validation"]
test_dataset = processed_datasets["test"]


# In[31]:


train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=collate_batch, batch_size=args.per_device_train_batch_size
)
valid_dataloader = DataLoader(eval_dataset, collate_fn=collate_batch, batch_size=args.per_device_eval_batch_size)
test_dataloader = DataLoader(test_dataset, collate_fn=collate_batch, batch_size=args.per_device_eval_batch_size)


# In[32]:


next(iter(train_dataloader))[1].shape


# In[33]:


criterion = nn.CrossEntropyLoss(ignore_index = tokenizer_french.pad_token_id,reduction='mean')


# In[34]:


rnn_model = RNN_MODEL(criterion,tokenizer_english,tokenizer_french)


# In[35]:


print(f'The model has {count_parameters(rnn_model.model):,} trainable parameters')


# In[36]:


optimizer = optim.Adam(rnn_model.model.parameters())


# In[37]:


scheduler = lr_scheduler.ExponentialLR(optimizer,gamma=0.9,verbose=True)


# In[38]:


teacher_forcing_ratio = 0.5


# In[39]:


num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
if args.max_train_steps is None:
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
else:
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)


# In[40]:


len(train_dataloader)


# In[41]:


args.max_train_steps


# In[42]:


progress_bar = tqdm(range(args.max_train_steps),position=0,leave=True)


# In[43]:


checkpoint_steps = 1000
best_valid_loss = float('inf')
completed_steps = 0


# In[44]:


clip = 1


# In[ ]:



for epoch in range(args.num_train_epochs):
    rnn_model.model.train()
    start_time = time.time()
    for step, batch in enumerate(train_dataloader):
        step = step+1
        src = batch[0]
        trg = batch[1]
        src = src.permute(1,0)
        trg = trg.permute(1,0)
        
        loss = rnn_model(src, trg).loss
        
        loss.backward()
        
        losses_train.update(loss.item(), src.size(1))
        if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
            torch.nn.utils.clip_grad_norm_(rnn_model.model.parameters(), clip)
            optimizer.step()
#             scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            completed_steps += 1
            
        if completed_steps%checkpoint_steps==0 or step == len(train_dataloader) - 1:
            scheduler.step()
            valid_loss = evaluate(rnn_model, valid_dataloader, criterion)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(rnn_model.model.state_dict(), 'tut3-model.pt')
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            print(f'Steps: {completed_steps} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {losses_train.avg:.3f} | Train PPL: {math.exp(losses_train.avg):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
            
            logging.info(f"{'Steps':^7} | {'Time':^12}")
            logging.info(f"{completed_steps:^7} | {epoch_mins:^6} {'m':^1} {epoch_secs:^6} {'s':^1}")
            logging.info("-"*70)
            logging.info(f"{'Train Loss':^8} | {'Train Perplexiety':^14}")
            logging.info(f"{losses_train.avg:^6.2} | {math.exp(losses_train.avg):^12.2}")
            logging.info(f"{'Validation Loss':^8} | {'Validation Perplexiety':^14}")
            logging.info(f"{valid_loss:^6.2} | {math.exp(valid_loss):^12.2}")

            start_time = end_time
            losses_train.reset()
            rnn_model.model.train()
            
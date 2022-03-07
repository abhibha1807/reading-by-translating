from spacy import Vocab
import utils
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, SubsetRandomSampler
from torch.autograd import Variable
from dataset import *
from architect import *
from Enc_Dec import *
from model1 import *
from model2 import *
from tokenizer_ import *
import random
from torch.autograd import Variable
from torch.optim import SGD
from attention_params import *
import argparse
import logging

# TASK: French (source) -> English (target)
# args 
parser = argparse.ArgumentParser("Document_summarization")


parser.add_argument('--begin_epoch', type=float, default=0, help='PC Method begin')
parser.add_argument('--stop_epoch', type=float, default=5, help='Stop training on the framework')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')

parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=25, help='num of training epochs')
parser.add_argument('--seed', type=int, default=seed_, help='random seed')

parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.25, help='portion of training data')

parser.add_argument('--A_learning_rate', type=float, default=3e-4, help='learning rate for A')


max_length = 10
vocabsize = 5000
save_location = ''
min_freq = 2
train_portion = 0.9
un_portion = 0.5
batch_size = 2
hidden_size = 256
model1_lr = 0.1
model2_lr = 0.1
model1_mom, model1_wd, A_wd, A_lr, model2_wd, model2_mom = 0.1,0.1, 0.1,0.1, 0.1,0.1 # dummy values
args = parser.parse_args()


#load data
input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print(random.choice(pairs))

#train tokenizer
tokenizer = get_tokenizer(pairs, max_length, min_freq, vocabsize, save_location)


#define models

vocab = tokenizer.get_vocab_size()
print('vocab:', vocab)
criterion = nn.NLLLoss()
model1 = Model1(vocab, vocab)
model2 = Model2(input_size = vocab, output_size = vocab)
model1_optim = SGD(model1.parameters(), lr=model1_lr)
model2_optim = SGD(model2.parameters(), lr=model2_lr)


#split 80% train 10% val 10% test
n = len(pairs)
print(n)
train_index = int(train_portion * n)
print(train_index)
valid_index = int(0.5 * train_index)
print(valid_index)
train_portion = pairs[0:valid_index]
un_portion = pairs[valid_index : train_index]
test_portion = pairs[train_index:]
print(len(train_portion), len(un_portion), len(test_portion))

train_data = get_train_dataset(train_portion[0:4], tokenizer)
un_data = get_un_dataset(un_portion[0:4], tokenizer)
test_data = get_test_dataset(test_portion[0:4], tokenizer)



train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), 
                        batch_size=batch_size, pin_memory=True, num_workers=0)

test_dataloader = DataLoader(test_data, sampler=RandomSampler(test_data), 
                      batch_size=batch_size, pin_memory=True, num_workers=0)

un_dataloader = DataLoader(un_data, sampler=RandomSampler(un_data), 
                        batch_size=batch_size, pin_memory=True, num_workers=0)


#define A
A = attention_params(train_portion[0:4])
print('A:', list(A.parameters()))




architect = Architect(model1, model1_mom, model1_wd, A, A_lr, A_wd, device, model2, model2_wd, model2_mom, batch_size,vocab)


def train(epoch, train_dataloader):

  for step, batch in enumerate(train_dataloader):
    model1.train()
    
    train_inputs = batch[0] #train inputs.
    idxs = batch[1] #A
    un_batch = next(iter(un_dataloader)) 
    un_inputs = un_batch[0]
    val_batch = next(iter(test_dataloader)) 
    val_inputs = val_batch[0]


    if args.begin_epoch <= epoch <= args.stop_epoch:
      architect.step(train_inputs, un_inputs, val_inputs, model1_lr, A, idxs, criterion, model2_lr, model2_optim, model1_optim)
    
    if epoch <= args.stop_epoch:
      model1_optim.zero_grad()
      loss_model1 = loss1(train_inputs, model1, idxs, A,batch_size, vocab)
      
      # store the batch loss
      batch_loss_model1 += loss_model1.item()

      loss_model1.backward()
      
      nn.utils.clip_grad_norm(model1.parameters(), args.grad_clip)
      
      model1_optim.step()
      
      # ######################################################################
      # # Update model2 model
      model2_optim.zero_grad()
      #un inputs
      loss_model2 = loss2(train_inputs, model1, model2, batch_size, vocab)
      
      # # store the batch loss
      batch_loss_model2 += loss_model2.item()

      loss_model2.backward()
      
      nn.utils.clip_grad_norm(model2.parameters(), args.grad_clip)
      
      model2_optim.step()
      #assess predictions
      model2.generate(val_inputs[0], tokenizer, vocab)
      model1.generate(val_inputs[0], tokenizer, vocab)


      
      if step % args.report_freq == 0:
  
        logging.info(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc':^10}")
        
        logging.info("-"*70)
        
        logging.info(f"{epoch + 1:^7} | {top1.avg:^7} | {top1.avg:^12.6f}")

      #break


      
    
epoch = 0

train(epoch, train_dataloader)   
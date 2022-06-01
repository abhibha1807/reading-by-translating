
   
import utils
from torch.utils.data import DataLoader, RandomSampler
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
from utils_ import *
import time
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
# from torchnlp.datasets import multi30k_dataset 
import glob
# TASK: French (source) -> English (target)
# args 
parser = argparse.ArgumentParser("main")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torchtext.datasets import Multi30k


print('using device', device)

print('eecuting Attn Decoder')
parser.add_argument('--begin_epoch', type=float, default=0, help='PC Method begin')
parser.add_argument('--stop_epoch', type=float, default=5, help='Stop training on the framework')
parser.add_argument('--report_freq', type=float, default=10, help='report frequency')

parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')

parser.add_argument('--batch_size', type=int, default=10, help='batch size')
####################################################################################
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--A_lr', type=float, default=3e-4, help='learning rate for A')


parser.add_argument('--A_wd', type=float, default=0, help=' weight decay for A')


parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--seed', type=int, default=100, help='random seed')
parser.add_argument('--max_length', type=int, default=10, help='max length of sentences')
parser.add_argument('--vocabsize', type=int, default=4000, help='total vocab size')
parser.add_argument('--save_location', type=str, default='./reading-by-translating/', help='save location')
parser.add_argument('--min_freq', type=int, default=5, help='min freq of words to be included in vocab')
parser.add_argument('--train_portion', type=float, default=0.9, help='fraction of dataset for training')

parser.add_argument('--un_portion', type=float, default=0.5, help='fraction of training dataset for creating unlabled dataset')

parser.add_argument('--hidden_size', type=int, default=256, help='hidden size')

parser.add_argument('--model1_lr', type=float, default=1e-3, help='model1 starting lr')
parser.add_argument('--model1_lr_min', type=float, default=5e-4, help='model1 min lr')
parser.add_argument('--model2_lr', type=float, default=1e-8, help='model2 starting lr')
parser.add_argument('--model2_lr_min', type=float, default=5e-9, help='model2 min lr')


parser.add_argument('--model1_wd', type=float, default=0, help='model1 weight decay')
parser.add_argument('--model2_wd', type=float, default=0, help='model2 weight decay')
# parser.add_argument('--model1_mom', type=float, default=0.9, help='model1 momentum')
# parser.add_argument('--model2_mom', type=float, default=0.9, help='model2 momentum')
parser.add_argument('--model1_mom', type=float, default=0.0, help='model1 momentum')
parser.add_argument('--model2_mom', type=float, default=0.0, help='model2 momentum')

parser.add_argument('--save', type=str, default='EXP', help='experiment name')


args = parser.parse_args()

max_length = args.max_length
vocabsize = args.vocabsize
save_location = args.save_location
min_freq = args.min_freq
train_portion = args.train_portion
un_portion = args.un_portion
batch_size = args.batch_size
hidden_size = args.hidden_size
model1_lr = args.model1_lr
model2_lr = args.model2_lr
model1_lr_min = args.model1_lr_min
model2_lr_min = args.model1_lr_min
A_lr = args.A_lr
model1_wd = args.model1_wd
model2_wd = args.model2_wd
model1_mom = args.model1_mom
model2_mom = args.model2_mom
A_wd = args.A_wd
report_freq = args.batch_size
print('running basic.py')
args.save = '{}-{}-test-30k'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
print('saving in:', str(args.save))
writer = SummaryWriter('runs/'+str(args.save))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


#Setting the seeds
np.random.seed(args.seed)
torch.cuda.set_device(args.gpu)
cudnn.benchmark = True
torch.manual_seed(args.seed)
cudnn.enabled=True
torch.cuda.manual_seed(args.seed)
logging.info('gpu device = %d' % args.gpu)
logging.info("args = %s", args)

#load data
# input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
# print(random.choice(pairs))

train_iter = Multi30k(split='train')

valid_iter = Multi30k(split='valid')

test_iter = Multi30k(split='test')

pairs = []
train_pairs = []
valid_pairs = []
test_pairs = []

for label, line in train_iter:
    pairs.append([label,line])
    train_pairs.append([label,line])

for label, line in valid_iter:
    pairs.append([label,line])
    valid_pairs.append([label, line])

for label, line in test_iter:
    pairs.append([label,line])
    test_pairs.append([label, line])




#train tokenizer
tokenizer = get_tokenizer(pairs, max_length, min_freq, vocabsize, save_location)
    
input_lang, output_lang, pairs = prepareData(pairs, 'dutch', 'english', True)
print(pairs[0:5])

#define models

vocab = tokenizer.get_vocab_size()
print('vocab:', vocab)
criterion = nn.NLLLoss(ignore_index = tokenizer.token_to_id("[PAD]"), reduction='none')
criterion = criterion.to(device)
model1 = Model1(vocab, vocab, criterion)
model1 = model1.to(device)

model1_optim = torch.optim.Adam(model1.parameters(),lr=model1_lr,weight_decay=model1_wd) #  loss decreased and then inc 



scheduler_model1  = torch.optim.lr_scheduler.CosineAnnealingLR(model1_optim, float(args.epochs), eta_min=args.model1_lr_min)


# Print the number of training valid and test examples
logging.info('dataset')


train_data = get_train_dataset(train_pairs, tokenizer)
# un_data = get_un_dataset(val_pairs, tokenizer)
valid_data = get_valid_dataset(valid_pairs, tokenizer)

# logging.info(f"{len(train_data):^7} | { len(un_data):^7} | { len(valid_data):^7}")

logging.info(f"{len(train_data):^7} | { len(valid_data):^7}")

train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), 
                        batch_size=batch_size, pin_memory=True, num_workers=0)

valid_dataloader = DataLoader(valid_data, sampler=RandomSampler(valid_data), 
                      batch_size=batch_size, pin_memory=True, num_workers=0)



#define A
A = attention_params(len(train_data))

print('A:', list(A.parameters()))
A = A.to(device)


def train(epoch, train_dataloader, valid_dataloader, A, model1, model1_optim, model1_lr,  instances_gone):



  batch_loss_model1, batch_loss_model2, batch_count = 0, 0, 0
  valid_batch_loss = 0
  model1_score, model2_score = 0, 0
  

    
  for step, batch in enumerate(train_dataloader):
    model1.train()
  
    train_inputs = Variable(batch[0], requires_grad=False).to(device) #train inputs.
    idxs = Variable(batch[1],requires_grad=False).to(device) #A

    val_batch = next(iter(valid_dataloader)) 
    val_inputs = Variable(val_batch[0], requires_grad=False).to(device)

   


 
      
    #logging.info('otherwise')
    model1_optim.zero_grad()
    loss_model1 = loss1(train_inputs, model1, idxs, A, batch_size, vocab)
    print('training loss model1:', loss_model1)
    
    # store the batch loss
    batch_loss_model1 += loss_model1.item()
    a = list(model1.parameters())[0].clone()
    loss_model1.backward()
    print('model1 enc mbeding grad:', model1.enc.embedding.weight.grad.data.sum())
    print(list(model1.parameters())[0].grad)
   
    nn.utils.clip_grad_norm_(model1.parameters(), args.grad_clip)
    
    model1_optim.step()
    b = list(model1.parameters())[0].clone()
    print('\n')
    print('are wts being updated??')
    print(torch.equal(a.data, b.data))

    instances_gone+= batch_size
    '''
    :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
    :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
    :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
    :return: decoded_batch
    '''
   

    if instances_gone % report_freq == 0:
 
      print('-'*40+'training batch stats after'+str(instances_gone)+'instances'+'-'*40)
    
      print("-"*70)
      print('train bleu score')
      model1_score, pred_model1, actual_model1 = get_bleu_score(model1,train_inputs[0], tokenizer, vocab)
      print('\n lets look at predictions and scores for training \n')
      logging.info('actual model1'+ str(actual_model1))
      logging.info('predicted model1'+ str(pred_model1))
      logging.info('\n')
  
      logging.info('\n')
      logging.info('model1_score'+ str(model1_score))
   
  return batch_loss_model1, batch_loss_model2, model1_score, model2_score

      
def infer(valid_dataloader, model2, instances_gone):

  
  model2_score = 0
  
  softmax = torch.nn.Softmax(-1)

  for step, batch_val in enumerate(valid_dataloader):
      
    model2.eval()
    
    # Input and its attentions
    val_inputs = Variable(batch_val[0], requires_grad=False).to(device)
    
    # Number of datapoints
    n = val_inputs.size(0)
    valid_batch_loss = 0
    epoch_val_loss = 0
    #val batch inputs
    for i in range(n):
      input_train = val_inputs[i][0]
     
      target_train = val_inputs[i][1]
      
      enc_hidden, enc_outputs = model2.enc_forward(input_train)
      valid_loss = model2.dec_forward(target_train, enc_hidden,enc_outputs) 
      #print('valid loss:', valid_loss)
      epoch_val_loss += valid_loss
      instances_gone+=batch_size
      #logging.info('validation batch loss:' + str(valid_batch_loss ))
      
      ######################################################################################

      # the training loss
      if instances_gone % report_freq == 0:
        print('*'*20 + 'batch validation stats'+'*'*20)
        print('validation epoch loss:' + str(valid_loss))
        logging.info('*'*20 + 'validation stats after'+ str(instances_gone) + 'instances' +'*'*20)
        logging.info('validation epoch loss:' + str(valid_loss))
        print("-"*70)
        model2_score, pred_model2, actual_model2 = get_bleu_score(model2,val_inputs[0], tokenizer, vocab)
        print('\n lets look at predictions and scores for validation \n')
        logging.info('actual model'+ str(actual_model2))
        logging.info('predicted model'+ str(pred_model2))
        logging.info('\n')
        logging.info('model_score'+ str(model2_score))
        
    # break
  return epoch_val_loss, model2_score
  

    
     
      

#  early_stopping = EarlyStopping(path = args.save)

start_epoch = 0

logging.info("Starting the Epochs:")

for epoch in range(start_epoch, args.epochs):
    instances_gone_train = 0
    instances_gone_val = 0

    model1_lr = scheduler_model1.get_lr()[0]



    logging.info(str(('epoch %d lr model1 %e lr model2 %e', epoch, model1_lr, model2_lr)))

  
    epoch_loss_model1, epoch_loss_model2, model1_score, model2_score = train(epoch, train_dataloader, valid_dataloader, 
         A, model1,  model1_optim, model1_lr,instances_gone_train)
    
    
    print('+'*20+'TRAIN EPOCH STATS'+'+'*20)
    print(str(epoch_loss_model1), str(epoch_loss_model2))
    logging.info('+'*20+'TRAIN EPOCH STATS'+'+'*20)
    logging.info(str(epoch_loss_model1)+'  '+str(epoch_loss_model2))

    
    logging.info('\n')


    epoch_val_loss, model2_score_val = infer(valid_dataloader, model1, instances_gone_val)

    print('+'*20+'VAL EPOCH STATS'+'+'*20)
    print(str(epoch_val_loss))
    logging.info('+'*20+'VAL EPOCH STATS'+'+'*20)
    logging.info(str(epoch_val_loss))
    
    writer.add_scalar('TrainLoss/model1', epoch_loss_model1, epoch)
    writer.add_scalar('TrainLoss/model2', epoch_loss_model2, epoch)
    writer.add_scalar('ValLoss/model2', epoch_val_loss, epoch)
    writer.add_scalar('TrainBleu/model1', model1_score, epoch)
    writer.add_scalar('TrainBleu/model2', model2_score, epoch)
    writer.add_scalar('ValBleu/model2', model2_score_val, epoch)
    
    scheduler_model1.step()
    
    
    
  
    # logging info
    # logging.info the attention weights and inspect it
    if epoch % 5 == 0:
        print('SAVING MODELS')
        torch.save(model1, args.save+'/model1.pt')
        logging.info(str(("Attention Weights A : ", A.alpha)))
        print('saving in:', str(args.save))

    


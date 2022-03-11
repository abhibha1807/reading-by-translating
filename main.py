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
from utils_ import *
import time
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/rbt-exp')
import glob
# TASK: French (source) -> English (target)
# args 
parser = argparse.ArgumentParser("main")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('using device', device)

parser.add_argument('--begin_epoch', type=float, default=0, help='PC Method begin')
parser.add_argument('--stop_epoch', type=float, default=20, help='Stop training on the framework')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')

parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--seed', type=int, default=seed_, help='random seed')

parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--A_lr', type=float, default=3e-4, help='learning rate for A')
parser.add_argument('--A_wd', type=float, default=1e-3, help=' weight decay for A')


parser.add_argument('--max_length', type=int, default=10, help='max length of sentences')
parser.add_argument('--vocabsize', type=int, default=5000, help='total vocab size')
parser.add_argument('--save_location', type=str, default='./reading-by-translating/', help='save location')
parser.add_argument('--min_freq', type=int, default=2, help='min freq of words to be included in vocab')
parser.add_argument('--train_portion', type=float, default=0.9, help='fraction of dataset for training')
parser.add_argument('--un_portion', type=float, default=0.5, help='fraction of training dataset for creating unlabled dataset')
parser.add_argument('--batch_size', type=int, default=10, help='batch size')
parser.add_argument('--hidden_size', type=int, default=256, help='hidden size')
parser.add_argument('--model1_lr', type=float, default=1e-3, help='model1 starting lr')
parser.add_argument('--model1_lr_min', type=float, default=5e-4, help='model1 min lr')
parser.add_argument('--model2_lr', type=float, default=1e-3, help='model2 starting lr')
parser.add_argument('--model2_lr_min', type=float, default=5e-4, help='model2 min lr')

parser.add_argument('--model1_wd', type=float, default=0, help='model1 weight decay')
parser.add_argument('--model2_wd', type=float, default=0, help='model2 weight decay')
parser.add_argument('--model1_mom', type=float, default=0.9, help='model1 momentum')
parser.add_argument('--model2_mom', type=float, default=0.9, help='model2 momentum')

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


args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


# Setting the seeds
np.random.seed(args.seed)
torch.cuda.set_device(args.gpu)
cudnn.benchmark = True
torch.manual_seed(args.seed)
cudnn.enabled=True
torch.cuda.manual_seed(args.seed)
logging.info('gpu device = %d' % args.gpu)
logging.info("args = %s", args)

#load data
input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print(random.choice(pairs))

#train tokenizer
tokenizer = get_tokenizer(pairs, max_length, min_freq, vocabsize, save_location)


#define models

vocab = tokenizer.get_vocab_size()
print('vocab:', vocab)
criterion = nn.NLLLoss()
criterion = criterion.cuda()
model1 = Model1(vocab, vocab, criterion)
model2 = Model2( vocab,  vocab, criterion)
model1 = model1.cuda()
model2 = model2.cuda()
model1_optim = SGD(model1.parameters(), lr=model1_lr, momentum=model1_mom,weight_decay=model1_wd)
model2_optim = SGD(model2.parameters(), lr=model2_lr, momentum=model1_mom,weight_decay=model1_wd)


scheduler_model1  = torch.optim.lr_scheduler.CosineAnnealingLR(model1_optim, float(args.epochs), eta_min=args.model1_lr_min)

scheduler_model2  = torch.optim.lr_scheduler.CosineAnnealingLR(model2_optim, float(args.epochs), eta_min=args.model1_lr_min)

#split 80% train 10% val 10% test
n = len(pairs)
print(n)
train_index = int(train_portion * n)
print(train_index)
valid_index = int(0.5 * train_index)
print(valid_index)
train_portion = pairs[0:valid_index]
un_portion = pairs[valid_index : train_index]
valid_portion = pairs[train_index:]
print(len(train_portion), len(un_portion), len(valid_portion))


# Print the number of training valid and test examples
logging.info('dataset')


train_data = get_train_dataset(train_portion, tokenizer)
un_data = get_un_dataset(un_portion, tokenizer)
valid_data = get_valid_dataset(valid_portion, tokenizer)

logging.info(f"{len(train_data):^7} | { len(un_data):^7} | { len(valid_data):^7}")


train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), 
                        batch_size=batch_size, pin_memory=True, num_workers=0)

valid_dataloader = DataLoader(valid_data, sampler=RandomSampler(valid_data), 
                      batch_size=batch_size, pin_memory=True, num_workers=0)

un_dataloader = DataLoader(un_data, sampler=RandomSampler(un_data), 
                        batch_size=batch_size, pin_memory=True, num_workers=0)


#define A
A = attention_params(len(train_data))

print('A:', list(A.parameters()))
A = A.cuda()
architect = Architect(model1, model1_mom, model1_wd, A, A_lr, A_wd, device, model2, model2_wd, model2_mom, batch_size,vocab)


def train(epoch, train_dataloader, un_dataloader, valid_dataloader, architect, A, model1, model2, model1_optim, model2_optim, model1_lr, model2_lr, instances_gone):
  epoch_loss_model1 = 0
  epoch_loss_model2 = 0
  actual_model1 = ''
  actual_model2 = ''
  pred_model1 = ''
  pred_model2 = ''
  model1_score = 0
  model2_score = 0
  for step, batch in enumerate(train_dataloader):
    model1.train()
    model2.train()
    #summary_bart = Variable(batch[2], requires_grad=False).cuda()
    train_inputs = Variable(batch[0], requires_grad=False).cuda() #train inputs.
    idxs = Variable(batch[1],requires_grad=False).cuda() #A
    un_batch = next(iter(un_dataloader)) 
    un_inputs = Variable(un_batch[0], requires_grad=False).cuda()
    val_batch = next(iter(valid_dataloader)) 
    val_inputs = Variable(val_batch[0], requires_grad=False).cuda()


    if args.begin_epoch <= epoch <= args.stop_epoch:
      logging.info('in architect')
      architect.step(train_inputs, un_inputs, val_inputs, model1_lr, A, idxs, criterion, model2_lr, model2_optim, model1_optim)
    
    if epoch <= args.stop_epoch:
      logging.info('otherwise')
      model1_optim.zero_grad()
      loss_model1 = loss1(train_inputs, model1, idxs, A, batch_size, vocab)
      #print('training loss model1:', loss_model1)
      
      # store the batch loss
      epoch_loss_model1 += loss_model1.item()

      loss_model1.backward()
      
      nn.utils.clip_grad_norm(model1.parameters(), args.grad_clip)
      
      model1_optim.step()
      
      # Update model2 model
      model2_optim.zero_grad()
      #un inputs
      loss_model2 = loss2(train_inputs, model1, model2, batch_size, vocab)
      #print('training loss model2:', loss_model2)
      
      # # store the batch loss
      epoch_loss_model2 += loss_model2.item()
      

      loss_model2.backward()
      
      nn.utils.clip_grad_norm(model2.parameters(), args.grad_clip)
      
      model2_optim.step()
      
      #assess predictions
      model1_score, pred_model1, actual_model1 = get_bleu_score(model1,val_inputs[0], tokenizer, vocab)
      model2_score, pred_model2, actual_model2 = get_bleu_score(model2,val_inputs[0], tokenizer, vocab)
    instances_gone+= batch_size
    if instances_gone % 50 ==0:
    
      print('\n lets look at predictions and scores \n')
      logging.info('actual model1'+ str(actual_model1))
      logging.info('predicted model1'+ str(pred_model1))
      logging.info('\n')
      logging.info('actual model2'+ str(actual_model2))
      logging.info('predicted model2'+ str(pred_model2))
      logging.info('\n')
      logging.info('model1_score'+ str(model1_score))
      logging.info('model2_score'+ str(model2_score))

        # writer.add_scalar('Loss/model1', loss_model1, epoch)
        # writer.add_scalar('Loss/model2', loss_model2, epoch)
      print('-'*40+'training batch stats after'+str(instances_gone)+'instances'+'-'*40)
      print('Epoch:'+str(epoch)+'batch_loss_model1:'+str(loss_model1.item())+'batch_loss_model2:'+str(loss_model2.item()))
          

      # if step % args.report_freq == 0:
    
      # logging.info(f"{'Epoch':^7} | {'Train Loss model 1':^12}  | {'Train Loss model 2':^12}")
      
      # logging.info("-"*70)
      
      # logging.info(f"{epoch + 1:^7} | {loss_model1:^7} | {loss_model2:^12.6f}")


  return epoch_loss_model1, epoch_loss_model2

      
def infer(valid_dataloader, model2, instances_gone):

  # objs = utils.AvgrageMeter()
  # top1 = utils.AvgrageMeter()
  # top5 = utils.AvgrageMeter()
  
  softmax = torch.nn.Softmax(-1)

  for step, batch_val in enumerate(valid_dataloader):
      
    #model2.eval()
    
    # Input and its attentions
    val_inputs = Variable(batch_val[0], requires_grad=False).cuda()
    
    # Number of datapoints
    n = val_inputs.size(0)
    valid_batch_loss = 0
    epoch_val_loss = 0
    #val batch inputs
    for i in range(args.batch_size):
      input_train = val_inputs[i][0]
      onehot_input = torch.zeros(input_train.size(0),vocab, device = 'cuda')
      index_tensor = input_train
      onehot_input.scatter_(1, index_tensor, 1.)
      input_train = onehot_input
      #print('input valid size:', input_train.size())
      target_train = val_inputs[i][1]
      
      enc_hidden, enc_outputs = model2.enc_forward(input_train)
      valid_loss = model2.dec_forward(target_train, enc_hidden) 
      #print('valid loss:', valid_loss)
      epoch_val_loss += valid_loss
      instances_gone+=batch_size
      #logging.info('validation batch loss:' + str(valid_batch_loss ))
      
      ######################################################################################

      # the training loss
      if instances_gone % 20 == 0:
        print('*'*20 + 'batch validation stats'+'*'*20)
        print('validation epoch loss:' + str(valid_loss))
        logging.info('*'*20 + 'validation stats after'+ str(instances_gone) + 'instances' +'*'*20)
        logging.info('validation epoch loss:' + str(valid_loss))
  return epoch_val_loss
  

    
     
      #break

#  early_stopping = EarlyStopping(path = args.save)

start_epoch = 0

logging.info("Starting the Epochs:")

for epoch in range(start_epoch, args.epochs):
    instances_gone_train = 0
    instances_gone_val = 0

    model1_lr = scheduler_model1.get_lr()[0]

    model2_lr = scheduler_model2.get_lr()[0]


    logging.info(str(('epoch %d lr model1 %e lr model2 %e', epoch, model1_lr, model1_lr)))

    # training
    epoch_loss_model1, epoch_loss_model2 = train(epoch, train_dataloader, un_dataloader, valid_dataloader, 
        architect, A, model1, model2,  model1_optim, model2_optim, model1_lr, model2_lr,instances_gone_train)
    
    print('+'*20+'TRAIN EPOCH STATS'+'+'*20)
    print(str(epoch_loss_model1), str(epoch_loss_model2))
    logging.info('+'*20+'TRAIN EPOCH STATS'+'+'*20)
    logging.info(str(epoch_loss_model1)+'  '+str(epoch_loss_model2))
    
    logging.info('\n')

    epoch_val_loss = infer(valid_dataloader, model2, instances_gone_val)
    print('+'*20+'VAL EPOCH STATS'+'+'*20)
    print(str(epoch_val_loss))
    logging.info('+'*20+'VAL EPOCH STATS'+'+'*20)
    logging.info(str(epoch_val_loss))
    
    writer.add_scalar('TrainLoss/model1', epoch_loss_model1, epoch)
    writer.add_scalar('TrainLoss/model2', epoch_loss_model2, epoch)
    writer.add_scalar('ValLoss/model2', epoch_val_loss, epoch)
    
    scheduler_model1.step()
    
    scheduler_model2.step()
    
    
    # logging.info(str(('train_loss_model1 %e train_loss_model2 %e', epoch_loss_model1, epoch_loss_model2)))
    # logging.info(str(('val_loss_model2 %e', epoch_loss_model2)))

    ################################################################################################

    # Note to the user: add all the information needed to analyse on the logfile
    
    # logging.info(f"{'Epoch':^7} | {'Train Loss model1':^12} | {'Train Loss model2':^12}")
    
    # logging.info("-"*70)
    
    # logging.info(f"{epoch + 1:^7} | {loss_model1:^7} | {loss_model1:^12.6f} ")
        
    # logging info
    # logging.info the attention weights and inspect it
    if epoch % 5 == 0:
        logging.info(str(("Attention Weights A : ", A.alpha)))
    # break
    

   
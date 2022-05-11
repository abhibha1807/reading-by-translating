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
CUDA_LAUNCH_BLOCKING=1.

import glob
# TASK: French (source) -> English (target)
# args 
parser = argparse.ArgumentParser("main")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('using device', device)

print('eecuting Attn Decoder')
parser.add_argument('--begin_epoch', type=float, default=0, help='PC Method begin')
parser.add_argument('--stop_epoch', type=float, default=20, help='Stop training on the framework')
parser.add_argument('--report_freq', type=float, default=10, help='report frequency')

parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')

parser.add_argument('--batch_size', type=int, default=64, help='batch size')
####################################################################################
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--A_lr', type=float, default=3e-4, help='learning rate for A')
#reduce lr 
# parser.add_argument('--A_lr', type=float, default=1e-6, help='learning rate for A')

# parser.add_argument('--A_wd', type=float, default=1e-6, help=' weight decay for A')
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
parser.add_argument('--model2_lr', type=float, default=1e-3, help='model2 starting lr')
parser.add_argument('--model2_lr_min', type=float, default=5e-4, help='model2 min lr')



#reduce lr
# parser.add_argument('--model1_lr', type=float, default=1e-4, help='model1 starting lr')
# parser.add_argument('--model1_lr_min', type=float, default=5e-6, help='model1 min lr')
# parser.add_argument('--model2_lr', type=float, default=1e-7, help='model2 starting lr')
# parser.add_argument('--model2_lr_min', type=float, default=5e-6, help='model2 min lr')

parser.add_argument('--model1_wd', type=float, default=0, help='model1 weight decay')
parser.add_argument('--model2_wd', type=float, default=0, help='model2 weight decay')
# parser.add_argument('--model1_mom', type=float, default=0.9, help='model1 momentum')
# parser.add_argument('--model2_mom', type=float, default=0.9, help='model2 momentum')
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
report_freq = args.batch_size

args.save = '{}-{}-e20-50'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
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
input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print(random.choice(pairs))

#train tokenizer
tokenizer = get_tokenizer(pairs, max_length, min_freq, vocabsize, save_location)


#define models

vocab = tokenizer.get_vocab_size()
print('vocab:', vocab)
criterion = nn.NLLLoss(ignore_index = tokenizer.token_to_id("[PAD]"), reduction='none')
#criterion = nn.CrossEntropyLoss(ignore_index = tokenizer.token_to_id("[PAD]"),  reduction='none')
criterion = criterion.to(device)
model1 = Model1(vocab, vocab, criterion)
model2 = Model2(vocab, vocab, criterion)
model1 = model1.to(device)
model2 = model2.to(device)
#momentum=model1_mom,weight_decay=model1_wd,  momentum=model1_mom,weight_decay=model1_wd
# model1_optim = SGD(model1.parameters(), lr=model1_lr) #reduced to a value and stayed conbstant
# model2_optim = SGD(model2.parameters(), lr=model2_lr)

model1_optim = torch.optim.Adam(model1.parameters(),lr=model1_lr,weight_decay=model1_wd) #  loss decreased and then inc 
model2_optim = torch.optim.Adam(model2.parameters(),lr=model2_lr, weight_decay=model2_wd)



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


train_data = get_train_dataset(train_portion[0:2000], tokenizer)
un_data = get_un_dataset(un_portion[0:2000], tokenizer)
valid_data = get_valid_dataset(valid_portion[0:50], tokenizer)

logging.info(f"{len(train_data):^7} | { len(un_data):^7} | { len(valid_data):^7}")


train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), 
                        batch_size=batch_size, pin_memory=True, num_workers=0)

valid_dataloader = DataLoader(valid_data, sampler=RandomSampler(valid_data), 
                      batch_size=batch_size, pin_memory=True, num_workers=0)

un_dataloader = DataLoader(un_data, sampler=RandomSampler(un_data), 
                        batch_size=batch_size, pin_memory=True, num_workers=0)


#define A
A = attention_params(len(train_data))
A.requires_grad_()
print('A:', list(A.parameters()))
A = A.to(device)
architect = Architect(model1, model1_mom, model1_wd, A, A_lr, A_wd, device, model2, model2_wd, model2_mom, batch_size,vocab)


def train(epoch, train_dataloader, un_dataloader, valid_dataloader, architect, A, model1, model2, model1_optim, model2_optim, model1_lr, model2_lr, instances_gone):

  # objs = utils.AvgrageMeter()
  # top1 = utils.AvgrageMeter()
  # top5 = utils.AvgrageMeter()

  batch_loss_model1, batch_loss_model2, batch_count = 0, 0, 0
  valid_batch_loss = 0
  model1_score, model2_score = 0, 0
  

    
  for step, batch in enumerate(train_dataloader):
    model1.train()
    model2.train()
    #summary_bart = Variable(batch[2], requires_grad=False).cuda()
    train_inputs = Variable(batch[0], requires_grad=False).to(device) #train inputs.
    idxs = Variable(batch[1],requires_grad=False).to(device) #A
    un_batch = next(iter(un_dataloader)) 
    un_inputs = Variable(un_batch[0], requires_grad=False).to(device)
    val_batch = next(iter(valid_dataloader)) 
    val_inputs = Variable(val_batch[0], requires_grad=False).to(device)

    #print('\n', train_inputs, '\n')
   


    if args.begin_epoch <= epoch <= args.stop_epoch:
      #logging.info('in architect')
      valid_batch_loss = architect.step(train_inputs, un_inputs, val_inputs, model1_lr, A, idxs, criterion, model2_lr, model2_optim, model1_optim)
  
    if epoch <= args.stop_epoch:
      
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
      #print('model1 enc gru grad:', model1.enc.gru.weight.grad)
      #print('model1 dec attn grad:', model1.dec.attn.weight.grad)
      #print('model1 dec attn combine grad:', model1.dec.attn_combine.weight.grad)
      #print('model1 dec gru grad:', model1.dec.gru.weight.grad)
        
      
      nn.utils.clip_grad_norm_(model1.parameters(), args.grad_clip)
      
      model1_optim.step()
      b = list(model1.parameters())[0].clone()
      print('\n')
      print('are wts being updated??')
      print(torch.equal(a.data, b.data))
    
    model2_optim.zero_grad()
    
    loss_model2 = loss2(un_inputs, model1, model2, batch_size, vocab)
    #loss_model2 = loss3(un_inputs, model2, batch_size, vocab)
    print(str(epoch)+'is loss being calculated or not?:', loss_model2)
    batch_loss_model2 += loss_model2.item()
    #print(str(epoch)+'calculated batch loss model 2:', batch_loss_model2)
    c = list(model2.parameters())[0].clone()
    loss_model2.backward()
    for param in model2.parameters():
      print(param.grad.data.sum())

    # start debugger
    #import pdb; pdb.set_trace()

    print('model2 enc embeding grad:', model2.enc.embedding.weight.grad.data.sum())
    print(list(model2.parameters())[0].grad)
    #print('model2 enc gru grad:', model2.enc.gru.weight.grad)
    
    #print('model2 dec attn grad:', model2.dec.attn.weight.grad)
    #print('model2 dec attn combine grad:', model2.dec.attn_combine.weight.grad)
    #print('model2 dec gru grad:', model2.dec.gru.weight.grad)
    nn.utils.clip_grad_norm_(model2.parameters(), args.grad_clip)
    model2_optim.step()
    d = list(model2.parameters())[0].clone()
    print('\n')
    print('are wts2 being updated??')
    print(torch.equal(c.data, d.data))
    


    # objs.update(loss_model2.item(), n)
    instances_gone+= batch_size
    '''
    :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
    :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
    :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
    :return: decoded_batch
    '''
    # if instances_gone % report_freq == 0:
    #   n = val_inputs.size(0)
      #val batch inputs
     
      # input_beam = val_inputs[0][0]
      # target_beam = val_inputs[0][1]
      # enc_hidden, enc_outputs = model2.enc_forward(input_beam)
      # target_beam = torch.unsqueeze(target_beam, dim=0)
      # enc_outputs = torch.unsqueeze(enc_outputs, dim=1)
      # # print('target:', target_beam.size())
      # # print('hidden:', enc_hidden.size())
      # # print('encoder_outputs:', enc_outputs.size())
      # decoded_batch = beam_decode(target_beam, enc_hidden, model2.dec, enc_outputs)
      # print(decoded_batch)
      # predicted = (tokenizer.decode((decoded_batch[0][0])))
      # print('tok decode pred:', predicted)
      # #predicted = pad_sentences(predicted)
      # length_pred = len(predicted.split(' '))
      # # actual = tokenizer.decode(list((test_inputs[1])))
      # actual = tokenizer.decode(list((torch.squeeze(target_beam, dim=0))))
      # print('tok decode actual :', actual)
      # length_actual = len(actual.split(' '))
      # print('len pred:', length_pred)
      # print('len actual:', length_actual)
      # if length_pred>length_actual:
      #   actual = pad_sentences(actual, length_pred)
      # else:
      #   predicted = pad_sentences(predicted, length_actual)
      # predicted = predicted.split(' ')
      # actual = actual.split(' ')
      # print('\n')
      # print('predicted:', predicted)
      # print('actual:', actual)
      # print(bleu_score(predicted, actual))

    
    
    # if step % args.report_freq == 0:

      # logging.info(f"{'Epoch':^7} | {'Train Loss':^12}")
      
      # logging.info("-"*70)
      
      # logging.info(f"{epoch + 1:^7} | {top1.avg:^7} ")

    if instances_gone % report_freq == 0:
 
      print('-'*40+'training batch stats after'+str(instances_gone)+'instances'+'-'*40)
      print('Epoch:'+str(epoch)+'batch_loss_model2:'+str(loss_model2))
    
      print("-"*70)
      model1_score, pred_model1, actual_model1 = get_bleu_score(model1,val_inputs[0], tokenizer, vocab)
      model2_score, pred_model2, actual_model2 = get_bleu_score(model2,val_inputs[0], tokenizer, vocab)
      print('\n lets look at predictions and scores for training \n')
      logging.info('actual model1'+ str(actual_model1))
      logging.info('predicted model1'+ str(pred_model1))
      logging.info('\n')
      logging.info('actual model2'+ str(actual_model2))
      logging.info('predicted model2'+ str(pred_model2))
      logging.info('\n')
      logging.info('model1_score'+ str(model1_score))
      logging.info('model2_score'+ str(model2_score))
     
    # break

  return batch_loss_model1, batch_loss_model2, model1_score, model2_score
  #return valid_batch_loss

      
def infer(valid_dataloader, model2, instances_gone):

  # objs = utils.AvgrageMeter()
  # top1 = utils.AvgrageMeter()
  # top5 = utils.AvgrageMeter()
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
        logging.info('actual model2'+ str(actual_model2))
        logging.info('predicted model2'+ str(pred_model2))
        logging.info('\n')
        logging.info('model2_score'+ str(model2_score))
        
    # break
  return epoch_val_loss, model2_score
  

    
     
      

#  early_stopping = EarlyStopping(path = args.save)

start_epoch = 0

logging.info("Starting the Epochs:")

for epoch in range(start_epoch, args.epochs):
    instances_gone_train = 0
    instances_gone_val = 0

    model1_lr = scheduler_model1.get_lr()[0]

    model2_lr = scheduler_model2.get_lr()[0]


    logging.info(str(('epoch %d lr model1 %e lr model2 %e', epoch, model1_lr, model2_lr)))

    #training
    epoch_loss_model1, epoch_loss_model2, model1_score, model2_score = train(epoch, train_dataloader, un_dataloader, valid_dataloader, 
        architect, A, model1, model2,  model1_optim, model2_optim, model1_lr, model2_lr,instances_gone_train)
    
    # valid_batch_loss = train(epoch, train_dataloader, un_dataloader, valid_dataloader, 
    #     architect, A, model1, model2,  model1_optim, model2_optim, model1_lr, model2_lr,instances_gone_train)
    
    print('+'*20+'TRAIN EPOCH STATS'+'+'*20)
    print(str(epoch_loss_model1), str(epoch_loss_model2))
    logging.info('+'*20+'TRAIN EPOCH STATS'+'+'*20)
    logging.info(str(epoch_loss_model1)+'  '+str(epoch_loss_model2))

    # logging.info('+'*20+'TRAIN EPOCH STATS'+'+'*20)
    # logging.info(str(valid_batch_loss))
    
    
    logging.info('\n')


    epoch_val_loss, model2_score_val = infer(valid_dataloader, model2, instances_gone_val)
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
    
    scheduler_model2.step()
    
    
    # logging.info(str(('train_loss_model1 %e train_loss_model2 %e', epoch_loss_model1, epoch_loss_model2)))
    # logging.info(str(('val_loss_model2 %e', epoch_loss_model2)))

    ################################################################################################


   
    # logging info
    # logging.info the attention weights and inspect it
    if epoch % 5 == 0:
        print('SAVING MODELS')
        torch.save(model1, args.save+'/model1.pt')
        torch.save(model2, args.save+'/model2.pt')
        logging.info(str(("Attention Weights A : ", A.alpha)))
        print('saving in:', str(args.save))
    # break
    
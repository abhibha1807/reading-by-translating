import torch
from dataset import *
import gc
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loss1(inputs, model, idxs, A, batch_size, vocab):
    A_idx = A(idxs)
    #print('A_idx:', A_idx)
    batch_loss = 0
    for i in range(batch_size):
      input_train = inputs[i][0]
      onehot_input = torch.zeros(input_train.size(0), vocab, device = device)
      index_tensor = input_train
      onehot_input.scatter_(1, index_tensor, 1.)
      input_train = onehot_input
      #print(input_train.size())
      target_train = inputs[i][1]
      idx = A_idx[i]
      enc_hidden, enc_outputs = model.enc_forward(input_train)
      loss = model.dec_forward(target_train, enc_hidden) # todo: find loss for each instnce and multiply A with the loss vec.
      #print('loss and idx size:', loss.size(), idx.size())
      loss = loss * idx
      batch_loss += loss 
    
    return batch_loss

def loss2(un_inputs, model1, model2, batch_size, vocab):
    batch_loss = 0
    
    #generate pseudo target by passing through decoder
    for i in range(batch_size):  
      input_un = un_inputs[i][0]
      decoder_input = torch.tensor([[SOS_token]], device=device)#where to put SOS_token
      decoder_hidden = model1.dec.initHidden()
      #print('forward pass through decoder')
      
      dec_soft_idxs = []
      decoder_outputs = []
      for di in range(MAX_LENGTH):
          decoder_output, decoder_hidden = model1.dec(
              decoder_input, decoder_hidden)
          topv, topi = decoder_output.topk(1)
          decoder_input = topi.squeeze().detach()  # detach from history as input
          #print('decoder output:', decoder_output.size())
          dec_soft_idx, dec_idx = torch.max(decoder_output, dim = -1, keepdims = True)
          dec_soft_idxs.append(dec_soft_idx) #save differentiable outputs
          # print('dec soft idx size:', dec_soft_idx)
          #print(dec_soft_idxs)
          decoder_outputs.append(torch.unsqueeze(torch.argmax(decoder_output), dim = -1))
          if decoder_input.item() == EOS_token:
              break

      #print(decoder_outputs) #pseudo target
      decoder_outputs = torch.stack(decoder_outputs)#differentiable,no break in computation graph

      #print(decoder_outputs.size())

      # gumbel softmax (prepare target for generating pseudo input using encoder)
      onehot_input_encoder1 = torch.zeros(decoder_outputs.size(0), vocab, device = device)
      #print(onehot_input.size())
      index_tensor = decoder_outputs
      #print(index_tensor.size())
      dec_soft_idxs = (torch.stack(dec_soft_idxs))
      onehot_input_encoder1 = onehot_input_encoder1.scatter_(1, index_tensor, 1.).float().detach() + (dec_soft_idxs).sum() - (dec_soft_idxs).sum().detach()
      #print(onehot_input.size(), onehot_input[0])

      enc_hidden, enc_outputs = model1.enc_forward(onehot_input_encoder1)
      
      pseudo_target = decoder_outputs
      # pseudo_input = enc_outputs

      # print('pseudo target:', pseudo_target, pseudo_target.size())
      # greedy decoding -> similar to model.generate() (hugging face)
      decoder_input = torch.tensor([[SOS_token]], device=device)#where to put SOS_token
      decoder_hidden = enc_hidden

      dec_soft_idxs = []
      decoder_outputs = []
      for di in range(MAX_LENGTH):
          decoder_output, decoder_hidden = model1.dec(
              decoder_input, decoder_hidden)
          topv, topi = decoder_output.topk(1)
          decoder_input = topi.squeeze().detach()  # detach from history as input
          dec_soft_idx, dec_idx = torch.max(decoder_output, dim = -1, keepdims = True)
          dec_soft_idxs.append(dec_soft_idx)
          decoder_outputs.append(torch.unsqueeze(torch.argmax(decoder_output), dim = -1))

          if decoder_input.item() == EOS_token:
              break
      
      # gumbel softmax 
      input_to_model2 = torch.stack(decoder_outputs)
      # print('input_to_model2 :', input_to_model2, input_to_model2.size())
      onehot_input_model2 = torch.zeros(input_to_model2.size(0), vocab, device = device)
      #print(onehot_input.size())
      index_tensor = input_to_model2
      #print(index_tensor.size())
      dec_soft_idxs = (torch.stack(dec_soft_idxs))
      onehot_input_model2 = onehot_input_model2.scatter_(1, index_tensor, 1.).float().detach() + (dec_soft_idxs).sum() - (dec_soft_idxs).sum().detach()
      #print(onehot_input.size(), onehot_input[0])

      pseudo_input = onehot_input_model2 
      # print('pseudo input:', pseudo_input, pseudo_input.size())

      #model2 forward pass
      enc_hidden, enc_outputs = model2.enc_forward(pseudo_input)
      loss = model2.dec_forward(pseudo_target, enc_hidden) # todo: find loss for each instnce and multiply A with the loss vec.
      # print('loss size:', loss)
      batch_loss += loss 

    del onehot_input_encoder1 , onehot_input_model2
        
    gc.collect()   
    
    return batch_loss

  


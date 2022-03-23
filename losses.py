import torch
from dataset import *
import gc
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loss1(inputs, model, idxs, A, batch_size, vocab):
    A_idx = A(idxs)
    print('A_idx:', A_idx)
    batch_loss = 0
    print('batch size:', inputs.size(0))
    for i in range(inputs.size(0)):
        # try:
        input_train = inputs[i][0]
        #print('dtype input:', input_train.dtype)
        # onehot_input = torch.zeros(input_train.size(0), vocab, device = device)
        # index_tensor = input_train
        # onehot_input.scatter_(1, index_tensor, 1.)
        # input_train = onehot_input
        #print('dtype input new:', input_train.dtype) ##torch.float32
        #print(input_train.size())
        # print('onehot input:', input_train)
        target_train = inputs[i][1]
        idx = A_idx[i]
        enc_hidden, enc_outputs = model.enc_forward(input_train)
        loss = model.dec_forward(target_train, enc_hidden, enc_outputs) # todo: find loss for each instnce and multiply A with the loss vec.
        #print('loss and idx :', loss, idx)
        loss = loss * idx
        batch_loss += loss 
   # print('batch loss loss1:',batch_loss)
    return batch_loss/inputs.size(0)
        # except:
        #     print('skipping this_______________________________________')
        
        
    

def loss2(un_inputs, model1, model2, batch_size, vocab):
    print('in loss 2')
    batch_loss = 0
    
    #generate pseudo target by passing through decoder
    for i in range(batch_size):  
        # try:
            input_un = un_inputs[i][0]
            enc_hidden_model1, enc_outputs_model1 = model1.enc_forward(input_un)
           # print('enc hidden:', enc_hidden)
           # print('enc_outputs:', enc_outputs)

            decoder_input_model1 = torch.tensor([[SOS_token]], device=device)#where to put SOS_token
            decoder_hidden_model1 = enc_hidden_model1
           # print('forward pass through decoder')
            
            soft_idxs_model1 = []
            decoder_outputs_model1 = []
            for di in range(MAX_LENGTH):
                decoder_output_model1, decoder_hidden_model1, decoder_attention_model1 = model1.dec(
                    decoder_input_model1, decoder_hidden_model1, enc_outputs_model1)
                topv, topi = decoder_output_model1.topk(1)
                decoder_input_model1 = topi.squeeze().detach()  # detach from history as input
                #print('decoder output:', decoder_output_model1.size())
                soft_idxs, dec_idx = torch.max(decoder_output_model1, dim = -1, keepdims = True)
                soft_idxs_model1.append(soft_idxs) #save differentiable outputs
                # print('dec soft idx size:', soft_idxs_model1)
                #print(soft_idxs_model1)
                decoder_outputs_model1.append(torch.unsqueeze(torch.argmax(decoder_output_model1), dim = -1))
                if decoder_input_model1.item() == EOS_token:
                    break

            #print(decoder_outputs_model1) #pseudo target
            #print('before dec_soft_idxs:', soft_idxs_model1)# every tensor has grad fun assopciated with it.
            decoder_outputs_model1 = torch.stack(decoder_outputs_model1)#differentiable,no break in computation graph

           # print('decoder_outputs:',decoder_outputs)

            # gumbel softmax (prepare target for generating pseudo input using encoder)
            onehot_input_encoder1 = torch.zeros(decoder_outputs_model1.size(0), vocab, device = device)
            #print(onehot_input_encoder1.size())
            index_tensor = decoder_outputs_model1
            #print(index_tensor.size())
            soft_idxs_model1 = (torch.stack(soft_idxs_model1))
            #print('dec_soft_idxs:', soft_idxs_model1)
            onehot_input_encoder1 = onehot_input_encoder1.scatter_(1, index_tensor, 1.).float().detach() + (soft_idxs_model1).sum() - (soft_idxs_model1).sum().detach()
            #print('onehot_input_encoder1:', onehot_input_encoder1)

            enc_hidden, enc_outputs = model1.enc_forward(onehot_input_encoder1)
            
            pseudo_target = decoder_outputs_model1

            # print('pseudo target:', pseudo_target, pseudo_target.size())
            # greedy decoding -> similar to model.generate() (hugging face)
            decoder_input = torch.tensor([[SOS_token]], device=device)#where to put SOS_token
            decoder_hidden = enc_hidden
            #print('decoder_hidden:', decoder_hidden)

            dec_soft_idxs = []
            decoder_outputs = []
            for di in range(MAX_LENGTH):
                decoder_output, decoder_hidden, decoder_attention = model1.dec(
                    decoder_input, decoder_hidden, enc_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input
                dec_soft_idx, dec_idx = torch.max(decoder_output, dim = -1, keepdims = True)
                dec_soft_idxs.append(dec_soft_idx)
                decoder_outputs.append(torch.unsqueeze(torch.argmax(decoder_output), dim = -1))

                if decoder_input.item() == EOS_token:
                    break
            
            # gumbel softmax 
            input_to_model2 = torch.stack(decoder_outputs)
            #print('input_to_model2 :', input_to_model2)
            onehot_input_model2 = torch.zeros(input_to_model2.size(0), vocab, device = device)
            #print(onehot_input.size())
            index_tensor = input_to_model2
            #print(index_tensor.size())
            dec_soft_idxs = (torch.stack(dec_soft_idxs))
            #print('dec_soft_idxs:', dec_soft_idxs)
            onehot_input_model2 = onehot_input_model2.scatter_(1, index_tensor, 1.).float().detach() + (dec_soft_idxs).sum() - (dec_soft_idxs).sum().detach()
            #print('onehot_input_model2:', onehot_input_model2)

            pseudo_input = onehot_input_model2 
            # print('pseudo input:', pseudo_input, pseudo_input.size())

            #model2 forward pass
            enc_hidden_model2, enc_outputs_model2 = model2.enc_forward(pseudo_input)
            loss = model2.dec_forward(pseudo_target, enc_hidden_model2, enc_outputs_model2) # todo: find loss for each instnce and multiply A with the loss vec.
            #print('loss:', loss)
            batch_loss += loss 

            #del onehot_input_encoder1 , onehot_input_model2
                
            gc.collect()   
            #print('batch_loss:', batch_loss)
    return batch_loss/batch_size
        # except:
        #     print('skipping this_______________________________________')

        


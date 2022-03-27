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
        print('loss and idx :', loss, idx)
        loss = loss * idx
        batch_loss += loss 
    print('batch loss loss1:',batch_loss)
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
            # onehot_input = torch.zeros(input_un.size(0), vocab, device = device)
            # index_tensor = input_un
            # onehot_input.scatter_(1, index_tensor, 1.)
            # input_un = onehot_input
            # print('input:', input_un)
            #enc_hidden, enc_outputs = model1.enc_forward(input_un)
            #print('enc hidden:', enc_hidden, enc_hidden.requires_grad)
            #print('enc_outputs:', enc_outputs, enc_outputs.requires_grad)

            #decoder_input = torch.tensor([[SOS_token]], device=device)#where to put SOS_token
            #decoder_hidden = enc_hidden
           # print('forward pass through decoder')
            decoder_input = input_un
            decoder_hidden = model1.dec.initHidden()
            
            dec_soft_idxs = []
            decoder_outputs = []
            for di in range(decoder_input.size(0)):
                embedded = model1.embedding(decoder_input[di]).view(1, 1, -1)
                #embedded = model1.embedding(decoder_input[di])
                print(embedded.size())
                decoder_output, decoder_hidden = model1.dec(
                    embedded, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input
                
                print('decoder output:', decoder_output,decoder_output.requires_grad)
                dec_soft_idx, dec_idx = torch.max(decoder_output, dim = -1, keepdims = True)
                dec_soft_idxs.append(dec_soft_idx) #save differentiable outputs
                print('dec soft idx size:', dec_soft_idx, dec_soft_idx.requires_grad)
                #print(dec_soft_idxs)
                decoder_outputs.append(torch.unsqueeze(torch.argmax(decoder_output), dim = -1))
                if decoder_input.item() == EOS_token:
                    break

            #print(decoder_outputs) #pseudo target
            print('before dec_soft_idxs:', dec_soft_idxs)# every tensor has grad fun assopciated with it.
            decoder_outputs = torch.stack(decoder_outputs)#differentiable,no break in computation graph

            print('decoder_outputs:',decoder_outputs, decoder_outputs.requires_grad)

            # gumbel softmax (prepare target for generating pseudo input using encoder)
            onehot_input_encoder1 = torch.zeros(decoder_outputs.size(0), vocab, device = device)
            print('onehot_input_encoder1:', onehot_input_encoder1, onehot_input_encoder1.requires_grad)
            index_tensor = decoder_outputs
            #print(index_tensor)
            dec_soft_idxs = (torch.stack(dec_soft_idxs))
            print('dec_soft_idxs:', dec_soft_idxs, dec_soft_idxs.requires_grad)
            onehot_input_encoder1 = onehot_input_encoder1.scatter_(1, index_tensor, 1.).float().detach() + (dec_soft_idxs).sum() - (dec_soft_idxs).sum().detach()
            print('onehot_input_encoder1:', onehot_input_encoder1, onehot_input_encoder1.requires_grad)

            enc_hidden_, enc_outputs_ = model1.enc_forward(onehot_input_encoder1)
            print('enc hidden_:', enc_hidden_, enc_hidden_.requires_grad)
            print('enc_outputs_:', enc_outputs_, enc_outputs_.requires_grad)
            
            pseudo_target = decoder_outputs

            # print('pseudo target:', pseudo_target, pseudo_target.size())
            # greedy decoding -> similar to model.generate() (hugging face)
            decoder_input = torch.tensor([[SOS_token]], device=device)#where to put SOS_token
            decoder_hidden = enc_hidden_
            #print('decoder_hidden:', decoder_hidden)

            dec_soft_idxs = []
            decoder_outputs = []
            for di in range(MAX_LENGTH):
                embedded = model1.embedding(decoder_input).view(1, 1, -1)
                decoder_output, decoder_hidden = model1.dec(
                    embedded, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                print('decoder output:', decoder_output,decoder_output.requires_grad)
                decoder_input = topi.squeeze().detach()  # detach from history as input
                dec_soft_idx, dec_idx = torch.max(decoder_output, dim = -1, keepdims = True)
                print('dec soft idx size:', dec_soft_idx, dec_soft_idx.requires_grad)
                dec_soft_idxs.append(dec_soft_idx)
                decoder_outputs.append(torch.unsqueeze(torch.argmax(decoder_output), dim = -1))

                if decoder_input.item() == EOS_token:
                    break
            
            print('decoder_outputs:',decoder_outputs)
            # gumbel softmax 
            input_to_model2 = torch.stack(decoder_outputs)
            print('input_to_model2 :', input_to_model2)
            onehot_input_model2 = torch.zeros(input_to_model2.size(0), vocab, device = device)
            print(onehot_input_model2.size(),onehot_input_model2.requires_grad)
            index_tensor = input_to_model2
            #print(index_tensor.size())
            dec_soft_idxs = (torch.stack(dec_soft_idxs))
            print('dec_soft_idxs:', dec_soft_idxs)
            onehot_input_model2 = onehot_input_model2.scatter_(1, index_tensor, 1.).float().detach() + (dec_soft_idxs).sum() - (dec_soft_idxs).sum().detach()
            print('onehot_input_model2:', onehot_input_model2)

            pseudo_input = onehot_input_model2 
            print('pseudo input:', pseudo_input, pseudo_input.requires_grad)

            #model2 forward pass
            enc_hidden, enc_outputs = model2.enc_forward(pseudo_input)
            loss = model2.dec_forward(pseudo_target, enc_hidden, enc_outputs) # todo: find loss for each instnce and multiply A with the loss vec.
            print('\nloss2:\n', loss)
            #print('loss:', loss)
            batch_loss += loss 

            #del onehot_input_encoder1 , onehot_input_model2
                
            gc.collect()   
            #print('batch_loss:', batch_loss)
    return batch_loss/batch_size
        # except:
        #     print('skipping this_______________________________________')

        

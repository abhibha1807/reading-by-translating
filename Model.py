import torch
import json
import torch.nn.functional as F
from transformers import BertModel, BertForMaskedLM, BertConfig, EncoderDecoderModel
from losses import compute_loss1, compute_loss2
from utils import _concat, calc_bleu, loadTokenizer
from transformers import BertModel, BertConfig
from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel

'''
Run once to load BERT encoder-decoder models from hugging face library and 
save them in the 'models' directory
'''
# with open('config.json', "r") as f:
#     config = json.load(f)
# en=config["encoder_params"]
# de=config["decoder_params"]
# config_encoder = BertConfig(vocab_size=en["vocab_size"], hidden_size=en["hidden_size"], num_hidden_layers=en["num_hidden_layers"],
#                                     num_attention_heads=en["num_attn_heads"])
# config_decoder = BertConfig(vocab_size=de["vocab_size"], hidden_size=de["hidden_size"], num_hidden_layers=de["num_hidden_layers"],
#                                     num_attention_heads=de["num_attn_heads"])
# config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)

# model1 = EncoderDecoderModel(config=config)
# model2 = EncoderDecoderModel(config=config)

# config_decoder.is_decoder = True
# config_decoder.add_cross_attention = True

# model1 = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased') # initialize Bert2Bert from pre-trained checkpoints
# model2 = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased') # initialize Bert2Bert from pre-trained checkpoints
# model1.save_pretrained(save_directory='./models/model1')
# model2.save_pretrained(save_directory='./models/model2')

'''
MT model class to load pretrained models from the 'models' directory and performs 
training according to the  pipeline described in the RBT paper. Training occurs in 3 steps
1) Train first MT model using matrix A to calculate loss
2) Train second MT model on a dataset created by first MT model on the unlabeled dataset
3) Estimate A by reducing the validation loss of second MT model on validation set of MT dataset
'''
class TranslationModel:
    def __init__(self, device, batch_size, logging, model1_path, model2_path, config):
        self.model1=EncoderDecoderModel.from_pretrained(model1_path)
        self.model2=EncoderDecoderModel.from_pretrained(model2_path)
        self.device=device
        self.batch_size=batch_size
        print('model in device:', self.device)
        self.model1 = self.model1.cuda()
        self.model2 = self.model2.cuda()
        self.logger=logging
        self.config=config
        
    #scheduler1
    def train_model1(self, A_batch, train_dataloader, optimizer1, tokenizer, criterion, ):
        self.model1.train()
        epoch_loss = 0
        num_train_batches = len(train_dataloader)

        for i, ((en_input, en_masks, de_output, de_masks), a) in enumerate(zip(train_dataloader, A_batch)):
            optimizer1.zero_grad()
            en_input = en_input.to(self.device) 
            de_output = de_output.to(self.device)
            en_masks = en_masks.to(self.device)
            de_masks = de_masks.to(self.device)
            lm_labels = de_output.clone().to(self.device)

            out = self.model1(input_ids=en_input, attention_mask=en_masks, decoder_input_ids=de_output, 
                                decoder_attention_mask=de_masks, labels=lm_labels)
                
            predictions = F.log_softmax(out[1], dim=2)
            loss1=compute_loss1(predictions, de_output, a, self.device, criterion)
            print('loss1:', loss1)
            epoch_loss+=loss1.item()
            loss1.backward(inputs=list(self.model1.parameters()), retain_graph=True) 
            #torch.nn.utils.clip_grad_norm_(self.model1.parameters(), self.config["model1"]['grad_clip'])
            optimizer1.step() # wt updation  
            # scheduler1.step() 
            #print('step 1 instances gone:', (i+1)*self.batch_size)

            if ((i+1)*self.batch_size)% self.config['report_freq'] == 0:
                self.logger.info('loss after %d instances: %d', ((i+1)*self.batch_size), loss1.item())
                self.logger.info('bleu score after %d instances: %d', (i+1)*self.batch_size, calc_bleu(en_input, lm_labels, self.model1, tokenizer))
        
        self.logger.info('Mean epoch loss for step 1: %d', epoch_loss)
        #print("Mean epoch loss for step 1:", (epoch_loss / num_train_batches))
        return ((epoch_loss / num_train_batches))
    #scheduler2
    def train_model2(self, unlabeled_dataloader, optimizer2, tokenizer, criterion, ):
        epoch_loss=0
        self.model2.train()
        num_train_batches = len(unlabeled_dataloader)
        #num_train_batches = 2
        for i, (en_input, en_masks, de_output, de_masks) in enumerate(unlabeled_dataloader):
            optimizer2.zero_grad()
            en_input = en_input.to(self.device)
            outputs=self.model1(input_ids=en_input, decoder_input_ids=en_input, output_hidden_states=True, return_dict=True)
            predictions = F.log_softmax(outputs.logits, dim=2)
            values, new_labels = torch.max(predictions, 2)
            
            out=self.model2(input_ids=en_input, decoder_inputs_embeds=outputs.decoder_hidden_states[-1], labels=new_labels)
            predictions = F.log_softmax(out[1], dim=2)
            loss2=compute_loss2(predictions, new_labels, self.device, criterion)

            epoch_loss += loss2.item()
            loss2.backward(inputs=list(self.model2.parameters()), retain_graph=True)
            #torch.nn.utils.clip_grad_norm_(self.model2.parameters(), self.config["model2"]['grad_clip'])
            optimizer2.step()
            # scheduler2.step()
            #print('step 2 instances gone:', (i+1)*self.batch_size)
            
            if ((i+1)*self.batch_size)% self.config['report_freq'] == 0:
                self.logger.info('loss after %d instances: %d', ((i+1)*self.batch_size), loss2.item())
                self.logger.info('bleu score after %d instances: %d', (i+1)*self.batch_size, calc_bleu(en_input, new_labels, self.model2, tokenizer))

        self.logger.info('Mean epoch loss for step 2: %d', (epoch_loss ))
        
        #print("Mean epoch loss for step 2:", (epoch_loss / num_train_batches))
        return ((epoch_loss / num_train_batches))

    
    #a_ind
    #scheduler3
    def val_model2(self, valid_dataloader, optimizer3, A, A_batch, tokenizer, criterion, ):
        epoch_loss=0
        self.model2.eval()
        a_ind=0
        optimizer3.zero_grad()
        # A.grad=torch.zeros(len(A), device=self.device)
        A.grad=torch.zeros(len(A), device='cuda')

        for i, ((en_input, en_masks, de_output, de_masks), a) in enumerate(zip(valid_dataloader, A_batch)):
            optimizer3.zero_grad()
            en_input = en_input.to(self.device)
            de_output = de_output.to(self.device)
            en_masks = en_masks.to(self.device)
            de_masks = de_masks.to(self.device)
            lm_labels = de_output.clone().to(self.device)
            
            out=self.model2(input_ids=en_input, attention_mask=en_masks, decoder_input_ids=de_output, 
                            decoder_attention_mask=de_masks, labels=de_output.clone())
            predictions = F.log_softmax(out[1], dim=2)
            loss3 = compute_loss2(predictions, de_output, self.device, criterion)
            print('loss3:', loss3)
            epoch_loss+=loss3.item()

            # for p in list(self.model2.parameters()):
            #     print(p)
            #     break
            loss3.backward(inputs=list(self.model2.parameters()), retain_graph=True)
            # for p in list(self.model2.parameters()):
            #     print(p)
            #     break
        
            # t = torch.cuda.get_device_properties(0).total_memory
            # r = torch.cuda.memory_reserved(0) 
            # al = torch.cuda.memory_allocated(0)
            # f = r-al  # free inside reserved
            # print('freeeee:', f)
            
            '''
            Implementation of chain rule: eq 8,9 and 10
            Note: conidering E and F in the paper as Wo -> first MT model's weights, W as weights of
            second MT model, A as matrix A and L as loss at step 3
            delL/delA = delWo/delA x delW/delWo x delL/delW 
            Hessian vector product calculated using finite difference approximation
            to calculate above mentioned chain rule.
            '''

            # compute hessian vector product
            # calculate delW/delWo x delL/delW 
            r=1e-2
            vector=[]
            for param in self.model2.parameters():
                # param.to(self.device)
                if param.grad!=None:
                    vector.append(param.grad.data.to(self.device))
                    #vector.append(param.grad.data)
                else:
                    vector.append(torch.ones(1).to(self.device))
                    #vector.append(torch.ones(1))
            print('vector:', vector[0])
            R = r / _concat(vector, self.device).norm()

            print('R:', R)
            for p, v in zip(self.model2.parameters(), vector):
                print('before p.data:', p.data)
                p.data.to(self.device)
                with torch.no_grad():
                    torch.Tensor.add_(p, R, v)
                #p.data.add_(alpha=R, other=v)
                print('after p.data:', p.data)
                break
                #p.data.to(self.device)
            t = torch.cuda.get_device_properties(0).total_memory
            r = torch.cuda.memory_reserved(0) 
            al = torch.cuda.memory_allocated(0)
            f = r-al  # free inside reserved
            print('freeeee:', f)
                        
            #calculate loss
            outputs=self.model1(input_ids=en_input, decoder_input_ids=en_input, output_hidden_states=True, return_dict=True)
            predictions = F.log_softmax(outputs.logits, dim=2)
            values, new_labels = torch.max(predictions, 2)
            
            out=self.model2(input_ids=en_input, decoder_inputs_embeds=outputs.decoder_hidden_states[-1], labels=new_labels)
            predictions = F.log_softmax(out[1], dim=2)
            loss2=compute_loss2(predictions, new_labels, self.device, criterion)
            
            grads_p=torch.autograd.grad(loss2, self.model1.parameters(), allow_unused=True)

            del loss2
            del predictions
            del out 
            del outputs
            del new_labels
            torch.cuda.empty_cache()
            for p, v in zip(self.model2.parameters(), vector):
                p.data.to(self.device)
                with torch.no_grad():
                    torch.Tensor.sub_(p, 2*R, v)
                #p.data.sub_(alpha=2 * R, other=v)
               
            
            #calculate loss
            outputs=self.model1(input_ids=en_input, decoder_input_ids=en_input, output_hidden_states=True, return_dict=True)
            predictions = F.log_softmax(outputs.logits, dim=2)
            values, new_labels = torch.max(predictions, 2)
            
            out=self.model2(input_ids=en_input, decoder_inputs_embeds=outputs.decoder_hidden_states[-1], labels=new_labels)
            predictions = F.log_softmax(out[1], dim=2)
            loss2=compute_loss2(predictions, new_labels, self.device, criterion)
        
            grads_n = torch.autograd.grad(loss2, self.model1.parameters(), allow_unused=True)

            del loss2
            del predictions
            del out 
            del outputs
            del new_labels
            for p, v in zip(self.model2.parameters(), vector):
                p.data.to(self.device)
                with torch.no_grad():
                    torch.Tensor.add_(p, R, v)
                #p.data.add_(R, v)
            
            del vector
            torch.cuda.empty_cache()
            t = torch.cuda.get_device_properties(0).total_memory
            r = torch.cuda.memory_reserved(0) 
            al = torch.cuda.memory_allocated(0)
            f = r-al  # free inside reserved
            print('freeeee:', f)
            vector=[]
            for x,y in zip(grads_p, grads_n):
                if x!=None and y!=None:
                    vector.append(((x - y).div_(2 * R)).to(self.device))
                    #vector.append((x - y).div_(2 * R))
                else:
                    vector.append(torch.ones(1, device=self.device))
                    #vector.append(torch.ones(1))
            
            del grads_n
            del grads_p
            torch.cuda.empty_cache()
            # calculate delL/delA = delWo/delA x delW/delWo x delL/delW 
            for p, v in zip(self.model1.parameters(), vector):
                p.to(self.device)
                with torch.no_grad():
                    torch.Tensor.add_(p, R, v)
                #p.data.add_(alpha=R, other=v)
                
            #calculate loss
            out = self.model1(input_ids=en_input, attention_mask=en_masks, decoder_input_ids=de_output, 
                                decoder_attention_mask=de_masks, labels=lm_labels.clone())
                
            predictions = F.log_softmax(out[1], dim=2)
            loss1=compute_loss1(predictions, de_output, a, self.device, criterion)    

            grads_p=torch.autograd.grad(loss1, a, allow_unused=True, retain_graph=True)

            for p, v in zip(self.model1.parameters(), vector):
                p.to(self.device)
                with torch.no_grad():
                    torch.Tensor.sub_(p, 2*R, v)
                #p.data.sub_(2 * R, v)

            del out
            del predictions
            del loss1
            torch.cuda.empty_cache()
            #calculate loss
            out = self.model1(input_ids=en_input, attention_mask=en_masks, decoder_input_ids=de_output, 
                                decoder_attention_mask=de_masks, labels=lm_labels.clone())
                
            predictions = F.log_softmax(out[1], dim=2)
            loss1=compute_loss1(predictions, de_output, a, self.device, criterion)    

            grads_n=torch.autograd.grad(loss1, a, allow_unused=True, retain_graph=True)

            del out
            del predictions
            del loss1
            torch.cuda.empty_cache()
            for p, v in zip(self.model1.parameters(), vector):
                p.to(self.device)
                with torch.no_grad():
                    torch.Tensor.add_(p, R, v)
                #p.data.add_(R, v)

            A.grad[a_ind:a_ind+self.batch_size]=[(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)][0]
            print('before A.grad:', A.grad)
            del grads_p
            del grads_n
            torch.cuda.empty_cache()
            
            #torch.nn.utils.clip_grad_norm_(A, 1e-2) 
            print('after A.grad:', A.grad)
            print('before A:', A)
            optimizer3.step()
            print('finallyyyy:', A) 
            # scheduler3.step()
            # print('before a_ind:', a_ind)
            a_ind=a_ind+self.batch_size
            # print('after a_ind:', a_ind)
            A.grad=torch.zeros(len(A), device=self.device)
            print('step 3 instances gone:', (i+1)*self.batch_size)
            if ((i+1)*self.batch_size)% self.config['report_freq'] == 0:
                self.logger.info('loss after %d instances: %d', (i+1)*self.batch_size, loss3.item())
                self.logger.info('bleu score after %d instances: %d', (i+1)*self.batch_size, calc_bleu(en_input, lm_labels, self.model2, tokenizer))
            # break

        self.logger.info('Mean epoch loss for step 3: %d', (epoch_loss )) 
            
        print("Mean epoch loss for step 3:", (epoch_loss / len(valid_dataloader)))
        return (epoch_loss / len(valid_dataloader))

    def save_model(self, save_directory):
        print('saving models')
        save_directory='./saved_models'
        self.model2.save_pretrained(save_directory= save_directory+'/model2')
        self.model1.save_pretrained(save_directory= save_directory+'/model1')
    
    #TO-DO
    def infer(self, test_dataloader, criterion):
        for i, ((en_input, en_masks, de_output, de_masks)) in enumerate(zip(test_dataloader)):
            en_input = en_input.to(self.device) 
            de_output = de_output.to(self.device)
            en_masks = en_masks.to(self.device)
            de_masks = de_masks.to(self.device)
            lm_labels = de_output.clone().to(self.device)

            out = self.model1(input_ids=en_input, attention_mask=en_masks, decoder_input_ids=de_output, 
                                decoder_attention_mask=de_masks, labels=lm_labels.clone())
                
            predictions = F.log_softmax(out[1], dim=2)
            loss=compute_loss1(predictions, de_output, self.device, criterion)

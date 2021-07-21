import torch
import torch.nn.functional as F
from transformers import BertModel, BertForMaskedLM, BertConfig, EncoderDecoderModel
from losses import compute_loss1, compute_loss2
from utils import _concat, calc_bleu, loadTokenizer

'''
Run once to load BERT encoder-decoder models from hugging face library and 
save them in the 'models' directory
'''
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
    def __init__(self, device, batch_size, logging, config):
        self.model1=EncoderDecoderModel.from_pretrained(config["model1"]['model_path'])
        self.model2=EncoderDecoderModel.from_pretrained(config["model1"]['model_path'])
        self.device=device
        self.batch_size=batch_size
        print(device)
        print('model in device:', self.device)
        self.model1 = self.model1.cuda()
        self.model2 = self.model2.cuda()
        self.logger=logging
        self.config=config
        

    def train_model1(self, A_batch, train_dataloader, optimizer1, tokenizer, criterion, scheduler1):
        self.model1.train()
        epoch_loss = 0
        optimizer1.zero_grad()
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
            loss1=compute_loss1(predictions, de_output, a, self.device , criterion)
            epoch_loss+=loss1.item()
            loss1.backward(inputs=list(self.model1.parameters()), retain_graph=True) 
            torch.nn.utils.clip_grad_norm_(self.model1.parameters(), self.config["model1"]['grad_clip'])
            optimizer1.step() # wt updation  
            scheduler1.step() 
            
            #print('step 1 instances gone:', (i+1)*self.batch_size)

            if ((i+1)*self.batch_size)% self.config['report_freq'] == 0:
                self.logger.info('loss after %d instances: %d', (i+1)*self.batch_size, epoch_loss)
                self.logger.info('bleu score after %d instances: %d', (i+1)*self.batch_size, calc_bleu(en_input, lm_labels, self.model1, tokenizer))
                break

        self.logger.info('Mean epoch loss for step 1: %d', (epoch_loss / num_train_batches))
        #print("Mean epoch loss for step 1:", (epoch_loss / num_train_batches))
        return ((epoch_loss / num_train_batches))

    def train_model2(self, unlabeled_dataloader, optimizer2, tokenizer, criterion, scheduler2):
        epoch_loss=0
        optimizer2.zero_grad()
        self.model2.train()
        num_train_batches = len(unlabeled_dataloader)
        # num_train_batches = 2
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
            torch.nn.utils.clip_grad_norm_(self.model2.parameters(), self.config["model2"]['grad_clip'])
            optimizer2.step()
            scheduler2.step()
            #print('step 2 instances gone:', (i+1)*self.batch_size)
            
            if ((i+1)*self.batch_size)% self.config['report_freq'] == 0:
                self.logger.info('loss after %d instances: %d', (i+1)*self.batch_size, epoch_loss)
                self.logger.info('bleu score after %d instances: %d', (i+1)*self.batch_size, calc_bleu(en_input, new_labels, self.model2, tokenizer))
                break

        self.logger.info('Mean epoch loss for step 2: %d', (epoch_loss / num_train_batches))
        
        #print("Mean epoch loss for step 2:", (epoch_loss / num_train_batches))
        return ((epoch_loss / num_train_batches))

    
    

    def val_model2(self, valid_dataloader, optimizer3, A, A_batch, tokenizer, criterion, scheduler3):
        epoch_loss=0
        self.model2.train()
        a_ind=0
        optimizer3.zero_grad()
        A.grad=torch.zeros(len(A), device=self.device)

        for i, ((en_input, en_masks, de_output, de_masks), a) in enumerate(zip(valid_dataloader, A_batch)):
            en_input = en_input.to(self.device)
            de_output = de_output.to(self.device)
            en_masks = en_masks.to(self.device)
            de_masks = de_masks.to(self.device)
            lm_labels = de_output.clone().to(self.device)
            
            out=self.model2(input_ids=en_input, attention_mask=en_masks, decoder_input_ids=de_output, 
                            decoder_attention_mask=de_masks, labels=de_output.clone())
            predictions = F.log_softmax(out[1], dim=2)
            loss3 = compute_loss2(predictions, de_output, self.device, criterion)
            #print('loss3:', loss3)
            epoch_loss+=loss3.item()

            loss3.backward(inputs=list(self.model2.parameters()), retain_graph=True)
            
            #print('bleu-score:', calc_bleu(en_input, lm_labels, model2, de_tokenizer, 16))

            # compute hessian vector product
            r=1e-2
            vector=[]
            for param in self.model2.parameters():
            # param.to(device)
                if param.grad!=None:
                    vector.append(param.grad.data.to(self.device))
                    #vector.append(param.grad.data)
                else:
                    vector.append(torch.ones(1).to(self.device))
                #vector.append(torch.ones(1))
            
            #print(len(vector))
            #print('vector 0:', vector[0])
            # R = r / _concat(vector).norm().to(device)
            R = r / _concat(vector, self.device).norm().to(self.device)

            print(R)
            print(vector[0])
            for p, v in zip(self.model2.parameters(), vector):
            # print(p.shape, v.shape,c)
            #print('before:', p.data, v)
                p.data.to(self.device)
                print(p.data)
                p.data.add_(alpha=R, other=v)
                #print('after:', p.data)
                # break
            
            #calculate loss
            outputs=self.model1(input_ids=en_input, decoder_input_ids=en_input, output_hidden_states=True, return_dict=True)
            predictions = F.log_softmax(outputs.logits, dim=2)
            values, new_labels = torch.max(predictions, 2)
            
            out=self.model2(input_ids=en_input, decoder_inputs_embeds=outputs.decoder_hidden_states[-1], labels=new_labels)
            predictions = F.log_softmax(out[1], dim=2)
            loss2=compute_loss2(predictions, new_labels, self.device, criterion)
            print('loss2:', loss2)
            
            grads_p=torch.autograd.grad(loss2, self.model1.parameters(), allow_unused=True, retain_graph=True)
            print('gradsp:', (grads_p)[0])

            del loss2
            del predictions
            del out 
            del outputs
            del new_labels

            for p, v in zip(self.model2.parameters(), vector):
                # print('before:', p)
                p.data.to(self.device)
                #p.data.sub_(5.0)
                p.data.sub_(alpha=2 * R, other=v)
                # print('after:', p)
                # break
            #calculate loss
            outputs=self.model1(input_ids=en_input, decoder_input_ids=en_input, output_hidden_states=True, return_dict=True)
            predictions = F.log_softmax(outputs.logits, dim=2)
            values, new_labels = torch.max(predictions, 2)
            
            out=self.model2(input_ids=en_input, decoder_inputs_embeds=outputs.decoder_hidden_states[-1], labels=new_labels)
            predictions = F.log_softmax(out[1], dim=2)
            loss2=compute_loss2(predictions, new_labels, self.device, criterion)
            #print('loss2:', loss2)
        
            grads_n = torch.autograd.grad(loss2, self.model1.parameters(), allow_unused=True, retain_graph=True)
            print('gradsn:', (grads_n)[0])

            for p, v in zip(self.model2.parameters(), vector):
                p.data.to(self.device)
                p.data.add_(R, v)
                
            del loss2
            del predictions
            del out 
            del outputs
            del new_labels

            vector=[]
            for x,y in zip(grads_p, grads_n):
                if x!=None and y!=None:
                    vector.append(((x - y).div_(2 * R)).to(self.device))
                else:
                    vector.append(torch.ones(1, device=self.device))
                # print(len(final))
                #print('vector0:', vector[0])

            del grads_n
            del grads_p


            # vector=final
            c=0
            for p, v in zip(self.model1.parameters(), vector):
            # print(p.shape, v.shape,c)
                p.to(self.device)
                p.data.add_(alpha=R, other=v)
                c=c+1
            
            out = self.model1(input_ids=en_input, attention_mask=en_masks, decoder_input_ids=de_output, 
                                decoder_attention_mask=de_masks, labels=lm_labels.clone())
                
            predictions = F.log_softmax(out[1], dim=2)
            loss1=compute_loss1(predictions, de_output, a, self.device, criterion)    
            print('loss1:', loss1)

            grads_p=torch.autograd.grad(loss1, a, allow_unused=True, retain_graph=True)
            grads_p[0].to(self.device)
            print('grads_p:', grads_p[0])

            for p, v in zip(self.model1.parameters(), vector):
                p.to(self.device)
                #print('before:', p.data)
                p.data.sub_(2 * R, v)
                #p.data.sub_(5.0)
                #print('after:', p.data)
                # break
            del out
            del predictions
            del loss1
            
            out = self.model1(input_ids=en_input, attention_mask=en_masks, decoder_input_ids=de_output, 
                                decoder_attention_mask=de_masks, labels=lm_labels.clone())
                
            predictions = F.log_softmax(out[1], dim=2)
            loss1=compute_loss1(predictions, de_output, a, self.device, criterion)    
            #print('loss1:', loss1)

            grads_n=torch.autograd.grad(loss1, a, allow_unused=True, retain_graph=True)
            grads_n[0].to(self.device)
            print('grads_n:', grads_n[0])

            del out
            del predictions
            del loss1

            for p, v in zip(self.model1.parameters(), vector):
                p.to(self.device)
                p.data.add_(R, v)
            
            print([(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)])
            # A.grad=[]

            A.grad[a_ind:a_ind+self.batch_size]=[(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)][0]
            print(A.grad)

            del grads_p
            del grads_n
          
            # torch.nn.utils.clip_grad_norm_(A, 1e-2) 
            print('before A:', A)
            optimizer3.step()
            # del A.grad
            print('finallyyyy:', A) 
            
            optimizer3.zero_grad()
            a_ind+=self.batch_size
            print('instances gone:', (i+1)*self.batch_size)
            break
            
        print("Mean epoch loss for step 3:", (epoch_loss / len(valid_dataloader)))
        return (epoch_loss / len(valid_dataloader))

    def save_model(self, save_directory):
        print('saving models')
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


    
    # def val_model2(self, valid_dataloader, optimizer3, A, A_batch, tokenizer, criterion, scheduler3):
    #     epoch_loss=0
    #     self.model2.eval()
    #     a_ind=0
    #     optimizer3.zero_grad()
    #     A.grad=torch.zeros(len(A), device='cpu')
    #     print('len A.grad:', len(A.grad))
    #     for i, ((en_input, en_masks, de_output, de_masks), a) in enumerate(zip(valid_dataloader, A_batch)):
    #         optimizer3.zero_grad()

    #         en_input = en_input.to(self.device)
    #         de_output = de_output.to(self.device)
    #         en_masks = en_masks.to(self.device)
    #         de_masks = de_masks.to(self.device)
    #         lm_labels = de_output.clone().to(self.device)
            
    #         out=self.model2(input_ids=en_input, attention_mask=en_masks, decoder_input_ids=de_output, 
    #                         decoder_attention_mask=de_masks, labels=lm_labels)
    #         predictions = F.log_softmax(out[1], dim=2)
    #         loss3 = compute_loss2(predictions, de_output, 'cpu', criterion)
    #         print('loss3:', loss3)
    #         epoch_loss+=loss3.item()
    #         del out
    #         loss3.backward(inputs=list(self.model2.parameters()), retain_graph=True)

    #         if ((i+1)*self.batch_size)% self.config['report_freq'] == 0:
    #             self.logger.info('loss after %d instances: %d', (i+1)*self.batch_size, epoch_loss)
    #             self.logger.info('bleu score after %d instances: %d', (i+1)*self.batch_size, calc_bleu(en_input, lm_labels, self.model2, tokenizer))

            
    #         '''
    #         Implementation of chain rule: eq 8,9 and 10
    #         Note: conidering E and F in the paper as Wo -> first MT model's weights, W as weights of
    #         second MT model, A as matrix A and L as loss at step 3
    #         delL/delA = delWo/delA x delW/delWo x delL/delW 
    #         Hessian vector product calculated using finite difference approximation
    #         to calculate above mentioned chain rule.
    #         '''

    #         # compute hessian vector product
    #         # calculate delW/delWo x delL/delW 
    #         r=1e-2
    #         vector=[]
    #         for param in self.model2.parameters():
    #             param.to(self.device)
    #             if param.grad!=None:
    #                 vector.append(param.grad.data.to(self.device))
    #             else:
    #                 vector.append(torch.ones(1).to(self.device))

    #         #R = r / _concat(vector, self.device).norm().to(self.device)
    #         R = r / _concat(vector, 'cuda').norm().to('cpu')
    #         print(R)
    #         for p, v in zip(self.model2.parameters(), vector):
    #             p.data.to(self.device)
    #             p.data.add_(alpha=R, other=v)
    #             #p.data.to(self.device)
                        
    #         #calculate loss
    #         outputs=self.model1(input_ids=en_input, decoder_input_ids=en_input, output_hidden_states=True, return_dict=True)
    #         predictions = F.log_softmax(outputs.logits, dim=2)
    #         values, new_labels = torch.max(predictions, 2)
            
    #         out=self.model2(input_ids=en_input, decoder_inputs_embeds=outputs.decoder_hidden_states[-1], labels=new_labels)
    #         predictions = F.log_softmax(out[1], dim=2)
    #         loss2=compute_loss2(predictions, new_labels, 'cpu', criterion)
    #         print('loss2:', loss2)
            
    #         grads_p=torch.autograd.grad(loss2, self.model1.parameters(), allow_unused=True, retain_graph=True)
            
    #         print('gradsp:', (grads_p)[0])
            
    #         del loss2
    #         del predictions
    #         del out 
    #         del outputs
    #         del new_labels

    #         for p, v in zip(self.model2.parameters(), vector):
    #             p.data.to(self.device)
    #             p.data.sub_(alpha=2 * R, other=v)
               
            
    #         #calculate loss
    #         outputs=self.model1(input_ids=en_input, decoder_input_ids=en_input, output_hidden_states=True, return_dict=True)
    #         predictions = F.log_softmax(outputs.logits, dim=2)
    #         values, new_labels = torch.max(predictions, 2)
    #         del values
    #         out=self.model2(input_ids=en_input, decoder_inputs_embeds=outputs.decoder_hidden_states[-1], labels=new_labels)
    #         predictions = F.log_softmax(out[1], dim=2)
    #         loss2=compute_loss2(predictions, new_labels, 'cpu', criterion)
        
    #         grads_n = torch.autograd.grad(loss2, self.model1.parameters(), allow_unused=True, retain_graph=True)
    #         print('gradsn:', (grads_n)[0])

    #         for p, v in zip(self.model2.parameters(), vector):
    #             p.data.to(self.device)
    #             p.data.add_(R, v)
            
    #         del loss2
    #         del predictions
    #         del out 
    #         del outputs
    #         del new_labels
            
    #         del vector

    #         # vector=[]
    #         # for x,y in zip(grads_p, grads_n):
    #         #     if x!=None and y!=None:
    #         #         vector.append(((x - y).div_(2 * R)).to(self.device))
    #         #         #vector.append(((x - y).div_(2 * R)))
    #         #     else:
    #         #         vector.append(torch.ones(1, device=self.device))
    #         #         #vector.append(torch.ones(1))
            
    #         # del grads_n
    #         # del grads_p

    #         # # calculate delL/delA = delWo/delA x delW/delWo x delL/delW 
    #         # for p, v in zip(self.model1.parameters(), vector):
    #         #     p.to(self.device)
    #         #     p.data.add_(alpha=R, other=v)
                
    #         # #calculate loss
    #         # out = self.model1(input_ids=en_input, attention_mask=en_masks, decoder_input_ids=de_output, 
    #         #                     decoder_attention_mask=de_masks, labels=lm_labels)
                
    #         # predictions = F.log_softmax(out[1], dim=2)
    #         # loss1=compute_loss1(predictions, de_output, a, 'cpu', criterion)    
    #         # print('loss1:', loss1)
            
    #         # grads_p=torch.autograd.grad(loss1, a, allow_unused=True, retain_graph=True)
    #         # print('gradsp:', (grads_p)[0])
            
    #         # for p, v in zip(self.model1.parameters(), vector):
    #         #     p.to(self.device)
    #         #     p.data.sub_(2 * R, v)

    #         # del out
    #         # del predictions
    #         # del loss1
            
    #         # #calculate loss
    #         # out = self.model1(input_ids=en_input, attention_mask=en_masks, decoder_input_ids=de_output, 
    #         #                     decoder_attention_mask=de_masks, labels=lm_labels.clone())
                
    #         # predictions = F.log_softmax(out[1], dim=2)
    #         # loss1=compute_loss1(predictions, de_output, a, 'cpu', criterion)    

    #         # grads_n=torch.autograd.grad(loss1, a, allow_unused=True, retain_graph=True)
    #         # print('gradsn:', (grads_n)[0])
    #         # del out
    #         # del predictions
    #         # del loss1

    #         # for p, v in zip(self.model1.parameters(), vector):
    #         #     p.to(self.device)
    #         #     p.data.add_(R, v)

    #         # A.grad[a_ind:a_ind+self.batch_size]=[(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)][0]

    #         # del grads_p
    #         # del grads_n
          
    #         # # torch.nn.utils.clip_grad_norm_(A, 1e-2) 
    #         # optimizer3.step()
    #         # scheduler3.step()
    #         # a_ind+=self.batch_size
    #         # print('step 3 instances gone:', (i+1)*self.batch_size)
          
    #     self.logger.info('Mean epoch loss for step 3: %d', (epoch_loss / len(valid_dataloader))) 
            
    #     #print("Mean epoch loss for step 3:", (epoch_loss / len(valid_dataloader)))
    #     return (epoch_loss / len(valid_dataloader))

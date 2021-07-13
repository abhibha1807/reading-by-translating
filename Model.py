import torch
import torch.nn.functional as F
from transformers import BertModel, BertForMaskedLM, BertConfig, EncoderDecoderModel
from losses import compute_loss1, compute_loss2
from utils import _concat, calc_bleu, loadTokenizer
# run once
# model1 = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased') # initialize Bert2Bert from pre-trained checkpoints
# model2 = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased') # initialize Bert2Bert from pre-trained checkpoints
# model1.save_pretrained(save_directory='./models/model1')
# model2.save_pretrained(save_directory='./models/model2')

class model:
    def __init__(self, device, batch_size, logging, config):
        self.model1=EncoderDecoderModel.from_pretrained(config["model1"]['model_path'])
        self.model2=EncoderDecoderModel.from_pretrained(config["model1"]['model_path'])
        self.device=device
        self.batch_size=batch_size
        if self.device=='cuda':
            self.model1.cuda()
            self.model2.cuda()
        self.logger=logging
        self.config=config
        
        

    # step 1
    def train_model1(self, A_batch, train_dataloader, optimizer1, tokenizer):
        self.model1.train()
        epoch_loss = 0
        self.model1.train()
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
                                decoder_attention_mask=de_masks, labels=lm_labels.clone())
                
            predictions = F.log_softmax(out[1], dim=2)
            loss1=compute_loss1(predictions, de_output, a, self.device)
            print(loss1)
            epoch_loss+=loss1.item()
            loss1.backward(inputs=list(self.model1.parameters()), retain_graph=True) 
            torch.nn.utils.clip_grad_norm(self.model1.parameters(), self.config["model1"]['grad_clip'])
            optimizer1.step() # wt updation   
            print('step 1 instances gone:', (i+1)*self.batch_size)

            if ((i+1)*self.batch_size)% self.config['report_freq'] == 0:
                self.logger.info('loss after %d instances: %d', (i+1)*self.batch_size, loss1.item())
                self.logger.info('bleu score after %d instances: %d', (i+1)*self.batch_size, calc_bleu(en_input, lm_labels, self.model1, tokenizer))
        
        self.logger.info('Mean epoch loss for step 1: %d', (epoch_loss / num_train_batches))
        #print("Mean epoch loss for step 1:", (epoch_loss / num_train_batches))
        return ((epoch_loss / num_train_batches))

    def train_model2(self, train_dataloader, optimizer2, tokenizer):
        epoch_loss=0
        optimizer2.zero_grad()
        self.model2.train()
        num_train_batches = len(train_dataloader)
        for i, (en_input, en_masks, de_output, de_masks) in enumerate(train_dataloader):
            en_input = en_input.to(self.device)
            outputs=self.model1(input_ids=en_input, decoder_input_ids=en_input, output_hidden_states=True, return_dict=True)
            predictions = F.log_softmax(outputs.logits, dim=2)
            values, new_labels = torch.max(predictions, 2)
            
            out=self.model2(input_ids=en_input, decoder_inputs_embeds=outputs.decoder_hidden_states[-1], labels=new_labels)
            predictions = F.log_softmax(out[1], dim=2)
            loss2=compute_loss2(predictions, new_labels, self.device)

            epoch_loss += loss2.item()
            loss2.backward(inputs=list(self.model2.parameters()), retain_graph=True)
            torch.nn.utils.clip_grad_norm(self.model2.parameters(), self.config["model2"]['grad_clip'])
            optimizer2.step()
            print('step 2 instances gone:', (i+1)*self.batch_size)
            if (i+1)%2 == 0:
                self.logger.info('loss after %d instances: %d', (i+1)*self.batch_size, loss2.item())
                self.logger.info('bleu score after %d instances: %d', (i+1)*self.batch_size, calc_bleu(en_input, new_labels, self.model2, tokenizer))

        self.logger.info('Mean epoch loss for step 2: %d', (epoch_loss / num_train_batches))
        
        #print("Mean epoch loss for step 2:", (epoch_loss / num_train_batches))
        return ((epoch_loss / num_train_batches))

    
        
    def val_model2(self, valid_dataloader, optimizer3, A, A_batch, tokenizer):
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
            loss3 = compute_loss2(predictions, de_output, self.device)
            print('loss3:', loss3)
            epoch_loss+=loss3.item()

            loss3.backward(inputs=list(self.model2.parameters()), retain_graph=True)

            # compute hessian vector product
            # calculate delW/delE x delL/delW
            r=1e-2
            vector=[]
            for param in self.model2.parameters():
                param.to(self.device)
                if param.grad!=None:
                    vector.append(param.grad.data.to(self.device))
                else:
                    vector.append(torch.ones(1).to(self.device))
            
            R = r / _concat(vector, self.device).norm().to(self.device)
            print(R)
            for p, v in zip(self.model2.parameters(), vector):
                p.data.to(self.device)
                p.data.add_(alpha=R, other=v)
                        
            #calculate loss
            outputs=self.model1(input_ids=en_input, decoder_input_ids=en_input, output_hidden_states=True, return_dict=True)
            predictions = F.log_softmax(outputs.logits, dim=2)
            values, new_labels = torch.max(predictions, 2)
            
            out=self.model2(input_ids=en_input, decoder_inputs_embeds=outputs.decoder_hidden_states[-1], labels=new_labels)
            predictions = F.log_softmax(out[1], dim=2)
            loss2=compute_loss2(predictions, new_labels, self.device)
            
            grads_p=torch.autograd.grad(loss2, self.model1.parameters(), allow_unused=True, retain_graph=True)

            for p, v in zip(self.model2.parameters(), vector):
                p.data.to(self.device)
                p.data.sub_(alpha=2 * R, other=v)
               
            
            #calculate loss
            outputs=self.model1(input_ids=en_input, decoder_input_ids=en_input, output_hidden_states=True, return_dict=True)
            predictions = F.log_softmax(outputs.logits, dim=2)
            values, new_labels = torch.max(predictions, 2)
            
            out=self.model2(input_ids=en_input, decoder_inputs_embeds=outputs.decoder_hidden_states[-1], labels=new_labels)
            predictions = F.log_softmax(out[1], dim=2)
            loss2=compute_loss2(predictions, new_labels, self.device)
        
            grads_n = torch.autograd.grad(loss2, self.model1.parameters(), allow_unused=True, retain_graph=True)

            for p, v in zip(self.model2.parameters(), vector):
                p.data.to(self.device)
                p.data.add_(R, v)

            vector=[]
            for x,y in zip(grads_p, grads_n):
                if x!=None and y!=None:
                    vector.append(((x - y).div_(2 * R)).to(self.device))
                else:
                    vector.append(torch.ones(1, device=self.device))

            # calculate delE/delA x delL/delE
            for p, v in zip(self.model1.parameters(), vector):
                p.to(self.device)
                p.data.add_(alpha=R, other=v)
                
            #calculate loss
            out = self.model1(input_ids=en_input, attention_mask=en_masks, decoder_input_ids=de_output, 
                                decoder_attention_mask=de_masks, labels=lm_labels.clone())
                
            predictions = F.log_softmax(out[1], dim=2)
            loss1=compute_loss1(predictions, de_output, a)    

            grads_p=torch.autograd.grad(loss1, a, allow_unused=True, retain_graph=True)

            for p, v in zip(self.model1.parameters(), vector):
                p.to(self.device)
                p.data.sub_(2 * R, v)
            
            #calculate loss
            out = self.model1(input_ids=en_input, attention_mask=en_masks, decoder_input_ids=de_output, 
                                decoder_attention_mask=de_masks, labels=lm_labels.clone())
                
            predictions = F.log_softmax(out[1], dim=2)
            loss1=compute_loss1(predictions, de_output, a)    

            grads_n=torch.autograd.grad(loss1, a, allow_unused=True, retain_graph=True)

            for p, v in zip(self.model1.parameters(), vector):
                p.to(self.device)
                p.data.add_(R, v)

            A.grad[a_ind:a_ind+self.batch_size]=[(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)][0]
          
            # torch.nn.utils.clip_grad_norm_(A, 1e-2) 
            optimizer3.step()
            a_ind+=self.batch_size
            print('step 3 instances gone:', (i+1)*self.batch_size)
            if (i+1)%2 == 0:
                self.logger.info('loss after %d instances: %d', (i+1)*self.batch_size, loss2.item())
                self.logger.info('bleu score after %d instances: %d', (i+1)*self.batch_size, calc_bleu(en_input, lm_labels, self.model2, tokenizer))

        self.logger.info('Mean epoch loss for step 2: %d', (epoch_loss / len(valid_dataloader))) 
            
        #print("Mean epoch loss for step 3:", (epoch_loss / len(valid_dataloader)))
        return (epoch_loss / len(valid_dataloader))

    def save_model(self, save_directory):
        print('saving models')
        self.model2.save_pretrained(save_directory= save_directory+'/model2')
        self.model1.save_pretrained(save_directory= save_directory+'/model1')
    
    def infer(self, test_dataloader):
        for i, ((en_input, en_masks, de_output, de_masks)) in enumerate(zip(test_dataloader)):
            en_input = en_input.to(self.device) 
            de_output = de_output.to(self.device)
            en_masks = en_masks.to(self.device)
            de_masks = de_masks.to(self.device)
            lm_labels = de_output.clone().to(self.device)

            out = self.model1(input_ids=en_input, attention_mask=en_masks, decoder_input_ids=de_output, 
                                decoder_attention_mask=de_masks, labels=lm_labels.clone())
                
            predictions = F.log_softmax(out[1], dim=2)
            loss=compute_loss1(predictions, de_output, self.device)
            
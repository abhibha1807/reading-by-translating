import gc
from torch.autograd import Variable
import torch
from losses import *
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs if x is not None])


class Architect(object):
  def __init__(self, model1, model1_mom, model1_wd, A, A_lr, A_wd, device, model2, model2_wd, model2_mom,batch_size,vocab):
    self.model1 = model1
    self.model1_mom = model1_mom
    self.model1_wd = model1_wd
    self.A = A
    self.A_optim =  torch.optim.Adam(self.A.parameters(),
          lr=A_lr, betas=(0.5, 0.999), weight_decay=A_wd)
    self.device = device
    self.model2 = model2
    self.model2_wd = model2_wd
    self.model2_mom = model2_mom
    self.batch_size = batch_size
    self.vocab = vocab
  

  
  def _compute_unrolled_enc_dec_model(self, train_inputs, model1_lr, idxs, model1_optim):
    batch_loss = loss1(train_inputs, self.model1, idxs, self.A,  self.batch_size, self.vocab)
    print('batch loss 1:', batch_loss)
    #Unrolled model
    theta = _concat(self.model1.parameters()).data
    # print(theta, len(theta))
    try:
        moment = _concat(model1_optim.state[v]['momentum_buffer'] for v in self.model1.parameters()).mul_(self.model1_mom)
    except:
        moment = torch.zeros_like(theta)
    # # print(moment)
    # dtheta = _concat(torch.autograd.grad(batch_loss, self.model1.parameters(), retain_graph = True, allow_unused=True )).data + self.model1_wd*theta
    # # print(dtheta)
    # # convert to the model
    # unrolled_model1 = self._construct_model1_from_theta(theta.sub(model1_lr, moment+dtheta))
    # #print(unrolled_enc)
    
    grads=torch.autograd.grad(batch_loss, self.model1.parameters(), retain_graph = True ,allow_unused=True)
    l=[]
    for i,(p,grad) in enumerate(zip( self.model1.parameters(),grads)):
        if grad is not None:
            l.append(grad)
        if grad is None:
            l.append(torch.autograd.Variable(torch.zeros(p.size()).type(torch.float32),requires_grad=True).cuda())
    dtheta = _concat(l).data + self.model1_wd*theta

    # convert to the model
    unrolled_model1 =self._construct_model1_from_theta(theta.sub(model1_lr, moment+dtheta))
    return unrolled_model1



  def _construct_model1_from_theta(self, theta):
    model1_dict = self.model1.state_dict()
  
    # create the new gpt model, input_lang, hidden_size, device
    model1_new = self.model1.new(self.vocab)
    #print('model1 new:', model1_new)

    #encoder update
    params, offset = {}, 0
    for k, v in self.model1.named_parameters():
        v_length = np.prod(v.size())
        params[k] = theta[offset: offset+v_length].view(v.size())
        offset += v_length

    assert offset == len(theta)
    model1_dict.update(params)
    model1_new.load_state_dict(model1_dict)
    #print([model2_new.state_dict()])
    return model1_new



  def _construct_model2_from_theta(self, theta):
    model2_dict = self.model2.state_dict()

    # create the new gpt model, input_lang, hidden_size, device
    model2_new = self.model2.new(self.vocab)
    # print('model2 new:', model2_new)

    #encoder update
    params, offset = {}, 0
    for k, v in self.model2.named_parameters():
        v_length = np.prod(v.size())
        params[k] = theta[offset: offset+v_length].view(v.size())
        offset += v_length

    assert offset == len(theta)
    model2_dict.update(params)
    model2_new.load_state_dict(model2_dict)
    #print([model2_new.state_dict()])
    return model2_new
  
  
  def _compute_unrolled_model2(self, un_inputs, unrolled_model1, idxs, model2_lr, model2_optim):
      batch_loss = loss2(un_inputs, unrolled_model1, self.model2,self.batch_size, self.vocab)
      theta = _concat(self.model2.parameters()).data
      try:
          moment = _concat(model2_optim.state[v]['momentum_buffer'] for v in self.model2.parameters()).mul_(self.model2_mom)
      except:
          moment = torch.zeros_like(theta)
      
      grads=torch.autograd.grad(batch_loss, self.model2.parameters(), retain_graph = True ,allow_unused=True)
      l=[]
      for i,(p,grad) in enumerate(zip(self.model2.parameters(),grads)):
          if grad is not None:
              l.append(grad)
          if grad is None:
              l.append(torch.autograd.Variable(torch.zeros(p.size()).type(torch.float32),requires_grad=True).to(device))
      dtheta = _concat(l).data + self.model2_wd*theta

      # convert to the model
      unrolled_model2 = self._construct_model2_from_theta(theta.sub(model2_lr, moment+dtheta))

      return unrolled_model2

  def _hessian_vector_product_A(self, vector, train_inputs, idxs, r = 1e-2):
    R = r / _concat(vector).norm()
    for p, v in zip(self.model1.parameters(), vector):
      if v is None:
          #.cuda()
          v=torch.autograd.Variable(torch.zeros(p.size()).type(torch.float32),requires_grad=True).to(device).data
      p.data.add_(R, v)
    loss = loss1(train_inputs, self.model1, idxs, self.A,  self.batch_size, self.vocab)
    #print('loss:', loss)
    grads_p = torch.autograd.grad(loss, self.A.parameters())
    #print('grads p:', grads_p)
  

    for p, v in zip(self.model1.parameters(), vector):
      if v is None:
          #.cuda()
          v=torch.autograd.Variable(torch.zeros(p.size()).type(torch.float32),requires_grad=True).to(device).data
      p.data.sub_(2*R, v)
    loss = loss1(train_inputs, self.model1, idxs, self.A,  self.batch_size, self.vocab)
    grads_n = torch.autograd.grad(loss, self.A.parameters())
    #print('grads n:', grads_n)
   

    for p, v in zip(self.model1.parameters(), vector):
      if v is None:
          #.cuda()
          v=torch.autograd.Variable(torch.zeros(p.size()).type(torch.float32),requires_grad=True).to(device).data
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]




  def _outer_A(self, vector_s_dash, train_inputs, un_inputs, idxs, unrolled_model1, unrolled_model2, model1_lr, model2_lr, r = 1e-2):
    R1 = r / _concat(vector_s_dash).norm()

    # plus W
    for p, v in zip(self.model2.parameters(), vector_s_dash):
        p.data.add_(R1, v)

    self.model1.train()

    loss_aug_p = loss2(un_inputs, unrolled_model1, self.model2, self.batch_size, self.vocab)
    #print('loss aug p:', loss_aug_p)
    vector_dash = torch.autograd.grad(loss_aug_p, unrolled_model1.parameters(), retain_graph = True, allow_unused=True)
    #print('vector dash:', vector_dash)

    grad_part1 = self._hessian_vector_product_A(vector_dash, train_inputs, idxs)
    #print('grad_part1:', grad_part1)

    # minus W
    for p, v in zip(self.model2.parameters(), vector_s_dash):
        p.data.sub_(2*R1, v)

    loss_aug_m = loss2(un_inputs, unrolled_model1, self.model2,self.batch_size, self.vocab)
    #print('loss aug m:', loss_aug_m)
    vector_dash = torch.autograd.grad(loss_aug_m, unrolled_model1.parameters(), retain_graph = True, allow_unused=True)
    grad_part2 = self._hessian_vector_product_A(vector_dash, train_inputs, idxs)
    #print('grad_part2:', grad_part2)

    for p, v in zip(self.model2.parameters(), vector_s_dash):
      p.data.add_(R1, v)

    grad = [(x-y).div_((2*R1)/(model1_lr*model2_lr)) for x, y in zip(grad_part1, grad_part2)]

    return grad




      
  def step(self, train_inputs, un_inputs, val_inputs, model1_lr, A, idxs, criterion, model2_lr, model2_optim, model1_optim):
      self.A_optim.zero_grad()
      # 1st step
      unrolled_model1 = self._compute_unrolled_enc_dec_model(train_inputs, model1_lr, idxs, model1_optim)
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      unrolled_model1.to(device)
      unrolled_model1.train()
      
      # 2nd step
      unrolled_model2 = self._compute_unrolled_model2(un_inputs, unrolled_model1, idxs, model2_lr, model2_optim)
      print('computed unrolled model2')
      valid_batch_loss = 0
      #val batch inputs
      for i in range(self.batch_size):
        input_train = val_inputs[i][0]
        target_train = val_inputs[i][1]
       
        enc_hidden, enc_outputs = unrolled_model2.enc_forward(input_train)
        valid_loss = unrolled_model2.dec_forward(target_train, enc_hidden,enc_outputs) 
        #print('valid loss:', valid_loss)
        valid_batch_loss += valid_loss
      valid_batch_loss = valid_batch_loss/self.batch_size
      unrolled_model2.train()
      valid_batch_loss.backward()

      l=[]
      for p in unrolled_model2.parameters():
        grad=p.grad
        if grad is not None:
            l.append(grad.data)
        if grad is None:
            l.append(torch.autograd.Variable(torch.zeros(p.size()).type(torch.float32),requires_grad=True).to(device).data)

      vector_s_dash = l
      #update A
      implicit_grads_A = self._outer_A(vector_s_dash, train_inputs, un_inputs, idxs, unrolled_model1,
            unrolled_model2, model1_lr, model2_lr)
      print('A grads:', implicit_grads_A)

      for v, g in zip(self.A.parameters(), implicit_grads_A):
        if v.grad is None:
            v.grad = Variable(g.data)
        else:
            v.grad.data.copy_(g.data)
      print('before A:', self.A)
      self.A_optim.step()
      print('after A:', self.A)

      del unrolled_model1

      del unrolled_model2

      gc.collect()
      return valid_batch_loss


import torch
'''
Implements loss functions for step 1,2 and 3 of the pipeline.
compute_loss1-> calculates batch loss for step1 by multiplying A
compute_loss2-> calculates batch loss for step 2 and 3 
'''
#computes loss for step 1
def compute_loss1(predictions, targets, a, device, criterion):

    predictions = predictions[:, :-1, :].contiguous()
    targets = targets[:, 1:]
    batch_loss=torch.empty(targets.shape[0], device=device)
    for i in range(a.shape[0]):
      batch_loss[i]=criterion(predictions[i], targets[i])
    y=torch.zeros(1, device=device)
    y=(batch_loss*a).sum() #multiply by ai
    return y

#computes loss for step 2 and 3
def compute_loss2(predictions, targets, device, criterion):

    predictions = predictions[:, :-1, :].contiguous()
    targets = targets[:, 1:]
    
    batch_loss=torch.empty(targets.shape[0], device=device)
    for i in range(targets.shape[0]):
      batch_loss[i]=criterion(predictions[i], targets[i])
    y=torch.zeros(1, device=device)
    y=(batch_loss).sum()
    return y

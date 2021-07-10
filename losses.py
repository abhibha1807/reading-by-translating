import torch

#computes loss for step 1
def compute_loss1(predictions, targets, a, batch_size, criterion, device):

    predictions = predictions[:, :-1, :].contiguous()
    targets = targets[:, 1:]
    batch_loss=torch.empty(batch_size, device=device)
    for i in range(batch_size):
      batch_loss[i]=criterion(predictions[i], targets[i])
    y=torch.zeros(1, device=device)
    y=(batch_loss*a).sum() #multiply by ai
    return y

#computes loss for step 2 and 3
def compute_loss2(predictions, targets, acc, batch_size, criterion, device):

    predictions = predictions[:, :-1, :].contiguous()
    targets = targets[:, 1:]
    batch_loss=torch.empty(batch_size, device=device)
    for i in range(batch_size):
      batch_loss[i]=criterion(predictions[i], targets[i])
    y=torch.zeros(1, device=device)
    y=(batch_loss).sum()
    return y

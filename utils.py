import torch
from torch.utils.data import Dataset

# creates a dataloader for weight matrix A 
class createBatchesA(Dataset):
    def __init__(self, A):
        self.samples = A

    def __len__(self):
      return len(self.samples)

    def __getitem__(self, idx):
      return self.samples[idx]

def _concat(xs, device):
  p=[]
  for x in xs:
    p.append(x.view(-1).to(device))
  return (torch.cat(p).to(device))
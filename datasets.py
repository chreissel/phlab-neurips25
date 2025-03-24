import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

def pairwise_product_sum(x,normalize=True):
    if len(x.size()) == 2:
        x = x.unsqueeze(-1) # B, D, 1
    else:
        assert len(x.size()) == 3
    b,n,d = x.shape
    t1 = (x.sum(dim=1)**2).sum(dim=1) # sum(x_i)^2 
    t2 = (x**2).sum(dim=2).sum(dim=1) # sum(x_i^2)
    sums = 0.5 * (t1 - t2) # sum of all pairwise dot products
    if normalize:
        norm_dot = d
        norm_pairwise = 0.5 * (n**2 - n)
        sums = sums / (norm_dot * norm_pairwise)
    return sums

class PairwiseRandom:
    def __init__(self,dim,noise_dim,N):
        self.dim = dim
        self.noise_dim = noise_dim
        self.N = N
        self.data = torch.rand(N,dim+noise_dim)
        self.sums = pairwise_product_sum(self.data[:,:dim])
        self.labels = (self.sums > 0.25).float().reshape(-1,1)

    def loader(self,bs):
        dataset = TensorDataset(self.data, self.labels)
        loader = DataLoader(dataset, batch_size=bs, shuffle=True)
        return loader
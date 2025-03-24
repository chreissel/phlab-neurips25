import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCLRModel(nn.Module):
    def __init__(self, encoder, projector):
        super(SimCLRModel, self).__init__()
        self.encoder = encoder
        self.projector = projector
    
    def forward(self, x, embed=False):
        h = self.encoder(x)
        if embed:
            return h
        z = self.projector(h)
        return z
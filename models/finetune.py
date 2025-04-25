import torch
import torch.nn as nn

class FineTuner(nn.Module):
    def __init__(self,encoder,projector,corrector):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.corrector = corrector

    def forward(self, x):
        return self.encoder(x)
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from .losses import SupervisedSimCLRLoss

class SimCLRModel(pl.LightningModule):
    def __init__(self, encoder, projector, temperature=0.1):
        super(SimCLRModel, self).__init__()
        self.encoder = encoder
        self.projector = projector
        self.criterion = SupervisedSimCLRLoss(temperature=temperature)
        self.save_hyperparameters()

    def embed(self,x):
        return self.encoder(x)
    
    def project(self,h):
        return self.projector(h)
    
    def forward(self, x, embed=False):
        z = self.embed(x)
        if embed:
            return z
        h = self.project(z)
        return h
    
    def training_step(self, batch, batch_idx):
        x, labels = batch
        aug0, aug1 = x
        
        h0 = self.encoder(aug0)
        z0 = F.normalize(self.projector(h0),dim=1)
        h1 = self.encoder(aug1)
        z1 = F.normalize(self.projector(h1),dim=1)

        features = torch.cat([z0.unsqueeze(1), z1.unsqueeze(1)], dim=1)
        loss = self.criterion(features, labels=None) # no labels in aug-based simclr

        self.log("train/loss",
                 loss,
                 on_step=True,
                 on_epoch=True,
                 reduce_fx='mean',
                 logger=True,
                 prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, labels = batch
        aug0, aug1 = x
        
        h0 = self.encoder(aug0)
        z0 = F.normalize(self.projector(h0),dim=1)
        h1 = self.encoder(aug1)
        z1 = F.normalize(self.projector(h1),dim=1)

        features = torch.cat([z0.unsqueeze(1), z1.unsqueeze(1)], dim=1)
        loss = self.criterion(features, labels=None) # no labels in aug-based simclr

        self.log("val/loss",
                 loss,
                 on_step=False,
                 on_epoch=True,
                 reduce_fx='mean',
                 logger=True,
                 prog_bar=True)

        return loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset, ConcatDataset
import numpy as np
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm
import torchvision.transforms.v2 as v2
from copy import deepcopy
import sys
sys.path.append("/n/home11/sambt/phlab-neurips25")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from models.litmodels import SimCLRModel
from models.networks import CustomResNet, MLP
from data.datasets import CIFAR10Dataset
from data.cifar import CIFAR5MDataset
import data.data_utils as dutils

from sklearn.metrics import roc_auc_score, top_k_accuracy_score
from utils.plotting import make_corner

class HueTransform(nn.Module):
    def __init__(self,factor):
        super().__init__()
        self.factor = factor
    def forward(self,x):
        return v2.functional.adjust_hue(x,self.factor)
    
CHECKPOINT = '/n/home11/sambt/phlab-neurips25/runs/cifar10_simCLR_ResNet50_T0.5/lightning_logs/qdcc0he3/checkpoints/epoch=15-step=1408.ckpt'

model = SimCLRModel.load_from_checkpoint(CHECKPOINT)
model = model.to(device)
model = model.eval()

classifier = model.classifier

for i in range(6):
    cifar5m_embeds = []
    cifar5m_labels = []
    cifar5m_preds = []
    
    hue_transform = v2.ColorJitter(hue=(-0.2,-0.2))
    cifar5m_full = CIFAR5MDataset("resnet50",[i],[(None,None)],exclude_classes=[9],
                                  grayscale=False,custom_pre_transforms=[hue_transform])
    for ims,labs in tqdm(DataLoader(cifar5m_full,batch_size=512,shuffle=False)):
        with torch.no_grad():
            cifar5m_embeds.append(model.encoder(ims.to(device)).cpu().numpy())
            cifar5m_labels.append(labs.numpy())
            cifar5m_preds.append(classifier(torch.tensor(cifar5m_embeds[-1]).to(device)).cpu().numpy())
        
    cifar5m_embeds = np.concatenate(cifar5m_embeds)
    cifar5m_labels = np.concatenate(cifar5m_labels)
    cifar5m_preds = np.concatenate(cifar5m_preds)
    np.save(f"cifar5m_classifierPreds_hueJitterNeg0p1_file{i}.npy",cifar5m_preds)
    np.save(f"cifar5m_embeds_hueJitterNeg0p1_file{i}.npy",cifar5m_embeds)
    np.save(f"cifar5m_labels_hueJitterNeg0p1_file{i}.npy",cifar5m_labels)

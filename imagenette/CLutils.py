import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.models as models

# Define a custom dataset
class DatasetCorruptedMNIST(Dataset):
    def __init__(self, data, labels):
        self.data = torch.from_numpy(data/255.).float()
        self.data = torch.moveaxis(self.data, 3, 1)
        self.labels = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class MNISTembedder(nn.Module):
    def __init__(self, latent_dim=4, device="cpu", input_channels=1):
        super(MNISTembedder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels,  20, 4)# 1 input channels, 10 output channels, 4x4 kernel
        self.conv2 = nn.Conv2d(20, 10, 4)
        self.conv3 = nn.Conv2d(10, 5, 4)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1445, 500)
        self.fc2 = nn.Linear(500, 64)
        self.fc3 = nn.Linear(64, latent_dim)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = torch.flatten(x, start_dim=1, end_dim=3) # flatten all dimensions except batch
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


#class ImageNETTEembedder(torch.nn.Module):
#    def __init__(self, latent_dim):
#        super(ImageNETTEembedder, self).__init__()
def ImageNETTEembedder(latent_dim):        
    # Load a pretrained ResNet50 model
    model = models.resnet50(pretrained=True)

    # Modify the final layer to fit the number of ImageNet classes (1000 classes)
    num_ftrs = model.fc.in_features

    # Freeze layers except for the last few layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last few layers for fine-tuning
    #for param in model.layer4.parameters():
    #    param.requires_grad = True

    # Use custom fully connected layers
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(num_ftrs, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        #torch.nn.Linear(512, 256),
        #torch.nn.ReLU(),
        #torch.nn.Dropout(0.5),
        torch.nn.Linear(256, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(128, latent_dim)  # Adjust for the number of classes
    )
    return model


# Simplified implemented from https://github.com/violatingcp/codec/blob/main/losses.py
class SimCLRLoss(nn.Module):
    """
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
        logits_mask = torch.logical_not(mask).float()

        logits = torch.div(
            torch.matmul(features, features.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        exp_logits += torch.exp(logits) * mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - self.temperature * mean_log_prob_pos
        loss = loss.view(1, batch_size).float().mean()
        return loss

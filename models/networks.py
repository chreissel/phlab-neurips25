import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet18, efficientnet_b0 
from .resnet_wider import resnet50x1, resnet50x2, resnet50x4

activations = {
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
}

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.0, internal_activation='relu', output_activation=None, input_activation=None):
        super().__init__()
        layers = []
        if input_activation is not None:
            layers.append(input_activation())
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(activations[internal_activation])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        if output_activation is not None:
            layers.append(activations[output_activation])
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
class DeepSetsEncoder(nn.Module):
    def __init__(self, phi, f):
        super().__init__()
        self.phi = phi
        self.f = f

    def forward(self, x):  # x shape: (batch_size, N, D) # N = number of particle, D = dimension of input
        x = self.phi(x).sum(dim=-2) # sum over "particles"
        return self.f(x)
    
class CustomResNet(nn.Module):
    def __init__(self,variant,fc_hidden,fc_out,**kwargs):
        # loads resnet50 then replaces the last fc layer with a user-specificed mlp
        # fc_dims specifies the dimensions 
        super().__init__()

        if variant == 'resnet50':
            self.model = resnet50()
        elif variant == 'resnet18':
            self.model = resnet18()
        else:
            print(f"Variant {variant} not recognized. Using resnet50")
            self.model = resnet50()

        dim_resnet = self.model.fc.in_features
        self.model.fc = MLP(dim_resnet,fc_hidden,fc_out,**kwargs)

    def forward(self,x):
        return self.model(x)
    
class CustomPretrainedResNet(nn.Module):
    def __init__(self,variant,fc_hidden,fc_out,**kwargs):
        # loads resnet50 then replaces the last fc layer with a user-specificed mlp
        # fc_dims specifies the dimensions 
        super().__init__()

        if variant == 'resnet50x1':
            weights = torch.load("/n/holystore01/LABS/iaifi_lab/Lab/sambt/neurips25/simclr_pretrained/resnet50-1x.pth")['state_dict']
            self.model = resnet50x1()
            self.model.load_state_dict(weights)
        elif variant == 'resnet50x2':
            weights = torch.load("/n/holystore01/LABS/iaifi_lab/Lab/sambt/neurips25/simclr_pretrained/resnet50-2x.pth")['state_dict']
            self.model = resnet50x2()
            self.model.load_state_dict(weights)
        elif variant == 'resnet50x4':
            weights = torch.load("/n/holystore01/LABS/iaifi_lab/Lab/sambt/neurips25/simclr_pretrained/resnet50-4x.pth")['state_dict']
            self.model = resnet50x4()
            self.model.load_state_dict(weights)
        else:
            print(f"Variant {variant} not recognized. Using resnet50x1")
            weights = torch.load("/n/holystore01/LABS/iaifi_lab/Lab/sambt/neurips25/simclr_pretrained/resnet50-1x.pth")['state_dict']
            self.model = resnet50x1()
            self.model.load_state_dict(weights)

        for param in self.model.parameters():
            param.requires_grad = False
        
        dim_resnet = self.model.fc.out_features
        self.output_mlp = MLP(dim_resnet,fc_hidden,fc_out,**kwargs)

    def forward(self,x):
        return self.output_mlp(self.model(x))
    

class CustomEfficientNet(nn.Module):
    def __init__(self,fc_hidden,fc_out,**kwargs):
        # loads efficientnet_b0 then replaces the last fc layer with a user-specificed mlp
        # fc_dims specifies the dimensions 
        super().__init__()

        self.model = efficientnet_b0()
        
        dim_resnet = self.model.classifier[1].in_features
        self.model.classifier = MLP(dim_resnet,fc_hidden,fc_out,**kwargs)

    def forward(self,x):
        return self.model(x)
    


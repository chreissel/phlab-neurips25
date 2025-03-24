import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.0, internal_activation=nn.ReLU, output_activation=None, input_activation=None):
        super().__init__()
        layers = []
        if input_activation is not None:
            layers.append(input_activation())
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(internal_activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        if output_activation is not None:
            layers.append(output_activation())
        
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
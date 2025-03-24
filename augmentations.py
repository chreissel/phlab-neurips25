import torch
import torch.nn as nn

def permute_vector_elements(x):
    assert len(x.size()) == 2
    randperm = torch.argsort(torch.rand(*x.shape),dim=1)
    return x[torch.arange(x.size(0)).unsqueeze(-1), randperm]
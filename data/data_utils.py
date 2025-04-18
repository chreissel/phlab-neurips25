import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.datasets import VisionDataset


class viewGenerator:
    """
        This class is used to generate multiple views of the same data point using `transform`.
        This is useful for SimCLR style training where we want to use multiple views of the same data point.
        Intended to be passed as the `transform` argument to a PyTorch DataLoader (e.g. torchvision datasets).
    """
    def __init__(self,transform,n_views):
        self.transform = transform
        self.n_views = n_views
    
    def __call__(self,x):
        return [self.transform(x) for _ in range(self.n_views)]
    
class AugmentationDataset(Dataset):
    """
        A dataset that generates multiple views of the same data point using `viewGenerator` and a `transform`
    """
    def __init__(self,base_dataset,view_generator):
        super().__init__()
        self.base_dataset = base_dataset
        self.view_generator = view_generator

    def __getitem__(self, index):
        """
        Assuming that dataset is spitting out something of the form `(batch,labels)` and we want to
        generate views of batch. TODO: Generalize this to handle arbitrary data formats.
        """
        data = self.base_dataset[index]
        return self.view_generator(data[0]), *data[1:]

### Stuff for pairwise product sum toy dataset ###
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

class permute_dims:
    """
        Randomly permute the first `dim` dimensions of x.
    """
    def __init__(self,dim):
        self.dim = dim
    
    def __call__(self,x):
        # assume x is a tensor of shape D where D is the full dimensionality
        randperm = torch.argsort(torch.rand(self.dim))
        aug = x.clone()
        aug[:self.dim] = aug[randperm]
        return aug
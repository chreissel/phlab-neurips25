import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, IterableDataset, ConcatDataset
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode
import torchvision.transforms.v2 as v2
import numpy as np

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
    
class MultiIter:
    def __init__(self,iterators,fractions):
        self.iterators = iterators
        self.fractions = fractions

    def __next__(self):
        i = np.random.choice(len(self.iterators), p=self.fractions)
        return next(self.iterators[i])

class InterleavedIterableDataset(IterableDataset):
    def __init__(self,datasets,fractions):
        """
        Datasets is a list of datasets to interleave. Fractions is a list of fractions for each dataset.
        The fractions should sum to 1.0.
        """
        assert len(datasets) == len(fractions)
        self.datasets = datasets
        self.fractions = fractions
        self.iters = [iter(d) for d in self.datasets]

    def __iter__(self):
        return MultiIter(self.iters,self.fractions)

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
    
def ResNet50Transform(resnet_type,grayscale=False,from_pil=True,custom_pre_transforms=None,custom_post_transforms=None):
        assert resnet_type in ['resnet18','resnet50']
        if resnet_type == 'resnet50':
            resize_size = 232
            crop_size = 224
        elif resnet_type == 'resnet18':
            resize_size = 256
            crop_size = 224
        else:
            print("resnet type not recognized, using resnet18 values")
            resize_size = 232
            crop_size = 224
        
        transforms = []

        if from_pil:
            transforms.append(v2.PILToTensor()) # CIFAR10 is stored as PIL; cifar5m is not
        
        if custom_pre_transforms is not None: # transforms applied to smaller cifar images before resizing/interpolation
            assert type(custom_pre_transforms) == list
            for t in custom_pre_transforms:
                transforms.append(t)
            
        if grayscale:
            transforms.append(v2.Grayscale(num_output_channels=3))
        
        transforms.append(v2.Resize(resize_size,interpolation=InterpolationMode.BILINEAR,antialias=True)) # standard resnet preprocessing
        transforms.append(v2.CenterCrop(crop_size)) # standard resnet preprocessing

        if custom_post_transforms is not None:
            assert type(custom_post_transforms) == list
            for t in custom_post_transforms:
                transforms.append(t)

        transforms.append(v2.ToDtype(torch.float32,scale=True)) # standard resnet preprocessing
        transforms.append(v2.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])) # standard resnet normalization

        return v2.Compose(transforms)

class TransformDataset(Dataset):
    def __init__(self,transform,data,labels):
        super().__init__()
        self.transform = transform
        self.data = data
        self.labels = labels
    
    def __getitem__(self, index):
        return self.transform(self.data[index]), self.labels[index]
    
    def __len__(self):
        return len(self.data)
    
    def subset(self,a,b):
        return TransformDataset(self.transform,self.data[slice(a,b)],self.labels[slice(a,b)])
    
    def random_split(self,fraction):
        N = len(self.data)
        indices = np.arange(N)
        np.random.shuffle(indices)
        split = int(fraction*N)
        i1, i2 = indices[:split], indices[split:]
        return TransformDataset(self.transform,self.data[i1],self.labels[i1]), \
               TransformDataset(self.transform,self.data[i2],self.labels[i2])
    
    def subselection(self,selection):
        return TransformDataset(self.transform,self.data[torch.tensor(selection)],self.labels[np.array(selection)])
    
class ConcatWithLabels(Dataset):
    def __init__(self, datasets,labels):
        assert len(datasets) == len(labels)
        self._datasets = datasets
        self._labels = [labels[i]*torch.ones(len(datasets[i])) for i in range(len(datasets))]
        self._len = sum(len(dataset) for dataset in datasets)
        self._indexes = []

        # Calculate distribution of indexes in all datasets
        cumulative_index = 0
        for idx, dataset in enumerate(datasets):
            next_cumulative_index = cumulative_index + len(dataset)
            self._indexes.append((cumulative_index, next_cumulative_index, idx))
            cumulative_index = next_cumulative_index

    def __getitem__(self, index):
        for start, stop, dataset_index in self._indexes:
            if start <= index < stop:
                dataset = self._datasets[dataset_index]
                return dataset[index - start], self._labels[dataset_index][index - start]

    def __len__(self) -> int:
        return self._len
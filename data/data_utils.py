import torch 
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.networks import MLP
from torch.utils.data import Dataset, DataLoader, TensorDataset, IterableDataset, ConcatDataset
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode
import torchvision.transforms.v2 as v2
import numpy as np
from utils.MAHALANOBISutils import compute_empirical_means,compute_empirical_cov_matrix,mahalanobis_test
from utils.ANALYSISutils import plot_2distribution_new

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

    def __len__(self):
        return len(self.base_dataset)

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


class rotate:
    """
        Randomly permute the first `dim` dimensions of x.
    """
    def __call__(self,x):
        # assume x is a tensor of shape D where D is the full dimensionality
        rot = 2.*3.141592741012*torch.rand(1)#x.shape[0]//3)
        x=x.reshape(4,3)
        aug  = self.rotateTheta(x,rot)
        return aug

    def theta_to_eta(self,theta):
        pTheta =  torch.where(theta > np.pi,2*np.pi - theta, theta)            
        return -np.log(np.tan(pTheta/2))

    def eta_to_theta(self,eta):
        return 2 * torch.atan(torch.exp(-eta))

    def convert_to_cart(self, vec):
        px = vec[:,0] * torch.cos(vec[:,2])
        py = vec[:,0] * torch.sin(vec[:,2])
        pz = vec[:,0] * torch.sinh(vec[:,1])
        return torch.stack((px,py,pz)).T

    def convert_to_phys(self,vec):
        pt = torch.sqrt(vec[:, 0]**2 + vec[:, 1]**2)
        phi = torch.atan2(vec[:, 1], vec[:, 0])    
        # Avoid division by zero
        eta = torch.where(pt != 0, torch.asinh(vec[:, 2] / pt), torch.sign(vec[:, 2]) * float('inf'))
        return torch.stack((pt,eta,phi)).T

    def rotateTheta(self,idau,itheta):
        v1  =self.convert_to_cart(idau)
        axis=torch.tensor([1.,0.,0.])
        axisr=axis.repeat((v1.shape[0])).reshape(v1.shape)
        rotmat=self.rotation_matrix_3d(axisr,itheta).float()
        v1  = v1.unsqueeze(2).float()
        #v1  = v1.reshape(v1.shape[0],v1.shape[1],1).float()
        v1rot=torch.matmul(rotmat, v1)
        v1rot=v1rot.squeeze(2)
        #v1rot=v1.reshape(v1.shape[0],v1.shape[1])
        v1rot=self.convert_to_phys(v1rot)
        return v1rot

    def rotation_matrix(self,axis, theta):
        """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        #axis = np.asarray(axis)
        axis = axis / torch.sqrt(torch.dot(torch.tensor(axis),torch.tensor(axis)))
        a = torch.cos(theta / 2.0)
        print("a:",a)
        b, c, d = -axis * torch.sin(theta / 2.0)
        print("b:",b,"c:",c,"d:",d)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return torch.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])



    def rotation_matrix_3d(self, axis, angle):
        axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True)
        if not isinstance(angle, torch.Tensor):
            angle = torch.tensor(float(angle))
    
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        # Ensure correct dimensions for broadcasting
        x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
        if axis.ndim == 1: #single rotation
            rot_matrix = torch.tensor([
                [cos_a + x**2 * (1 - cos_a), x * y * (1 - cos_a) - z * sin_a, x * z * (1 - cos_a) + y * sin_a],
                [y * x * (1 - cos_a) + z * sin_a, cos_a + y**2 * (1 - cos_a), y * z * (1 - cos_a) - x * sin_a],
                [z * x * (1 - cos_a) - y * sin_a, z * y * (1 - cos_a) + x * sin_a, cos_a + z**2 * (1 - cos_a)]
            ])
    
        elif axis.ndim == 2: #batched rotation
            rot_matrix = torch.stack([
                torch.stack([cos_a + x**2 * (1 - cos_a), x * y * (1 - cos_a) - z * sin_a, x * z * (1 - cos_a) + y * sin_a], dim=-1),
                torch.stack([y * x * (1 - cos_a) + z * sin_a, cos_a + y**2 * (1 - cos_a), y * z * (1 - cos_a) - x * sin_a], dim=-1),
                torch.stack([z * x * (1 - cos_a) - y * sin_a, z * y * (1 - cos_a) + x * sin_a, cos_a + z**2 * (1 - cos_a)], dim=-1)
            ], dim=-2)
        
        else:
            raise ValueError("Axis must be a 1D or 2D tensor")
        return rot_matrix


class smear:
    def __call__(self,x):
        # assume x is a tensor of shape D where D is the full dimensionality
        aug=x.reshape(4,3)
        aug[:,1] += (torch.randn(4)*0.1/aug[:,0])
        aug[:,2] += (torch.randn(4)*0.1/aug[:,0])
        return aug.flatten().float()

class smearAndRotate:
#    def __init__(self):
#        self.sme = smear()
#        self.rot = rotate()

    def __call__(self,x):
        #return smear()(x)
        return rotate()(x)
        if torch.rand(1) > 0.5:
            #t = smear()
            return smear()(x)
        else:
            #t = rotate()
            return rotate()(x)
        #return t(x).flatten().float()
    
class GenericDataset(Dataset):
    def __init__(self,data,labels,normalize=False):
        self.data   = data
        self.labels = labels
    
    def generate_augmentation(self,batch):
        return None
    
    def normalize(self,batch):
        return (batch - self.mean.to(batch)) / self.std.to(batch)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self,idx):
        return self.data[idx], self.labels[idx]

def train_generic(inepochs,itrainloader,imodel,icriterion,ioptimizer):
    losses = []
    for epoch in tqdm(range(inepochs)):
        imodel.train()
        epoch_loss = []
        for batch_data, labels in itrainloader:
            batch_data = batch_data.float()

            # Potential to add any augmentation here
            features = imodel(batch_data).unsqueeze(1)
        
            # Compute SimCLR loss
            loss = icriterion(features,labels=labels)
        
            # Backward pass and optimization
            ioptimizer.zero_grad()
            loss.backward()
            ioptimizer.step()
        
            epoch_loss.append(loss.item())
        mean_loss = np.mean(epoch_loss)
        losses.append(mean_loss)
        #if epoch % 1 == 0:
        #print(f'Epoch [{epoch+1}/{inepochs}], Loss: {mean_loss:.4f}')
    
    plt.figure(figsize=(8,6))
    plt.plot(np.arange(len(losses)),losses)

def train_disc(inepochs,itrain,input_dim,last_dim=16,output_dim=3):
    num_epochs=inepochs
    hidden_dims= [64,128,32,last_dim]
    disc_criterion = nn.CrossEntropyLoss()
    disc_model     = MLP(input_dim=input_dim,hidden_dims=hidden_dims,output_dim=output_dim,output_activation="sigmoid",dropout=0.)#.to(device)
    disc_optimizer = torch.optim.AdamW(disc_model.parameters(), lr=0.5e-2)
    losses = []
    for epoch in tqdm(range(num_epochs)):
        disc_model.train()
        epoch_loss = []
        for batch_data, labels in itrain:
            batch_data = batch_data.float()
            features = disc_model(batch_data)
            if output_dim == 1:
                features=features.squeeze(1)
                loss = disc_criterion(features,labels)
            else:
                loss = disc_criterion(features,labels.long())
            disc_optimizer.zero_grad()
            loss.backward()
            disc_optimizer.step()        
            epoch_loss.append(loss.item())
        mean_loss = np.mean(epoch_loss)
        losses.append(mean_loss)
    #if epoch % 10 == 0:
    #    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {mean_loss:.4f}')
    
    plt.figure(figsize=(8,6))
    plt.plot(np.arange(len(losses)),losses)
    plt.yscale('log')
    return disc_model
        
#DE-SC0021943 #ECA
#DE-SC001193 #Extra

from torchmetrics import Accuracy,AUROC

def check_disc(itest_data,itest_labels,imodel):
    test_accuracy = Accuracy(task="binary", num_classes=2)#,top_k=2)
    labels=itest_labels.int()
    with torch.no_grad():
        output = (imodel(itest_data.float()))
    print(output.shape[1])
    if output.shape[1] > 1:
        vars = output[:,0]/(output[:,0]+output[:,1])
        print("Accuracy:",test_accuracy(vars[labels < 2],labels[labels < 2]))
    else:
        vars = output.flatten()
        print(vars.shape)
        print("Accuracy:",test_accuracy(vars[labels!=1],labels[labels!=1]//2))

    #plt.plot(output[labels==0][:,0],output[labels==0][:,1],'.',alpha=0.5)
    #plt.plot(output[labels==1][:,0],output[labels==1][:,1],'.',alpha=0.5)
    #plt.plot(output[labels==2][:,0],output[labels==2][:,1],'.',alpha=0.5)    
    plt.hist(vars[labels==0],alpha=0.5)
    plt.hist(vars[labels==1],alpha=0.5)
    #plt.hist(vars[labels==2],alpha=0.5)
    plt.show()

def approxDist(iData, iModel, iLabel, nsamps):
        dists=[]
        for pVal in iData:
                pDist=[]
                for pSamp in range(nsamps):
                    pModel = iModel[iLabel == pSamp]
                    ppDist=torch.sqrt(torch.sum((pModel-pVal)**2,axis=1))
                    pDist.append(ppDist.mean()/ppDist.std())
                dists.append(torch.min(torch.tensor(pDist)))
        return torch.tensor(dists)

from torchmetrics.classification import ROC
def approxAUC(dist, labels,nsamps):
    metric = AUROC(task="binary")
    auc_score = metric(dist, labels//(nsamps-1))
    #test_accuracy = Accuracy(task="binary", num_classes=2)
    print("AUC:",auc_score)
    #print("Acc:",test_accuracy(dist., labels//(nsamps-1)))
    roc = ROC(task="binary")
    roc.update(dist, labels//(nsamps-1))
    fpr, tpr, thresholds = roc.compute()
    plt.plot(fpr.cpu().numpy(), tpr.cpu().numpy(), color='darkorange', label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
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

#def approxDist(iData, iModel, iLabel, nsamps):


def mahalanobis_dist(data, ref, ref_label,plot=True):#, sig_label=-1, seed=0, n_ref=1e4, n_bkg=1e3, n_sig=1e2, z_ratio=0.1, anomaly_type ='', plot=True, pois_ON=False):
    '''
    - computes the mahalnobis test for the dataset 
    '''
    # random seed                                                                                                                    
    #np.random.seed(seed)
    #print('Random seed: '+str(seed))
    
    # train on GPU?                                                                                                                  
    cuda = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if cuda else "cpu")
    #data   = data.to(DEVICE)
    #model  = model.to(DEVICE)
    #label  = label.to(DEVICE)

    # estimate parameters of the bkg model 
    means=compute_empirical_means(ref,ref_label)
    emp_cov=compute_empirical_cov_matrix(ref, ref_label, means)
    M_data = mahalanobis_test(data, means, emp_cov)
    if plot:
        M_ref  = mahalanobis_test(ref, means, emp_cov)
        # visualize mahalanobis
        fig = plt.figure(figsize=(9,6))
        fig.patch.set_facecolor('white')
        ax= fig.add_axes([0.15, 0.1, 0.78, 0.8])
        plt.hist([M_ref, M_data], density=True, label=['REF', 'DATA'])
        #font = font_manager.FontProperties(family='serif', size=16)
        plt.legend(fontsize=18, ncol=2, loc='best')
        plt.yscale('log')
        #plt.yticks(fontsize=16, fontname='serif')
        #plt.xticks(fontsize=16, fontname='serif')
        plt.ylabel("density")#, fontsize=22, fontname='serif')
        plt.xlabel("mahalanobis metric")#, fontsize=22, fontname='serif')
        #plt.savefig(output_folder+'distribution.pdf')
        plt.show()

    # compute the test as the reduce sum of the mahalanobis distance over the dataset
    t = -1* torch.sum(M_data)
    #print('Mahalanobis test: ', "%f"%(t))
    return t

def run_toy( nsig, nbkg, nref, data, labels, model, model_labels,sig_idx,ntoys=100):
    t_sig = []
    t_ref = []
    refs      = model       [model_labels != sig_idx]
    refs_label= model_labels[model_labels != sig_idx]
    sigs     = data[labels == sig_idx]
    bkgs     = data[labels != sig_idx]

    ntotsig = len(sigs)
    ntotbkg = len(bkgs)
    ntotref = len(refs)
    
    nsigs   = np.random.poisson(lam=nsig, size=ntoys)
    nbkgs   = np.random.poisson(lam=nbkg, size=ntoys)
    nrefs   = np.random.poisson(lam=nref, size=ntoys)
    nbrfs   = np.random.poisson(lam=nbkg, size=ntoys)
    for pToy in range(ntoys):
        sigidx  = np.random.choice(ntotsig, size=nsigs[pToy], replace=True)
        bkgidx  = np.random.choice(ntotbkg, size=nbkgs[pToy], replace=True)
        refidx  = np.random.choice(ntotref, size=nrefs[pToy], replace=True)
        brfidx  = np.random.choice(ntotbkg, size=nbkgs[pToy], replace=True) #note to be accurate thsi should be ref, but statisically correct is bkg (its just cheating)
        sig     = sigs[sigidx]
        bkg     = bkgs[bkgidx]
        ref     = refs[refidx]
        brf     = bkgs[brfidx] # in the long run we change this to ref
        ref_label=refs_label[refidx]
        #for pMetric in metrics: #just one for now, otherwise t_sig/t_ref have to be fixed
        dist    = mahalanobis_dist(torch.cat((sig,bkg)),ref,ref_label,plot=False)
        ref_dist= mahalanobis_dist(brf,ref,ref_label,plot=False)
        t_sig.append(dist)
        t_ref.append(ref_dist)

    ts, tr = np.array(t_sig), np.array(t_ref)
    z_as, z_emp = plot_2distribution_new(ts, tr, df=np.median(ts), xmin=np.min(tr)-10, xmax=np.max(tr)+10, #ymax=0.03, 
                       nbins=8, save=False, output_path='./', Z_print=[1.645,2.33],
                       label1='REF', label2='DATA', save_name='', print_Zscore=True)
    return z_as,z_emp

#from GENutils import *
#from ANALYSISutils import *

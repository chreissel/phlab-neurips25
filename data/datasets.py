import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import lightning as pl
from . import data_utils as dutils
from . import toy4vec as toy4vec
from torchvision.transforms import v2
from torchvision.datasets import Imagenette
import numpy as np
from torchvision.datasets import Imagenette, CIFAR10
from torchvision.models import ResNet50_Weights, ResNet18_Weights
from .customImagenette import TensorImagenette
import glob
from .jetclass.dataset import SimpleIterDataset

class GenericDataModule(pl.LightningDataModule):
    def __init__(self,batch_size=512,num_workers=4,pin_memory=False):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.loader_kwargs = {"batch_size":self.batch_size,
                              "num_workers":self.num_workers,
                              "pin_memory":self.pin_memory}
    
class PairwiseSumDataset(GenericDataModule):
    def __init__(self,dim,noise_dim,
                 num_train,num_val,num_test,
                 **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.noise_dim = noise_dim
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test

        self.view_generator = dutils.viewGenerator(dutils.permute_dims(dim),2)

        self.train_data, self.train_labels = self.generate_data(self.num_train)
        self.train_dataset = dutils.AugmentationDataset(TensorDataset(self.train_data, self.train_labels),self.view_generator)

        self.val_data, self.val_labels = self.generate_data(self.num_val)
        self.val_dataset = dutils.AugmentationDataset(TensorDataset(self.val_data, self.val_labels),self.view_generator)

        self.test_data, self.test_labels = self.generate_data(self.num_test)
        self.test_dataset = TensorDataset(self.test_data, self.test_labels)


    def generate_data(self,N):
        data = torch.rand(N,self.dim+self.noise_dim)
        sums = dutils.pairwise_product_sum(data[:,:self.dim])
        labels = (sums > 0.25).float().reshape(-1,1)
        return data,labels
    
    def train_dataloader(self):
        loader = DataLoader(self.train_dataset, shuffle=True, **self.loader_kwargs)
        return loader
    
    def val_dataloader(self):
        loader = DataLoader(self.val_dataset, shuffle=True, **self.loader_kwargs)
        return loader
    
    def test_dataloader(self):
        loader = DataLoader(self.test_dataset, shuffle=False, **self.loader_kwargs)
        return loader
    
class ImagenetteDataset(GenericDataModule):
    def __init__(self,image_width,sup_simclr=False,**kwargs):
        super().__init__(**kwargs)
        
        if sup_simclr:
            self.simclr_augment = v2.Compose([
                v2.PILToTensor(), # operations are more efficient on tensors
                #v2.RandomResizedCrop(image_width),
                v2.Resize(256),
                v2.CenterCrop(image_width),
                v2.ToDtype(torch.float32,scale=True)
            ])
            self.simclr_views = self.simclr_augment
        else:
            # augmentations from original simCLR paper on ImageNet
            self.simclr_augment = v2.Compose([
                v2.PILToTensor(), # operations are more efficient on tensors
                v2.RandomResizedCrop(image_width),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomApply([v2.ColorJitter(0.8,0.8,0.8,0.2)],p=0.8),
                v2.RandomGrayscale(p=0.2),
                v2.RandomApply([v2.GaussianBlur(kernel_size=23)],p=0.5),
                v2.ToDtype(torch.float32,scale=True)
            ])
            # view generator for getting two augmentations per image
            self.simclr_views = dutils.viewGenerator(self.simclr_augment,2)

        # augmentations for ImageNet test evaluation - just resize and crop
        self.test_augment = v2.Compose([v2.PILToTensor(),
                                        v2.Resize(256),
                                        v2.CenterCrop(image_width),
                                        v2.ToDtype(torch.float32,scale=True)
                                        ])
        
        # Imagenette datasets
        self.train_dataset = Imagenette(root="/n/holystore01/LABS/iaifi_lab/Lab/sambt/neurips25/imagenette/",
                           split='train',
                           size='full',
                           download=False,
                           transform=self.simclr_views)
        self.val_dataset = Imagenette(root="/n/holystore01/LABS/iaifi_lab/Lab/sambt/neurips25/imagenette/",
                                split='val',
                                size='full',
                                download=False,
                                transform=self.simclr_views)
        self.test_dataset = Imagenette(root="/n/holystore01/LABS/iaifi_lab/Lab/sambt/neurips25/imagenette/",
                                split='val',
                                size='full',
                                download=False,
                                transform=self.test_augment)
        
    def train_dataloader(self):
        loader = DataLoader(self.train_dataset, shuffle=True, **self.loader_kwargs)
        return loader
    
    def val_dataloader(self):
        loader = DataLoader(self.val_dataset, shuffle=True, **self.loader_kwargs)
        return loader
    
    def test_dataloader(self):
        loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                            pin_memory=self.pin_memory, num_workers=self.num_workers)
        return loader

class ToyJetDataset(GenericDataModule):
    def __init__(self,npart,num_train,num_val,num_test,nrand=16,
                 **kwargs):
        super().__init__(**kwargs)
        self.npart = npart
        self.nrand = nrand
        self.num_train = num_train
        self.num_val   = num_val
        self.num_test  = num_test
        self.jdgs     = toy4vec.jet_data_generator("signal",npart, npart, True,nrandparticle=nrand)
        self.jdgb     = toy4vec.jet_data_generator("background",npart, npart, True,nrandparticle=nrand)
        self.jdgd     = toy4vec.jet_data_generator("signal_data",npart, npart, True,nrandparticle=nrand)
        
        self.view_generator = dutils.viewGenerator(dutils.smearAndRotate(),2)
        self.train_data, self.train_labels = self.generate_mc(self.num_train)
        self.train_dataset = dutils.AugmentationDataset(TensorDataset(self.train_data, self.train_labels),self.view_generator)
        self.train_dataset_basic = dutils.GenericDataset(self.train_data, self.train_labels)
        
        self.val_data, self.val_labels = self.generate_mc(self.num_val)
        self.val_dataset = dutils.AugmentationDataset(TensorDataset(self.val_data, self.val_labels),self.view_generator)
        self.val_dataset_basic = dutils.GenericDataset(self.train_data, self.train_labels)

        self.test_data, self.test_labels = self.generate_mc(self.num_test)
        self.test_dataset = TensorDataset(self.test_data, self.test_labels)
        self.test_dataset_basic = dutils.GenericDataset(self.train_data, self.train_labels)

        self.true_data, self.true_labels = self.generate_data(self.num_test)
        self.true_dataset = TensorDataset(self.true_data, self.true_labels)
        self.true_dataset_basic = dutils.GenericDataset(self.true_data, self.true_labels)

        self.trut_data, self.trut_labels = self.generate_data(self.num_test)
        self.trut_dataset = TensorDataset(self.true_data, self.true_labels)
        self.trut_dataset_basic = dutils.GenericDataset(self.true_data, self.true_labels)

    def generate_mc(self,n):
        sig,_,_=self.jdgs.generate_dataset(n)
        bkg,_,_=self.jdgb.generate_dataset(n)
        data   = torch.cat((torch.tensor(sig),torch.tensor(bkg)))
        labels = torch.cat((torch.ones(len(sig)),torch.zeros(len(bkg))))
        return data,labels

    def generate_data(self,n):
        sig,_,_=self.jdgd.generate_dataset(n)
        bkg,_,_=self.jdgb.generate_dataset(n)
        data   = torch.cat((torch.tensor(sig),torch.tensor(bkg)))
        labels = torch.cat((torch.ones(len(sig)),torch.zeros(len(bkg))))
        return data,labels
    
    def train_dataloader(self):
        loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, 
                            pin_memory=self.pin_memory, num_workers=self.num_workers)
        return loader
    
    def val_dataloader(self):
        loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True,
                            pin_memory=self.pin_memory, num_workers=self.num_workers)
        return loader
    
    def test_dataloader(self):
        loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                            pin_memory=self.pin_memory, num_workers=self.num_workers)
        return loader

class FlatDataset(GenericDataModule):
    def __init__(self,ndisc,num_train,num_val,num_test,nrand=16,dilute=1.,
                 **kwargs):
        super().__init__(**kwargs)
        self.dilute = dilute
        self.ndisc = ndisc
        self.nrand = nrand
        self.num_train = num_train
        self.num_val   = num_val
        self.num_test  = num_test
        self.rand_matrix = self.random_rotation_matrix(ndisc+nrand)
        
        self.view_generator = dutils.viewGenerator(dutils.smear,2)
        self.train_data, self.train_labels = self.generate(self.num_train,iDilute=False)
        self.train_dataset = dutils.AugmentationDataset(TensorDataset(self.train_data, self.train_labels),self.view_generator)
        self.train_dataset_basic = dutils.GenericDataset(self.train_data, self.train_labels)
        
        self.val_data, self.val_labels = self.generate(self.num_val)
        self.val_dataset = dutils.AugmentationDataset(TensorDataset(self.val_data, self.val_labels),self.view_generator)
        self.val_dataset_basic = dutils.GenericDataset(self.train_data, self.train_labels)

        self.test_data, self.test_labels = self.generate(self.num_test)
        self.test_dataset = TensorDataset(self.test_data, self.test_labels)
        self.test_dataset_basic = dutils.GenericDataset(self.train_data, self.train_labels)

        self.true_data, self.true_labels = self.generate(self.num_test,True,iDilute=False)
        self.true_dataset = TensorDataset(self.true_data, self.true_labels)
        self.true_dataset_basic = dutils.GenericDataset(self.true_data, self.true_labels)

        self.trut_data, self.trut_labels = self.generate(self.num_test,True)
        self.trut_dataset = TensorDataset(self.true_data, self.true_labels)
        self.trut_dataset_basic = dutils.GenericDataset(self.true_data, self.true_labels)

    def random_rotation_matrix(self,dim):
        # Generate a random orthogonal matrix
        random_matrix = np.random.randn(dim, dim)
        Q, R = np.linalg.qr(random_matrix)
        # Ensure the determinant is 1 to represent a proper rotation
        D = np.diag(np.sign(np.diag(R)))
        return Q @ D

    def generate(self,n,iData=False,iMix=True,iDilute=False):
        ndim=self.ndisc+self.nrand
        s = np.empty((n,ndim))
        b = np.empty((n,ndim))
        c = np.empty((n,ndim))
        #s[:,-1]=np.random.normal(0,0.5,n)
        #b[:,-1]=np.random.uniform(-3,3,n)
        for pVar in range(self.nrand):
            s[:,pVar+self.ndisc]=np.random.uniform(0.0,1,n)
            b[:,pVar+self.ndisc]=np.random.uniform(0.0,1,n)
            c[:,pVar+self.ndisc]=np.random.uniform(0.0,1,n)
        val=0.05
        if iData == 1:
            val=0.2
        for pVar in range(self.ndisc):
            if iDilute:
                nsig=int(self.dilute*n)
                print("nsig",nsig,n)
                s[0:nsig,pVar]=np.random.triangular(0.,1.-val, 1, nsig)
                s[nsig:n,pVar]=np.random.triangular(0.,val, 1, n-nsig)
            else:
                s[:,pVar]=np.random.triangular(0.,1.-val, 1, n)
            b[:,pVar]=np.random.triangular(0, val,    1, n)
            c[:,pVar]=np.random.triangular(0, 0.5,    1, n)
        if iMix:
            m=self.rand_matrix
            ms=np.tile(m, (n, 1,1))
            mb=np.tile(m, (n, 1,1))
            mc=np.tile(m, (n, 1,1))
            stmp = np.reshape(s[:,:],(n,1,ndim))
            btmp = np.reshape(b[:,:],(n,1,ndim))
            ctmp = np.reshape(c[:,:],(n,1,ndim))
            stmp = np.matmul(stmp , ms)
            btmp = np.matmul(btmp , mb)
            ctmp = np.matmul(ctmp , mc)
            s[:,:] = stmp[:,0,:]
            b[:,:] = btmp[:,0,:]
            c[:,:] = ctmp[:,0,:]
        data   = torch.cat((torch.tensor(s),torch.tensor(b),torch.tensor(c)))
        labels = torch.cat((torch.ones(len(s)),torch.zeros(len(b)),torch.ones(len(c))*2))
        return data,labels


    def train_dataloader(self):
        loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, 
                            pin_memory=self.pin_memory, num_workers=self.num_workers)
        return loader
    
    def val_dataloader(self):
        loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True,
                            pin_memory=self.pin_memory, num_workers=self.num_workers)
        return loader
    
    def test_dataloader(self):
        loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                            pin_memory=self.pin_memory, num_workers=self.num_workers)
        return loader
    
class NoisyImagenetteDataset(GenericDataModule):
    def __init__(self,image_width,eps=0.2,p=0.5,sup_simclr=False,**kwargs):
        super().__init__(**kwargs)
        
        if sup_simclr:
            self.simclr_augment = v2.Compose([
                v2.PILToTensor(), # operations are more efficient on tensors
                v2.Resize(256),
                v2.CenterCrop(image_width),
                v2.ToDtype(torch.float32,scale=True),
                v2.RandomApply([v2.GaussianNoise(eps)],p=p)
            ])
            self.simclr_views = self.simclr_augment
        else:
            # augmentations from original simCLR paper on ImageNet
            self.simclr_augment = v2.Compose([
                v2.PILToTensor(), # operations are more efficient on tensors
                v2.RandomResizedCrop(image_width),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomApply([v2.ColorJitter(0.8,0.8,0.8,0.2)],p=0.8),
                v2.RandomGrayscale(p=0.2),
                v2.RandomApply([v2.GaussianBlur(kernel_size=23)],p=0.5),
                v2.ToDtype(torch.float32,scale=True)
            ])
            # view generator for getting two augmentations per image
            self.simclr_views = dutils.viewGenerator(self.simclr_augment,2)

        # augmentations for ImageNet test evaluation - just resize and crop
        self.test_augment = v2.Compose([v2.PILToTensor(),
                                        v2.Resize(256),
                                        v2.CenterCrop(image_width),
                                        v2.ToDtype(torch.float32,scale=True),
                                        v2.RandomApply([v2.GaussianNoise(eps)],p=p)
                                        ])
        
        # Imagenette datasets
        self.train_dataset = Imagenette(root="/n/holystore01/LABS/iaifi_lab/Lab/sambt/neurips25/imagenette/",
                           split='train',
                           size='full',
                           download=False,
                           transform=self.simclr_views)
        self.val_dataset = Imagenette(root="/n/holystore01/LABS/iaifi_lab/Lab/sambt/neurips25/imagenette/",
                                split='val',
                                size='full',
                                download=False,
                                transform=self.simclr_views)
        self.test_dataset = Imagenette(root="/n/holystore01/LABS/iaifi_lab/Lab/sambt/neurips25/imagenette/",
                                split='val',
                                size='full',
                                download=False,
                                transform=self.test_augment)
        
    def train_dataloader(self):
        loader = DataLoader(self.train_dataset, shuffle=True, **self.loader_kwargs)
        return loader
    
    def val_dataloader(self):
        loader = DataLoader(self.val_dataset, shuffle=True, **self.loader_kwargs)
        return loader
    
    def test_dataloader(self):
        loader = DataLoader(self.test_dataset, shuffle=False, **self.loader_kwargs)
        return loader

class TensorImagenetteDataset(GenericDataModule):
    def __init__(self,image_width,preload=True,**kwargs):
        super().__init__(**kwargs)
        
        # augmentations from original simCLR paper on ImageNet
        self.simclr_augment = v2.Compose([
            v2.RandomResizedCrop(image_width),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomApply([v2.ColorJitter(0.8,0.8,0.8,0.2)],p=0.8),
            v2.RandomGrayscale(p=0.2),
            v2.RandomApply([v2.GaussianBlur(kernel_size=23)],p=0.5),
            v2.ToDtype(torch.float32,scale=True)
        ])
        # view generator for getting two augmentations per image
        self.simclr_views = dutils.viewGenerator(self.simclr_augment,2)

        # augmentations for ImageNet test evaluation - just resize and crop
        self.test_augment = v2.Compose([v2.Resize(256),
                                        v2.CenterCrop(image_width),
                                        v2.ToDtype(torch.float32,scale=True)
                                        ])
        
        # Imagenette datasets
        self.train_dataset = TensorImagenette(root="/n/holystore01/LABS/iaifi_lab/Lab/sambt/neurips25/imagenette_tensors/",
                           split='train',
                           size='full',
                           download=False,
                           transform=self.simclr_views,
                           preload=preload)
        self.val_dataset = TensorImagenette(root="/n/holystore01/LABS/iaifi_lab/Lab/sambt/neurips25/imagenette_tensors/",
                                split='val',
                                size='full',
                                download=False,
                                transform=self.simclr_views,
                                preload=preload)
        self.test_dataset = TensorImagenette(root="/n/holystore01/LABS/iaifi_lab/Lab/sambt/neurips25/imagenette_tensors/",
                                split='val',
                                size='full',
                                download=False,
                                transform=self.test_augment,
                                preload=preload)
        
    def train_dataloader(self):
        loader = DataLoader(self.train_dataset,shuffle=True, **self.loader_kwargs)
        return loader
    
    def val_dataloader(self):
        loader = DataLoader(self.val_dataset, shuffle=True, **self.loader_kwargs)
        return loader
    
    def test_dataloader(self):
        loader = DataLoader(self.test_dataset, shuffle=False, **self.loader_kwargs)
        return loader
    
class JetClassDataset(GenericDataModule):
    def __init__(self,classes,input_config,
                 **kwargs):
        super().__init__(**kwargs)
        self.train_dir = "/n/holystore01/LABS/iaifi_lab/Lab/sambt/JetClass/train_100M/"
        self.val_dir = "/n/holystore01/LABS/iaifi_lab/Lab/sambt/JetClass/val_5M/"
        self.test_dir = "/n/holystore01/LABS/iaifi_lab/Lab/sambt/JetClass/test_20M/"
        
        self.all_classes = ["qcd","wqq","zqq","ttbar","hbb"]
        self.all_class_fileHeaders = {
            "qcd":"ZJetsToNuNu",
            "wqq":"WToQQ",
            "zqq":"ZToQQ",
            "ttbar":"TTBar",
            "hbb":"HToBB"
        }

        assert set(classes).issubset(self.all_classes)
        self.classes = classes
        self.input_config = input_config
        
        self.train_file_dict = {c:glob.glob(f"{self.train_dir}/{self.all_class_fileHeaders[c]}_*.root") for c in self.classes}
        self.val_file_dict = {c:glob.glob(f"{self.val_dir}/{self.all_class_fileHeaders[c]}_*.root") for c in self.classes}
        self.test_file_dict = {c:glob.glob(f"{self.test_dir}/{self.all_class_fileHeaders[c]}_*.root") for c in self.classes}

    def train_dataloader(self):
        train_dataset = SimpleIterDataset(
            self.train_file_dict,
            self.input_config,
            for_training=True,
            extra_selection=None,
            fetch_by_files=False,
            fetch_step=0.01,
            file_fraction=1,
            infinity_mode=False,
            in_memory=False,
            remake_weights=True,
            load_range_and_fraction=((0,1),1),
            name='train',
            async_load=True
        )
        loader = DataLoader(train_dataset,persistent_workers=True,**self.loader_kwargs)
        return loader
        
    def val_dataloader(self):
        val_dataset = SimpleIterDataset(
            self.val_file_dict,
            self.input_config,
            for_training=True,
            extra_selection=None,
            fetch_by_files=False,
            fetch_step=0.01,
            file_fraction=1,
            infinity_mode=False,
            in_memory=False,
            remake_weights=True,
            load_range_and_fraction=((0,1),1),
            name='val',
            async_load=True
        )
        loader = DataLoader(val_dataset,persistent_workers=True,**self.loader_kwargs)
        return loader

    def test_dataloader(self):
        test_dataset = SimpleIterDataset(
            self.test_file_dict,
            self.input_config,
            for_training=False,
            extra_selection=None,
            fetch_by_files=False,
            fetch_step=0.01,
            file_fraction=1,
            infinity_mode=False,
            in_memory=False,
            remake_weights=True,
            load_range_and_fraction=((0,1),1),
            name='val',
            async_load=True
        )
        loader = DataLoader(test_dataset,persistent_workers=True,**self.loader_kwargs)
        return loader
    

class CIFAR10Dataset(GenericDataModule):
    def __init__(self,resnet_type,grayscale=False,**kwargs):
        super().__init__(**kwargs)
        self.transform = dutils.ResNet50Transform(resnet_type=resnet_type,grayscale=grayscale)

        self.train_dataset = CIFAR10(root="/n/holystore01/LABS/iaifi_lab/Lab/sambt/neurips25/cifar10",
                                    train=True,
                                    download=False,
                                    transform=self.transform)
        self.val_dataset = CIFAR10(root="/n/holystore01/LABS/iaifi_lab/Lab/sambt/neurips25/cifar10/",
                                    train=False,
                                    download=False,
                                    transform=self.transform)
        self.test_dataset = CIFAR10(root="/n/holystore01/LABS/iaifi_lab/Lab/sambt/neurips25/cifar10/",
                                    train=False,
                                    download=False,
                                    transform=self.transform)
    def train_dataloader(self):
        loader = DataLoader(self.train_dataset,shuffle=True, **self.loader_kwargs)
        return loader
    
    def val_dataloader(self):
        loader = DataLoader(self.val_dataset, shuffle=True, **self.loader_kwargs)
        return loader
    
    def test_dataloader(self):
        loader = DataLoader(self.test_dataset, shuffle=False, **self.loader_kwargs)
        return loader


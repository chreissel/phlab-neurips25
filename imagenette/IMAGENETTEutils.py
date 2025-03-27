from PIL import Image
import torchvision
import random
from pathlib import Path
from tqdm import tqdm
import torch
from torchvision.transforms import v2

class ImagenetteDataset(object):
    def __init__(self, data_path='/n/holystore01/LABS/iaifi_lab/Lab/sambt/neurips25/',
                 augmentation=None,
                 img_size=None,
                 validation=False, 
                 preload=False):
        if img_size is None:
            self.folder = Path(f"{data_path}/imagenette2/train") if not validation else Path(f"{data_path}/imagenette2/val")
        else:
            self.folder = Path(f'{data_path}/imagenette2-{img_size}/train/') if not validation else Path(f'{data_path}/imagenette2-{img_size}/val/')
        self.classes = ['n01440764', 'n02102040', 'n02979186', 'n03000684', 'n03028079',
                        'n03394916', 'n03417042', 'n03425413', 'n03445777', 'n03888257']
        
        self.augmentation = augmentation

        self.images = []
        for cls in self.classes:
            cls_images = list(self.folder.glob(cls + '/*.JPEG'))
            self.images.extend(cls_images)
        random.shuffle(self.images)
        self.validation = validation

        self.preload = preload
        if preload:
            self.all_images = []
            self.all_classes = []
            for image_fname in tqdm(self.images):
                image, label = self.load_image(image_fname)
                self.all_images.append(image)
                self.all_classes.append(label)
            self.all_classes = torch.tensor(self.all_classes).reshape(-1,1)

    def load_image(self, image_fname):
        image = Image.open(image_fname)
        label = image_fname.parent.stem
        label = self.classes.index(label)
        image = torchvision.transforms.functional.to_tensor(image)
        if image.shape[0] == 1: image = image.expand(3, -1, -1)

        return image, label

    def __getitem__(self, index):
        if self.preload:
            img,c = self.all_images[index], self.all_classes[index]
            if self.augmentation is not None:
                img = self.augmentation(img)
            return img,c
        else:
            image_fname = self.images[index]
            image, label = self.load_image(image_fname)
            if self.augmentation is not None:
                image = self.augmentation(image)
            return image, label

    def __len__(self):
        return len(self.all_images) if self.preload else len(self.images)
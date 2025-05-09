import numpy as np
from torch.utils.data import Dataset

class GwakDataset(Dataset):
    def __init__(self, 
                 path='/n/holystore01/LABS/iaifi_lab/Lab/phlab-neurips25/GWAK/', era='a',
                 anomaly_index = 7, remove_anomaly=True, normalize=False):
        data = self.fill_data(path+'/O3'+era+'_dataset.npz', normalize=normalize)
        glitches = self.fill_data(path+'/glitches.npz', normalize=normalize, label=9)
        data_full = np.concatenate([data[0], glitches[0]], axis=0)
        labels_full = np.concatenate([data[1], glitches[1]], axis=0)
        idx_mask = mask(labels_full, anomaly_idx, remove_anomaly).flatten()
        self.data = np.delete(data_full, idx_mask, axis=0)
        self.labels = np.delete(labels_full, idx_mask, axis=0)

    def fill_data(self, path, normalize=True, label=None):
        file = np.load(path)
        if path.endswith('npz'):
            data = file['data']
        else:
            data = file
        if normalize:
            stds = np.std(data, axis=-1)[:, :, np.newaxis]
            data = data/stds
        #data = np.swapaxes(data, 1, 2)
        data = np.float32(data)
        if 'label' in file.keys():
            target = file['label']
        else:
            target = np.full(data.shape[0], label)
        return data, target
    
    def generate_augmentation(self,batch):
        return None
    
    def normalize(self,batch):
        return (batch - self.mean.to(batch)) / self.std.to(batch)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self,idx):
        return self.data[idx, :, :], self.labels[idx]

    def mask(self, labels, anomaly_idx, remove_anomaly=True):
        if remove_anomaly:
            return (labels == anomaly_idx)
        else:
            return (labels != anomaly_idx)

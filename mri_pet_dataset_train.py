import os
import torch
import scipy.io as sio
import numpy as np
from torch.utils.data.dataset import Dataset
from sklearn.model_selection import KFold
import nibabel as nib

_PATH = './testdata'
class MRIandPETdataset(Dataset):
    def __init__(self, image_path = './Dataset/ADNI1'):  
        self.path = image_path
        label = os.listdir(self.path)
        train_data = []
        for image_label in label:
            train_data.append(image_label)
        train_data = np.asarray(train_data)
        self.name = train_data
    
    def __len__(self):
        return len(self.name)
        
    def __getitem__(self, index):

        file_name = self.name[index]                    
        path = os.path.join(self.path, file_name)
        label = int(file_name[0]) 
        out = nib.load(path).get_fdata()
        data = np.array(out).astype(np.float32)       
        return data, label, file_name
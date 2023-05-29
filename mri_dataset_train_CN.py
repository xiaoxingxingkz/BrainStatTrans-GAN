import os
import torch
import scipy.io as sio
import numpy as np
from torch.utils.data.dataset import Dataset
from sklearn.model_selection import KFold
import random
import nibabel as nib

_PATH = './testdata'
class MRIandPETdataset_CN(Dataset):
    def __init__(self, image_path = './ADNI1_MRI_CN'):  
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
        age = int(file_name[15:17])
        out = nib.load(path).get_fdata()
        data = np.array(out).astype(np.float32)           
        return data, label, age, file_name
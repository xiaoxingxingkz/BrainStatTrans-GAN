import os
import torch
import scipy.io as sio
import numpy as np
import nibabel as nib
from torch.utils.data.dataset import Dataset

class TestDataset(Dataset):
    def __init__(self, image_path = './Dataset/ADNI2'):   
        self.path = image_path
        label = os.listdir(self.path)
        test_data = []
        for image_label in label:
            test_data.append(image_label)
        test_data = np.asarray(test_data)
        self.name = test_data
    
    def __len__(self):
        return len(self.name)
        
    def __getitem__(self, index):
        file_name = self.name[index]               
        path = os.path.join(self.path, file_name)
        out = nib.load(path).get_fdata()
        data = np.array(out).astype(np.float32)
        label = int(file_name[0]) 
        n = len(self.name)
        return data, label, file_name
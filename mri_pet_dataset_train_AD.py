import os
import torch
import scipy.io as sio
import numpy as np
from torch.utils.data.dataset import Dataset
from sklearn.model_selection import KFold
import nibabel as nib

_PATH = './testdata'
class MRIandPETdataset_AD(Dataset):
    def __init__(self, image_path = '/media/sdb/gaoxingyu/CN_AD/ADNI1_MRI_AD'):   
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
        # img = sio.loadmat(path)
        # out = img['data']

        out = nib.load(path).get_fdata()
        data = np.array(out).astype(np.float32)           
        # input_data = data[0, :, :, :]
        # output_data = data[1, :, :, :]
        # max_val = output_data.max()
        # min_val = output_data.min()
        # output_data = (output_data - min_val) / (max_val - min_val)
        # data = np.stack((input_data, output_data), axis=0)

        return data, label, age, file_name
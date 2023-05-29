import os
import torch
import scipy.io as sio
import numpy as np
from torch.utils.data.dataset import Dataset
import nibabel as nib

class AD_Dataset(Dataset):
    def __init__(self, image_path = './Dataset/ADNI2'):  #/media/sdb/gaoxingyu/BrainTransGAN/ADNI2_MRI_AD 
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
        # img = sio.loadmat(path)
        # out = img['data']   
        out = nib.load(path).get_fdata()
        data = np.array(out).astype(np.float32)
        # age = int(file_name[15:17])           
        # input_data = data[0, :, :, :]
        # output_data = data[1, :, :, :]
        # max_val = output_data.max()
        # min_val = output_data.min()
        # output_data = (output_data - min_val) / (max_val - min_val)
        # data = np.stack((input_data, output_data), axis=0)
        label = int(file_name[0]) 
        n = len(self.name)

        return data, label, file_name
import scipy.io as sio 
import numpy as np
import torch
import os
import math
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch import autograd
from torch.autograd import Variable
import nibabel as nib
from scipy.ndimage.interpolation import zoom

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cuda = torch.cuda.is_available()


from gan_models import *
from mri_pet_dataset_test_normal import CN_Dataset
from mri_pet_dataset_test_abnormal import AD_Dataset
from densenet import *

seed = 23
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

SAVE_PATH_cn = './ad_results'
SAVE_PATH_ad = './ad_results'

# SAVE_PATH_cn = './gen_cn_seq'
# SAVE_PATH_ad = './gen_ad_seq'


WORKERS = 0
TEST_BATCH_SIZE = 1
dataset_test_cn = CN_Dataset()
data_loader_test_cn = torch.utils.data.DataLoader(dataset_test_cn, batch_size = TEST_BATCH_SIZE, shuffle = False, num_workers = WORKERS)

dataset_test_ad = AD_Dataset()
data_loader_test_ad = torch.utils.data.DataLoader(dataset_test_ad, batch_size = TEST_BATCH_SIZE, shuffle = False, num_workers = WORKERS)

G_MRI = Generator_U_DAM().cuda() 
G_MRI_CN = Generator_U_DAM().cuda() 


G_MRI.load_state_dict(torch.load('./generated_models/Generator_U_DAM_NEW/170_Loss14.3326_TestSPE1.0_TestSEN0.0_G_MRI.pth'))  
G_MRI_CN.load_state_dict(torch.load('./generated_models/Generator_U_DAM/150_Loss3.0568_TestSPE0.855_TestSEN0.9167_G_MRI.pth')) 


T_MRI = densenet21().cuda()
T_MRI.load_state_dict(torch.load('./classification_models/Densenet_MRI_255_NEW/23_TLoss0.0164_TrainACC0.9715_TestACC0.882_TestSEN0.9103_TestSPE0.86_TestAUC0.934_F1S0.8711.pth'))




# fixed age_coding
def PositionalEncoding(d_model, max_len=3):
    # Compute the positional encodings once in log space.
    pe = Variable(torch.zeros(max_len, d_model), requires_grad=True).cuda()
    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) *
                        -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
age_coding = PositionalEncoding(64*8*10*8, 36)

##########################################################
#                        test cn
##########################################################
for test_s in range(1):
    for val_test_data in data_loader_test_cn:

        val_test_imgs = val_test_data[0]
        val_test_data_batch_size = val_test_imgs.size()[0]
        original_images = val_test_imgs[:, :, :, :].view(val_test_data_batch_size, 1, 76, 94, 76)
        original_images = Variable(original_images.cuda(), requires_grad=False)
        # age = val_test_data[2]-55
        # position = Variable(age_coding[age].cuda(), requires_grad=False)

        result_mri = T_MRI(original_images)
        out_mri = F.softmax(result_mri, dim=1)
        _, predicted_mri = torch.max(out_mri.data, 1)
        PREDICTED_MRI = predicted_mri.data.cpu().numpy()
 
        if PREDICTED_MRI == 0:
            gen_fake = G_MRI_CN(original_images)
        else:
            gen_fake = G_MRI(original_images)



        ori_data_mri = np.squeeze(original_images.data.cpu().numpy())
        gen_data_mri = np.squeeze(gen_fake.data.cpu().numpy())
        residual_mri = gen_data_mri - ori_data_mri 
        residual_mri = np.maximum(residual_mri, - residual_mri) 
        # residual_mri[residual_mri > 10] = 0


        img_output_bl = zoom(gen_data_mri, (2, 2, 2))
        np.maximum(img_output_bl, 0)
        gen_data_mri = img_output_bl.astype(np.uint8)   

        # residual_mri = np.stack((ori_data_mri, residual_mri), axis=0)

        nii_image_gen_mri_res = nib.Nifti1Image(residual_mri, np.eye(4))
        nii_image_gen_mri_gen = nib.Nifti1Image(gen_data_mri, np.eye(4))
        nii_image_gen_mri_real = nib.Nifti1Image(ori_data_mri, np.eye(4))

        val_test_name = val_test_data[2][0].split('.')[0]

        val_test_name_mri_res = val_test_name + '_res.nii.gz'
        path_fullname_mri_res = os.path.join(SAVE_PATH_cn, val_test_name_mri_res)
        nib.save(nii_image_gen_mri_res, path_fullname_mri_res)

        val_test_name_mri_gen = val_test_name + '_gen.nii.gz'
        path_fullname_mri_gen = os.path.join(SAVE_PATH_cn, val_test_name_mri_gen)
        nib.save(nii_image_gen_mri_gen, path_fullname_mri_gen)

        val_test_name_mri_real = val_test_name + '_real.nii.gz'
        path_fullname_mri_real = os.path.join(SAVE_PATH_cn, val_test_name_mri_real)
        nib.save(nii_image_gen_mri_real, path_fullname_mri_real)

##########################################################
#                        test ad
##########################################################
for test_s in range(1):
    for val_test_data in data_loader_test_ad:

        val_test_imgs = val_test_data[0]
        val_test_data_batch_size = val_test_imgs.size()[0]
        original_images = val_test_imgs[:, :, :, :].view(val_test_data_batch_size, 1, 76, 94, 76)
        original_images = Variable(original_images.cuda(), requires_grad=False)
        # age = val_test_data[2]-55
        # position = Variable(age_coding[age].cuda(), requires_grad=False)

        result_mri = T_MRI(original_images)
        out_mri = F.softmax(result_mri, dim=1)
        _, predicted_mri = torch.max(out_mri.data, 1)
        PREDICTED_MRI = predicted_mri.data.cpu().numpy()
 
        if PREDICTED_MRI == 0:
            gen_fake = G_MRI_CN(original_images)
        else:
            gen_fake = G_MRI(original_images)

        ori_data_mri = np.squeeze(original_images.data.cpu().numpy())
        gen_data_mri = np.squeeze(gen_fake.data.cpu().numpy())

        residual_mri = gen_data_mri - ori_data_mri
        residual_mri = np.maximum(residual_mri, - residual_mri)
        residual_mri[residual_mri > 10] = 0

        # img_output_bl = zoom(ori_data_mri, (2, 2, 2))
        # np.maximum(img_output_bl, 0)
        # ori_data_mri = img_output_bl.astype(np.uint8)   

        nii_image_gen_mri_res = nib.Nifti1Image(residual_mri, np.eye(4))
        nii_image_gen_mri_gen = nib.Nifti1Image(gen_data_mri, np.eye(4))
        nii_image_gen_mri_real = nib.Nifti1Image(ori_data_mri, np.eye(4))

        val_test_name = val_test_data[2][0].split('.')[0]

        val_test_name_mri_res = val_test_name + '_res.nii.gz'
        path_fullname_mri_res = os.path.join(SAVE_PATH_ad, val_test_name_mri_res)
        nib.save(nii_image_gen_mri_res, path_fullname_mri_res)

        val_test_name_mri_gen = val_test_name + '_gen.nii.gz'
        path_fullname_mri_gen = os.path.join(SAVE_PATH_ad, val_test_name_mri_gen)
        nib.save(nii_image_gen_mri_gen, path_fullname_mri_gen)

        val_test_name_mri_real = val_test_name + '.nii.gz'
        path_fullname_mri_real = os.path.join(SAVE_PATH_ad, val_test_name_mri_real)
        nib.save(nii_image_gen_mri_real, path_fullname_mri_real)

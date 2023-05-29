import numpy as np
import nibabel as nib
import scipy.io as sio 
import torch
import os
from torch import nn
from torch import optim
from torch.nn import functional as F 
from torch import autograd
from torch.autograd import Variable
import time
from ssim_loss import SSIM 
from sklearn.metrics import roc_curve, auc
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cuda = torch.cuda.is_available()

from gan_models import *
from mri_dataset_train_CN import MRIandPETdataset_CN
from mri_dataset_train_AD import MRIandPETdataset_AD
from mri_dataset_test_normal import CN_Dataset
from mri_dataset_test_abnormal import AD_Dataset
from densenet import *


# initial for recurrence
seed = 23
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False

# initial setup
MODEL_PATH = './generated_models/Generator_U_DAM_NEW'
ISHOWN_PATH = './cn_results'
ISHOWN_PATH_ = './ad_results'
ISHOWN_PATH80 = './cn_150'
ISHOWN_PATH_80 = './ad_150'
LOSS_PATH = './loss'
TRAIN_BATCH_SIZE = 4
TEST_BATCH_SIZE = 1
LR = 1e-4
EPOCH = 200
WORKERS = 0

factor = 0.7

dataset_train_cn = MRIandPETdataset_CN()
data_loader_train_cn = torch.utils.data.DataLoader(dataset_train_cn, batch_size = TRAIN_BATCH_SIZE, shuffle = True, num_workers = WORKERS)
data_loader_valid_cn = torch.utils.data.DataLoader(dataset_train_cn, batch_size = TEST_BATCH_SIZE, shuffle = True, num_workers = WORKERS)

dataset_train_ad = MRIandPETdataset_AD()
data_loader_train_ad = torch.utils.data.DataLoader(dataset_train_ad, batch_size = TRAIN_BATCH_SIZE, shuffle = True, num_workers = WORKERS)
data_loader_valid_ad = torch.utils.data.DataLoader(dataset_train_ad, batch_size = TEST_BATCH_SIZE, shuffle = True, num_workers = WORKERS)


# load test data 
dataset_test_cn = CN_Dataset()
data_loader_test_cn = torch.utils.data.DataLoader(dataset_test_cn, batch_size = TEST_BATCH_SIZE, shuffle = False, num_workers = WORKERS)

dataset_test_ad = AD_Dataset()
data_loader_test_ad = torch.utils.data.DataLoader(dataset_test_ad, batch_size = TEST_BATCH_SIZE, shuffle = False, num_workers = WORKERS)


criterion_bce = nn.BCELoss().cuda()
criterion_l1 = nn.L1Loss().cuda()
criterion_mse = nn.MSELoss().cuda()
cirterion_ssim = SSIM().cuda()
cirterion = nn.CrossEntropyLoss().cuda()


# number of iterations
iter_g = 1
iter_d = 1
iter_t = 0

G_MRI = Generator_U_DAM().cuda() 

T_MRI = densenet21().cuda() #This pre-tained model can be found in our first repository!!! 
T_MRI.load_state_dict(torch.load('./classification_models/Densenet_MRI/23_TLoss0.0164_TrainACC0.9715_TestACC0.882_TestSEN0.9103_TestSPE0.86_TestAUC0.934_F1S0.8711.pth'))


DIS_MRI = Discriminator_().cuda()
g_mri_optimizer = optim.Adam(G_MRI.parameters(), lr=0.0001) 
d_mri_optimizer = optim.Adam(DIS_MRI.parameters(), lr=0.0004) 


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
age_coding = PositionalEncoding(1*76*94*76, 40)


'''
########################################################################################################
########################################################################################################
####################################### Processing of Generating #######################################
######################################################################################################## 
########################################################################################################
'''
# loss_filename = os.path.join(LOSS_PATH, 'loss_data.txt')
# fw = open(loss_filename, 'w')  


for iteration in range(EPOCH):

    print(iteration + 1)
    start_time = time.time()
    G_MRI.train()

    # if iteration >= 200:
    #     for p in t_optimizer.param_groups:
    #         p['lr'] *= 0.9
    
    total_loss_mri = 0
    total_num_mri = 0

    if iteration < 150:
        load_valid_data = data_loader_valid_cn
    else: 
        load_valid_data = data_loader_valid_ad

    if iteration < 150:
        for train_data in data_loader_train_cn:

            labels = train_data[1]
            labels_ = Variable(labels).cuda()
            images = train_data[0]
            _batch_size = images.size()[0]

            mri_images = images[:, :, :, :].view(_batch_size, 1, 76, 94, 76)
            mri_images = Variable(mri_images.cuda(), requires_grad=False)

            real_y = Variable(torch.ones((_batch_size, 1)).cuda())
            fake_y = Variable(torch.zeros((_batch_size, 1)).cuda())
            ##########################################################
            #                      Generator_MRI
            ##########################################################
            # age = train_data[2]-55
            # position = Variable(age_coding[age].cuda(), requires_grad=False)


            for itera_g in range(iter_g):
                for p in G_MRI.parameters():
                    p.requires_grad = True

                g_mri_optimizer.zero_grad()
                mri_fake = G_MRI(mri_images)
                d_mri = DIS_MRI(mri_fake)

                # loss function
                discriminate_loss_mri = criterion_bce(d_mri, real_y[:_batch_size])
                loss1 = criterion_l1(mri_fake, mri_images) 
                loss2 = criterion_mse(mri_fake, mri_images) 
                loss3 = cirterion_ssim(mri_fake, mri_images)  
                a = loss1.cpu().data
                b = loss2.cpu().data
                c = loss3.cpu().data
                
                max_value = max(a, b, c)
                a_value = int(math.log(max_value/a, 10))
                b_value = int(math.log(max_value/b, 10))
                c_value = int(math.log(max_value/c, 10))
            
                theta_a = 1
                theta_b = 1
                theta_c = 1

                if a_value > 0:
                    theta_a = 10**a_value

                if b_value > 0:
                    theta_b = 10**b_value

                if c_value > 0:
                    theta_c = 10**c_value

                generate_loss = loss1 + loss2 + theta_c * loss3 


                if iteration < 40:
                    loss_g_mri = generate_loss 
                elif iteration >= 40 and iteration <= 80:
                    loss_g_mri = generate_loss + discriminate_loss_mri
                else:
                    loss_g_mri = generate_loss 
                

                loss_g_mri.backward()
                g_mri_optimizer.step()
            total_loss_mri += loss_g_mri.item()
            total_num_mri += 1

            ##########################################################
            #                      Discriminator MRI
            ##########################################################
            if iteration >= 40 and iteration <= 80:
                for itera_d in range(iter_d):
                    for p in DIS_MRI.parameters():
                        p.requires_grad = True

                    d_mri_optimizer.zero_grad()

                    mri_fake = G_MRI(mri_images)
                    x_d_mri = DIS_MRI(mri_images)
                    y_d_mri = DIS_MRI(mri_fake)

                    # loss function
                    x_real_loss = criterion_bce(x_d_mri, real_y[:_batch_size])         
                    y_fake_loss = criterion_bce(y_d_mri, fake_y[:_batch_size])

                    loss_d = x_real_loss + y_fake_loss 

                    loss_d.backward() 
                    d_mri_optimizer.step()
    else:
        for train_data in data_loader_train_ad:

            labels = train_data[1]
            labels_ = Variable(labels).cuda()
            images = train_data[0]
            _batch_size = images.size()[0]

            mri_images = images[:, :, :, :].view(_batch_size, 1, 76, 94, 76)
            mri_images = Variable(mri_images.cuda(), requires_grad=False)


            real_y = Variable(torch.ones((_batch_size, 1)).cuda())
            fake_y = Variable(torch.zeros((_batch_size, 1)).cuda())
            ##########################################################
            #                      Generator_MRI
            ##########################################################
            # age = train_data[2]-55
            # position = Variable(age_coding[age].cuda(), requires_grad=False)

            for itera_g in range(iter_g):
                for p in G_MRI.parameters():
                    p.requires_grad = True

                g_mri_optimizer.zero_grad()
                mri_fake = G_MRI(mri_images)
                t_mri = T_MRI(mri_fake)
                t_mri_assist = T_MRI_assist(mri_fake)

                # loss function
                generate_loss = criterion_l1(mri_fake, mri_images) + criterion_mse(mri_fake, mri_images) 
                target = 1 - labels_
                t_mri_loss = cirterion(t_mri, target)
                # discriminate_loss_mri = criterion_bce(d_mri, real_y[:_batch_size])


                # if iteration%5==0:
                #     factor = factor - 0.05
                loss_g_mri = 30 * t_mri_loss + generate_loss #10 * t_mri_loss + factor * generate_loss #+ discriminate_loss_mri 


                loss_g_mri.backward()
                g_mri_optimizer.step()
            total_loss_mri += loss_g_mri.item()
            total_num_mri += 1



    ##########################################################
    #                    validation
    ##########################################################
    for p in G_MRI.parameters():
        p.requires_grad = False
 
    for train_s in range(1):
        TP_MRI = 0
        FP_MRI = 0
        FN_MRI = 0
        TN_MRI = 0

        TP_PET = 0
        FP_PET = 0
        FN_PET = 0
        TN_PET = 0

        for val_trian_data in load_valid_data:

            val_trian_imgs = val_trian_data[0]
            val_trian_labels = val_trian_data[1]
            val_trian_labels_ = Variable(val_trian_labels).cuda()
            val_trian_data_batch_size = val_trian_imgs.size()[0]

            # whole dataset with MRI & PET format
            #mri_images = np.zeros(shape=(_batch_size, 1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE), dtype='float32')
            mri_images = val_trian_imgs[:, :, :, :].view(val_trian_data_batch_size, 1, 76, 94, 76)
            mri_images = Variable(mri_images.cuda(), requires_grad=False)


            # MRI
            mri_fake = G_MRI(mri_images)
            result_mri = T_MRI(mri_fake)
            out_mri = F.softmax(result_mri, dim=1)
            _, predicted_mri = torch.max(out_mri.data, 1)
            PREDICTED_MRI = predicted_mri.data.cpu().numpy()
            REAL_ = val_trian_labels_.data.cpu().numpy()
            if PREDICTED_MRI == 1 and REAL_ == 1:
                TP_MRI += 1
            elif PREDICTED_MRI == 1 and REAL_ == 0:
                FP_MRI += 1
            elif PREDICTED_MRI == 0 and REAL_ == 1:
                FN_MRI += 1 
            elif PREDICTED_MRI == 0 and REAL_ == 0:
                TN_MRI += 1

        if iteration < 150:
            train_spe_mri = TN_MRI/(FP_MRI + TN_MRI)
        else: 
            train_sen_mri = TP_MRI/(TP_MRI + FN_MRI)

    ##########################################################
    #                        test cn
    ##########################################################
    for test_s in range(1):
        TP_MRI_ = 0
        FP_MRI_ = 0
        FN_MRI_ = 0
        TN_MRI_ = 0

        TP_PET_ = 0
        FP_PET_ = 0
        FN_PET_ = 0
        TN_PET_ = 0

        labels = []
        scores_MRI = []
        scores_PET = []
        for val_test_data in data_loader_test_cn:

            val_test_imgs = val_test_data[0]
            val_test_labels = val_test_data[1]
            val_test_labels_ = Variable(val_test_labels).cuda()
            val_test_data_batch_size = val_test_imgs.size()[0]

            mri_images_ = val_test_imgs[:, :, :, :].view(val_test_data_batch_size, 1, 76, 94, 76)
            mri_images_ = Variable(mri_images_.cuda(), requires_grad=False)

            REAL = val_test_labels_.data.cpu().numpy()
            labels.append(REAL)

            mri_fake = G_MRI(mri_images_)
            result_c_mri = T_MRI(mri_fake)
            out_c_mri = F.softmax(result_c_mri, dim=1)
            score_mri = out_c_mri[0][1].data.cpu().item()
            score_mri = round(score_mri, 4)
            scores_MRI.append(score_mri)
            _, predicted_MRI = torch.max(out_c_mri.data, 1)
            PREDICTED_MRI = predicted_MRI.data.cpu().numpy()

            if PREDICTED_MRI == 1 and REAL == 1:
                TP_MRI_ += 1
            elif PREDICTED_MRI == 1 and REAL == 0:
                FP_MRI_ += 1
            elif PREDICTED_MRI == 0 and REAL == 1:
                FN_MRI_ += 1 
            elif PREDICTED_MRI == 0 and REAL == 0:
                TN_MRI_ += 1

            if iteration == 150: 
                ori_data_mri = np.squeeze(mri_images_.data.cpu().numpy())
                fake_data_mri = np.squeeze(mri_fake.data.cpu().numpy())
                residual_mri = ori_data_mri - fake_data_mri


                nii_image_gen_mri_res = nib.Nifti1Image(residual_mri, np.eye(4))
                nii_image_gen_mri_gen = nib.Nifti1Image(fake_data_mri, np.eye(4))
                nii_image_gen_mri_real = nib.Nifti1Image(ori_data_mri, np.eye(4))

                val_test_name = val_test_data[2][0].split('.')[0]
                val_test_name_mri_res = val_test_name + '_res.nii.gz'
                path_fullname_mri_res = os.path.join(ISHOWN_PATH80, val_test_name_mri_res)
                nib.save(nii_image_gen_mri_res, path_fullname_mri_res)

                val_test_name_mri_gen = val_test_name + '_gen.nii.gz'
                path_fullname_mri_gen = os.path.join(ISHOWN_PATH80, val_test_name_mri_gen)
                nib.save(nii_image_gen_mri_gen, path_fullname_mri_gen)

                val_test_name_mri_real = val_test_name + '_real.nii.gz'
                path_fullname_mri_real = os.path.join(ISHOWN_PATH80, val_test_name_mri_real)
                nib.save(nii_image_gen_mri_real, path_fullname_mri_real)



            if (iteration + 1) % 10 == 0: 
                ori_data_mri = np.squeeze(mri_images_.data.cpu().numpy())
                fake_data_mri = np.squeeze(mri_fake.data.cpu().numpy())
                residual_mri = ori_data_mri - fake_data_mri

                residual_mri = np.maximum(residual_mri, - residual_mri)
                # residual_mri[residual_mri > 10] = 0

                nii_image_gen_mri_res = nib.Nifti1Image(residual_mri, np.eye(4))
                nii_image_gen_mri_gen = nib.Nifti1Image(fake_data_mri, np.eye(4))
                nii_image_gen_mri_real = nib.Nifti1Image(ori_data_mri, np.eye(4))

                val_test_name = val_test_data[2][0].split('.')[0]
                val_test_name_mri_res = val_test_name + '_res.nii.gz'
                path_fullname_mri_res = os.path.join(ISHOWN_PATH, val_test_name_mri_res)
                nib.save(nii_image_gen_mri_res, path_fullname_mri_res)

                val_test_name_mri_gen = val_test_name + '_gen.nii.gz'
                path_fullname_mri_gen = os.path.join(ISHOWN_PATH, val_test_name_mri_gen)
                nib.save(nii_image_gen_mri_gen, path_fullname_mri_gen)

                val_test_name_mri_real = val_test_name + '_real.nii.gz'
                path_fullname_mri_real = os.path.join(ISHOWN_PATH, val_test_name_mri_real)
                nib.save(nii_image_gen_mri_real, path_fullname_mri_real)


        test_spe_MRI_cn = TN_MRI_/(FP_MRI_ + TN_MRI_)


    ##########################################################
    #                        test ad
    ##########################################################
    for test_s in range(1):
        TP_MRI = 0
        FP_MRI = 0
        FN_MRI = 0
        TN_MRI = 0
        
        TP_PET = 0
        FP_PET = 0
        FN_PET = 0
        TN_PET = 0

        labels = []
        scores_MRI = []
        scores_PET = []
        for val_test_data in data_loader_test_ad:

            val_test_imgs = val_test_data[0]
            val_test_labels = val_test_data[1]
            val_test_labels_ = Variable(val_test_labels).cuda()
            val_test_data_batch_size = val_test_imgs.size()[0]

            mri_images_ = val_test_imgs[:, :, :, :].view(val_test_data_batch_size, 1, 76, 94, 76)
            mri_images_ = Variable(mri_images_.cuda(), requires_grad=False)

            REAL = val_test_labels_.data.cpu().numpy()
            labels.append(REAL)

            mri_fake = G_MRI(mri_images_)
            result_c_mri = T_MRI(mri_fake)
            out_c_mri = F.softmax(result_c_mri, dim=1)
            score_mri = out_c_mri[0][1].data.cpu().item()
            score_mri = round(score_mri, 4)
            scores_MRI.append(score_mri)
            _, predicted_MRI = torch.max(out_c_mri.data, 1)
            PREDICTED_MRI = predicted_MRI.data.cpu().numpy()

            if PREDICTED_MRI == 1 and REAL == 1:
                TP_MRI += 1
            elif PREDICTED_MRI == 1 and REAL == 0:
                FP_MRI += 1
            elif PREDICTED_MRI == 0 and REAL == 1:
                FN_MRI += 1 
            elif PREDICTED_MRI == 0 and REAL == 0:
                TN_MRI += 1


            if iteration == 150: 
                ori_data_mri = np.squeeze(mri_images_.data.cpu().numpy())
                fake_data_mri = np.squeeze(mri_fake.data.cpu().numpy())
                residual_mri = ori_data_mri - fake_data_mri
                nii_image_gen_mri = nib.Nifti1Image(residual_mri, np.eye(4))
                nii_image_gen_mri_gen = nib.Nifti1Image(fake_data_mri, np.eye(4))
                nii_image_gen_mri_real = nib.Nifti1Image(ori_data_mri, np.eye(4))

                val_test_name = val_test_data[2][0].split('.')[0]
                val_test_name_mri = val_test_name + '_res.nii.gz'
                path_fullname_mri = os.path.join(ISHOWN_PATH_80, val_test_name_mri)
                nib.save(nii_image_gen_mri, path_fullname_mri)

                val_test_name_mri_gen = val_test_name + '_gen.nii.gz'
                path_fullname_mri_gen = os.path.join(ISHOWN_PATH_80, val_test_name_mri_gen)
                nib.save(nii_image_gen_mri_gen, path_fullname_mri_gen)

                val_test_name_mri_real = val_test_name + '_real.nii.gz'
                path_fullname_mri_real = os.path.join(ISHOWN_PATH_80, val_test_name_mri_real)
                nib.save(nii_image_gen_mri_real, path_fullname_mri_real)

            if (iteration + 1) %10 == 0: 
                ori_data_mri = np.squeeze(mri_images_.data.cpu().numpy())
                fake_data_mri = np.squeeze(mri_fake.data.cpu().numpy())
                residual_mri = ori_data_mri - fake_data_mri

                residual_mri = np.maximum(residual_mri, - residual_mri)
                # residual_mri[residual_mri > 20] = 0

                
                nii_image_gen_mri = nib.Nifti1Image(residual_mri, np.eye(4))
                nii_image_gen_mri_gen = nib.Nifti1Image(fake_data_mri, np.eye(4))
                nii_image_gen_mri_real = nib.Nifti1Image(ori_data_mri, np.eye(4))

                val_test_name = val_test_data[2][0].split('.')[0]
                val_test_name_mri = val_test_name + '_res.nii.gz'
                path_fullname_mri = os.path.join(ISHOWN_PATH_, val_test_name_mri)
                nib.save(nii_image_gen_mri, path_fullname_mri)

                val_test_name_mri_gen = val_test_name + '_gen.nii.gz'
                path_fullname_mri_gen = os.path.join(ISHOWN_PATH_, val_test_name_mri_gen)
                nib.save(nii_image_gen_mri_gen, path_fullname_mri_gen)

                val_test_name_mri_real = val_test_name + '_real.nii.gz'
                path_fullname_mri_real = os.path.join(ISHOWN_PATH_, val_test_name_mri_real)
                nib.save(nii_image_gen_mri_real, path_fullname_mri_real)


        test_sen_MRI_ad = TP_MRI/(TP_MRI + FN_MRI + 0.00001)

 
    t_comp = (time.time() - start_time)
    if iteration < 150:
        print('[{}/{}]'.format(iteration + 1, EPOCH),
              'Gen_MRI_loss: {:.4f}'.format(total_loss_mri/total_num_mri),
              'Train_MRI_SPE:{:.4f}'.format(train_spe_mri*100/100),
              'Test_MRI_SPE_CN:{:.4f} {}/{}'.format(test_spe_MRI_cn*100/100, TN_MRI_, (FP_MRI_ + TN_MRI_)),
              'Test_MRI_SEN_AD:{:.4f} {}/{}'.format(test_sen_MRI_ad*100/100, TP_MRI, (TP_MRI + FN_MRI)),
              'Time_Taken: {} sec'.format(t_comp)
            )
    else:
        print('[{}/{}]'.format(iteration + 1, EPOCH),
              'Gen_MRI_loss: {:.4f}'.format(total_loss_mri/total_num_mri),
              'Train_MRI_SEN:{:.4f}'.format(train_sen_mri*100/100),
              'Test_MRI_SPE_CN:{:.4f} {}/{}'.format(test_spe_MRI_cn*100/100, TN_MRI_, (FP_MRI_ + TN_MRI_)),
              'Test_MRI_SEN_AD:{:.4f} {}/{}'.format(test_sen_MRI_ad*100/100, TP_MRI , (TP_MRI + FN_MRI)),
              'Time_Taken: {} sec'.format(t_comp)
         )
    #fw.write(str(loss_g) + '\n')
    
    #save model
    ###################################### manually ##############################################
    ###################################### manually ##############################################
    ###################################### manually ##############################################
    torch.save(G_MRI.state_dict(), os.path.join(MODEL_PATH, '{}_Loss{}_TestSPE{}_TestSEN{}_G_MRI.pth'.format(
        iteration + 1,
        round(total_loss_mri/total_num_mri, 4), 
        round(test_spe_MRI_cn, 4),
        round(test_sen_MRI_ad, 4),

              )))
# fw.close()

'''
########################################################################################################
########################################################################################################
###################################### Processing of Generating ended ##################################
######################################################################################################## 
########################################################################################################
'''

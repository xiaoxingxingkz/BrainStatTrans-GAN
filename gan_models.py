import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
           
        self.fc = nn.Sequential(nn.Conv3d(in_planes, in_planes // 8, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv3d(in_planes // 8, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Generator_U_CBAM(nn.Module):
    """Generator Unet structure"""

    def __init__(self, conv_dim=8):
        super(Generator_U_CBAM, self).__init__()
        self.conv_dim = conv_dim
        self.relu = nn.ReLU()
        self.down_sampling = nn.MaxPool3d(kernel_size=3, stride=2)
        self.ca1 = ChannelAttention(8)
        self.sa1 = SpatialAttention(7)
        self.ca2 = ChannelAttention(16)
        self.sa2 = SpatialAttention(5)
        self.ca3 = ChannelAttention(32)
        self.sa3 = SpatialAttention(3)

        #Encoder
        self.tp_conv1 = nn.Conv3d(1, conv_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm3d(conv_dim)
        self.tp_conv2 = nn.Conv3d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm3d(conv_dim)

        self.tp_conv3 = nn.Conv3d(conv_dim, conv_dim*2, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm3d(conv_dim*2)
        self.tp_conv4 = nn.Conv3d(conv_dim*2, conv_dim*2, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn4 = nn.BatchNorm3d(conv_dim*2)  

        self.tp_conv5 = nn.Conv3d(conv_dim*2, conv_dim*4, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn5 = nn.BatchNorm3d(conv_dim*4)
        self.tp_conv6 = nn.Conv3d(conv_dim*4, conv_dim*4, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn6 = nn.BatchNorm3d(conv_dim*4)  

        self.tp_conv7 = nn.Conv3d(conv_dim*4, conv_dim*8, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn7 = nn.BatchNorm3d(conv_dim*8)
        self.tp_conv8 = nn.Conv3d(conv_dim*8, conv_dim*8, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn8 = nn.BatchNorm3d(conv_dim*8)  

        self.rbn = nn.Conv3d(conv_dim*8, conv_dim*8, kernel_size=3, stride=1, padding=1, bias=True)

        #Decoder
        self.tp_conv9 = nn.ConvTranspose3d(conv_dim*8, conv_dim*4, kernel_size=3, stride=2, padding=0, output_padding=1, bias=True)
        self.bn9 = nn.BatchNorm3d(conv_dim*8)
        self.tp_conv10 = nn.Conv3d(conv_dim*8, conv_dim*4, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn10 = nn.BatchNorm3d(conv_dim*4)
        self.tp_conv11 = nn.Conv3d(conv_dim*4, conv_dim*4, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn11 = nn.BatchNorm3d(conv_dim*4)

        self.tp_conv12 = nn.ConvTranspose3d(conv_dim*4, conv_dim*2, kernel_size=3, stride=2, padding=(0, 0, 0), output_padding=(0, 1, 0), bias=True)
        self.bn12 = nn.BatchNorm3d(conv_dim*4)
        self.tp_conv13 = nn.Conv3d(conv_dim*4, conv_dim*2, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn13 = nn.BatchNorm3d(conv_dim*2)
        self.tp_conv14 = nn.Conv3d(conv_dim*2, conv_dim*2, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn14 = nn.BatchNorm3d(conv_dim*2)

        self.tp_conv15 = nn.ConvTranspose3d(conv_dim*2, conv_dim*1, kernel_size=3, stride=2, padding=(0, 0, 0), output_padding=(1, 1, 1), bias=True)
        self.bn15 = nn.BatchNorm3d(conv_dim*2)
        self.tp_conv16 = nn.Conv3d(conv_dim*2, conv_dim*1, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn16 = nn.BatchNorm3d(conv_dim*1)
        self.tp_conv17 = nn.Conv3d(conv_dim*1, conv_dim*1, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn17 = nn.BatchNorm3d(conv_dim*1)


        self.tp_conv18 = nn.Conv3d(conv_dim*1, 1, kernel_size=3, stride=1, padding=1, bias=True)


    def forward(self, z):
   

        h = self.tp_conv1(z)
        h = self.tp_conv2(self.relu(h))

        original3 = h
        h = self.ca1(h) * h
        h = self.sa1(h) * h
        skip3 = h
        h = self.down_sampling(self.relu(original3))



        h = self.tp_conv3(h)
        h = self.tp_conv4(self.relu(h))
  
        original2 = h
        h = self.ca2(h) * h
        h = self.sa2(h) * h     
        skip2 = h
        h = self.down_sampling(self.relu(original2))



        h = self.tp_conv5(h)
        h = self.tp_conv6(self.relu(h))

        original1 = h
        h = self.ca3(h) * h
        h = self.sa3(h) * h   
        skip1 = h
        h = self.down_sampling(self.relu(original1))        



        h = self.tp_conv7(h)
        h = self.tp_conv8(self.relu(h))
        c1 = h       

        #RNB
        h = self.rbn(self.relu(c1))
        h = self.rbn(self.relu(h))
        c2 = h + c1

        h = self.rbn(self.relu(c2))
        h = self.rbn(self.relu(h))
        c3 = h + c2

        h = self.rbn(self.relu(c3))
        h = self.rbn(self.relu(h))
        c4 = h + c3        

        h = self.rbn(self.relu(c4))
        h = self.rbn(self.relu(h))
        c5 = h + c4

        h = self.rbn(self.relu(c5))
        h = self.rbn(self.relu(h))
        c6 = h + c5

        h = self.rbn(self.relu(c6))
        h = self.rbn(self.relu(h))
        c7 = h + c6
        #RBN

        h = self.tp_conv9(self.relu(c7))
        h = torch.cat([h, skip1], 1)
        h = self.relu(h)
        h = self.relu(self.tp_conv10(h))
        h = self.relu(self.tp_conv11(h))

        h = self.tp_conv12(h)
        h = torch.cat([h, skip2], 1)
        h = self.relu(h)
        h = self.relu(self.tp_conv13(h))
        h = self.relu(self.tp_conv14(h))

        h = self.tp_conv15(h)
        h = torch.cat([h, skip3], 1)
        h = self.relu(h)
        h = self.relu(self.tp_conv16(h))
        h = self.relu(self.tp_conv17(h))

        h = self.tp_conv18(h)

        return h

class PAM(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W X Z)
            returns :
                out : attention value + input feature
                attention: B X (HxWxZ) X (HxWxZ)
        """
        m_batchsize, C, height, width, deep = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height*deep).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height*deep)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height*deep)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width, deep)

        out = self.gamma*out + x
        return out

class CAM(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W X Z)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width, deep = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width, deep)

        out = self.gamma*out + x
        return out

class Generator_U_DAM(nn.Module):
    """Generator Unet structure"""

    def __init__(self, conv_dim=8):
        super(Generator_U_DAM, self).__init__()
        self.conv_dim = conv_dim
        self.relu = nn.ReLU()
        self.down_sampling = nn.MaxPool3d(kernel_size=3, stride=2)
        self.ca1 = CAM(8)
        self.pa = PAM(64)
        self.ca2 = CAM(16)
        self.pa2 = PAM(16)
        self.ca3 = CAM(32)
        self.pa3 = PAM(32)

        #Encoder
        self.tp_conv1 = nn.Conv3d(1, conv_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm3d(conv_dim)
        self.tp_conv2 = nn.Conv3d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm3d(conv_dim)

        self.tp_conv3 = nn.Conv3d(conv_dim, conv_dim*2, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm3d(conv_dim*2)
        self.tp_conv4 = nn.Conv3d(conv_dim*2, conv_dim*2, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn4 = nn.BatchNorm3d(conv_dim*2)  

        self.tp_conv5 = nn.Conv3d(conv_dim*2, conv_dim*4, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn5 = nn.BatchNorm3d(conv_dim*4)
        self.tp_conv6 = nn.Conv3d(conv_dim*4, conv_dim*4, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn6 = nn.BatchNorm3d(conv_dim*4)  

        self.tp_conv7 = nn.Conv3d(conv_dim*4, conv_dim*8, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn7 = nn.BatchNorm3d(conv_dim*8)
        self.tp_conv8 = nn.Conv3d(conv_dim*8, conv_dim*8, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn8 = nn.BatchNorm3d(conv_dim*8)  

        self.rbn = nn.Conv3d(conv_dim*8, conv_dim*8, kernel_size=3, stride=1, padding=1, bias=True)

        #Decoder
        self.tp_conv9 = nn.ConvTranspose3d(conv_dim*8, conv_dim*4, kernel_size=3, stride=2, padding=0, output_padding=1, bias=True)
        self.bn9 = nn.BatchNorm3d(conv_dim*8)
        self.tp_conv10 = nn.Conv3d(conv_dim*8, conv_dim*4, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn10 = nn.BatchNorm3d(conv_dim*4)
        self.tp_conv11 = nn.Conv3d(conv_dim*4, conv_dim*4, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn11 = nn.BatchNorm3d(conv_dim*4)

        self.tp_conv12 = nn.ConvTranspose3d(conv_dim*4, conv_dim*2, kernel_size=3, stride=2, padding=(0, 0, 0), output_padding=(0, 1, 0), bias=True)
        self.bn12 = nn.BatchNorm3d(conv_dim*4)
        self.tp_conv13 = nn.Conv3d(conv_dim*4, conv_dim*2, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn13 = nn.BatchNorm3d(conv_dim*2)
        self.tp_conv14 = nn.Conv3d(conv_dim*2, conv_dim*2, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn14 = nn.BatchNorm3d(conv_dim*2)

        self.tp_conv15 = nn.ConvTranspose3d(conv_dim*2, conv_dim*1, kernel_size=3, stride=2, padding=(0, 0, 0), output_padding=(1, 1, 1), bias=True)
        self.bn15 = nn.BatchNorm3d(conv_dim*2)
        self.tp_conv16 = nn.Conv3d(conv_dim*2, conv_dim*1, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn16 = nn.BatchNorm3d(conv_dim*1)
        self.tp_conv17 = nn.Conv3d(conv_dim*1, conv_dim*1, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn17 = nn.BatchNorm3d(conv_dim*1)


        self.tp_conv18 = nn.Conv3d(conv_dim*1, 1, kernel_size=3, stride=1, padding=1, bias=True)



        self.conva1 = nn.Sequential(nn.Conv3d(conv_dim, conv_dim, 3, padding=1, bias=False),
                                   nn.BatchNorm3d(conv_dim),
                                   nn.ReLU())
        self.convb1 = nn.Sequential(nn.Conv3d(conv_dim, conv_dim, 3, padding=1, bias=False),
                                   nn.BatchNorm3d(conv_dim),
                                   nn.ReLU())
        self.conv_level1 = nn.Sequential(nn.Conv3d(conv_dim, conv_dim, 1))


        self.conva2 = nn.Sequential(nn.Conv3d(conv_dim*2, conv_dim*2, 3, padding=1, bias=False),
                                   nn.BatchNorm3d(conv_dim*2),
                                   nn.ReLU())
        self.convb2 = nn.Sequential(nn.Conv3d(conv_dim*2, conv_dim*2, 3, padding=1, bias=False),
                                   nn.BatchNorm3d(conv_dim*2),
                                   nn.ReLU())
        self.conv_level2 = nn.Sequential(nn.Conv3d(conv_dim*2, conv_dim*2, 1))


        self.conva3 = nn.Sequential(nn.Conv3d(conv_dim*4, conv_dim*4, 3, padding=1, bias=False),
                                   nn.BatchNorm3d(conv_dim*4),
                                   nn.ReLU())
        self.convb3 = nn.Sequential(nn.Conv3d(conv_dim*4, conv_dim*4, 3, padding=1, bias=False),
                                   nn.BatchNorm3d(conv_dim*4),
                                   nn.ReLU())
        self.conv_level3 = nn.Sequential(nn.Conv3d(conv_dim*4, conv_dim*4, 1))

        self.convpa = nn.Sequential(nn.Conv3d(conv_dim*8, conv_dim*8, 3, padding=1, bias=False),
                                   nn.BatchNorm3d(conv_dim*8),
                                   nn.ReLU())

    def forward(self, z):
   

        h = self.tp_conv1(z)
        h = self.tp_conv2(self.relu(h))
        skip3 = h
        # ca_feat1 = self.conva1(original3)
        # ca_feat1 = self.ca1(ca_feat1) 
        # pa_feat1 = self.convb1(original3)
        # pa_feat1 = self.pa1(pa_feat1)
        # feat_sum1 = ca_feat1 + pa_feat1
        # skip3 = self.conv_level1(ca_feat1)

        h = self.down_sampling(self.relu(skip3))




        h = self.tp_conv3(h)
        h = self.tp_conv4(self.relu(h))
        skip2 = h
        # ca_feat2 = self.conva2(original2)
        # ca_feat2 = self.ca2(ca_feat2) 
        # pa_feat2 = self.convb2(original2)
        # pa_feat2 = self.pa2(pa_feat2)
        # feat_sum2 = ca_feat2 + pa_feat2
        # skip2 = self.conv_level2(ca_feat2)

        h = self.down_sampling(self.relu(skip2))




        h = self.tp_conv5(h)
        h = self.tp_conv6(self.relu(h))

        skip1 = h
        # ca_feat3 = self.conva3(original1)
        # ca_feat3 = self.ca3(ca_feat3) 
        # pa_feat3 = self.convb1(original1)
        # pa_feat3 = self.pa3(pa_feat3)
        # feat_sum3 = ca_feat3 + pa_feat3
        # skip1 = self.conv_level3(ca_feat3)
        h = self.down_sampling(self.relu(skip1))        

        h = self.tp_conv7(h)
        h = self.tp_conv8(self.relu(h))
        # c1 = h       




        pa_feat = self.convpa(h)
        pa_feat = self.pa(pa_feat)





        h = self.tp_conv9(self.relu(pa_feat))
        h = torch.cat([h, skip1], 1)
        h = self.relu(h)
        h = self.relu(self.tp_conv10(h))
        h = self.relu(self.tp_conv11(h))

        h = self.tp_conv12(h)
        h = torch.cat([h, skip2], 1)
        h = self.relu(h)
        h = self.relu(self.tp_conv13(h))
        h = self.relu(self.tp_conv14(h))

        h = self.tp_conv15(h)
        h = torch.cat([h, skip3], 1)
        h = self.relu(h)
        h = self.relu(self.tp_conv16(h))
        h = self.relu(self.tp_conv17(h))

        h = self.tp_conv18(h)

        return h
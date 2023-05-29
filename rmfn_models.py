import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math



class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm_1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu_1', nn.ReLU(inplace=True))
        self.add_module('conv_1',
                        nn.Conv3d(
                            num_input_features,
                            bn_size * growth_rate,
                            kernel_size=1,
                            stride=1,
                            bias=False))
        self.add_module('norm_2', nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module('relu_2', nn.ReLU(inplace=True))
        self.add_module('conv_2',
                        nn.Conv3d(
                            bn_size * growth_rate,
                            growth_rate,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False))
        self.drop_rate = drop_rate
        self.dropout = nn.Dropout3d(self.drop_rate)

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = self.dropout(new_features)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
                 drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv',
                        nn.Conv3d(
                            num_input_features,
                            num_output_features,
                            kernel_size=1,
                            stride=1,
                            bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))

class _Attention(nn.Sequential):
    def __init__(self, num_input_features):
        super(_Attention, self).__init__()

        self.add_module('norm_1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu_1', nn.ReLU(inplace=True))
        self.add_module('conv_1',
                        nn.Conv3d(
                            num_input_features,
                            num_input_features,
                            kernel_size=1,
                            stride=1,
                            bias=False))
        self.add_module('norm_2', nn.BatchNorm3d(num_input_features))
        self.add_module('relu_2', nn.ReLU(inplace=True))
        self.add_module('conv_2',
                        nn.Conv3d(
                            num_input_features,
                            num_input_features,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False))
    def forward(self, x):
        features = super(_Attention, self).forward(x)
        output = nn.Sigmoid()(features)
        return output

class DenseNet(nn.Module):
    def __init__(self,
                 #sample_size,
                 #sample_duration,
                 growth_rate=32,
                 block_config=(6, 12, 24, 16),
                 num_init_features=64,
                 bn_size=4,
                 drop_rate=0,
                 num_classes=1000):

        super(DenseNet, self).__init__()


        # First convolution
        self.features = nn.Sequential(
            OrderedDict([
                ('conv0',
                 nn.Conv3d(
                     1,
                     num_init_features,
                     kernel_size=3,     #7
                     stride=(1, 1, 1),  #2
                     padding=(1, 1, 1), #3
                     bias=False)),
                ('norm0', nn.BatchNorm3d(num_init_features)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
            ]))
       
        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):                    
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))
        self.features.add_module('relu', nn.ReLU(inplace=True))
        self.features.add_module('pool', nn.MaxPool3d(kernel_size=(4, 5, 4), stride=2))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # Linear layer
        self.classifier = nn.Linear(1*1*1*num_features, num_classes)  #num_features
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        features = self.features(x)
        out = features.view(features.size(0), -1)
        out = self.classifier(out)
        return out

class DCN(nn.Module):
    
    def __init__(self):
        super(DCN, self).__init__()
        growth_rate=16
        num_layers = 2 
        num_init_features=16
        bn_size=4
        drop_rate=0
        num_classes=2

        # First convolution
        self.features0 = nn.Sequential(
            OrderedDict([
                ('conv0',
                 nn.Conv3d(
                     1,
                     num_init_features,
                     kernel_size=3,     #7
                     stride=(1, 1, 1),  #2
                     padding=(1, 1, 1), #3
                     bias=False)),
                ('norm0', nn.BatchNorm3d(num_init_features)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
            ]))

        # pathwise transferblock1
        self.transeferblock1 = nn.Sequential(
            OrderedDict([            
                ('conv0',
                 nn.Conv3d(
                     num_init_features * 2,
                     num_init_features * 2,
                     kernel_size=1,     
                     stride=1,  
                     padding=0,
                     bias=False)),
                ('norm0', nn.BatchNorm3d(num_init_features * 2)),
                ('relu0', nn.ReLU(inplace=True)),
                ('conv1',
                 nn.Conv3d(
                     num_init_features * 2,
                     num_init_features,
                     kernel_size=3,     
                     stride=1,  
                     padding=1,
                     bias=False)),
            ]))        

        # denseblock 1 
        num_features = num_init_features                 
        block1 = _DenseBlock(
            num_layers=num_layers,
            num_input_features=num_features,
            bn_size=bn_size,
            growth_rate=growth_rate,
            drop_rate=drop_rate)
        self.denseblock1 = nn.Sequential(
                                OrderedDict([
                                                ('denseblock1', block1)
                                            ]))         
        num_features = num_features + num_layers * growth_rate
        trans1 = _Transition(
            num_input_features=num_features,
            num_output_features=num_features // 2)
        self.denseblock1.add_module('transition1', trans1)
        num_features = num_features // 2

        # pathwise transferblock2
        self.transeferblock2 = nn.Sequential(
            OrderedDict([
                ('conv0',
                 nn.Conv3d(
                     num_features * 2,
                     num_features * 2,
                     kernel_size=1,     
                     stride=1,  
                     padding=0,
                     bias=False)),
                ('norm0', nn.BatchNorm3d(num_features * 2)),
                ('relu0', nn.ReLU(inplace=True)),
                ('conv1',
                 nn.Conv3d(
                     num_features * 2,
                     num_features,
                     kernel_size=3,     
                     stride=1,  
                     padding=1,
                     bias=False)),
            ])) 
 
        # denseblock 2 
        num_features = num_features                  
        block2 = _DenseBlock(
            num_layers=num_layers,
            num_input_features=num_features,
            bn_size=bn_size,
            growth_rate=growth_rate,
            drop_rate=drop_rate)
        self.denseblock2 = nn.Sequential(
                                OrderedDict([
                                                ('denseblock2', block2)
                                            ]))         
        num_features = num_features + num_layers * growth_rate
        trans2 = _Transition(
            num_input_features=num_features,
            num_output_features=num_features // 2)
        self.denseblock2.add_module('transition2', trans2)
        num_features = num_features // 2
        
        # pathwise transferblock3
        self.transeferblock3 = nn.Sequential(
            OrderedDict([            
                ('conv0',
                 nn.Conv3d(
                     num_features * 2,
                     num_features * 2,
                     kernel_size=1,     
                     stride=1,  
                     padding=0,
                     bias=False)),
                ('norm0', nn.BatchNorm3d(num_features * 2)),
                ('relu0', nn.ReLU(inplace=True)),
                ('conv1',
                 nn.Conv3d(
                     num_features * 2,
                     num_features,
                     kernel_size=3,     
                     stride=1,  
                     padding=1,
                     bias=False)),
            ])) 

        # denseblock 3 
        num_features = num_features                
        block3 = _DenseBlock(
            num_layers=num_layers,
            num_input_features=num_features,
            bn_size=bn_size,
            growth_rate=growth_rate,
            drop_rate=drop_rate)
        self.denseblock3 = nn.Sequential(
                                OrderedDict([
                                                ('denseblock3', block3)
                                            ]))        
        num_features = num_features + num_layers * growth_rate
        trans3 = _Transition(
            num_input_features=num_features,
            num_output_features=num_features // 2)
        self.denseblock3.add_module('transition3', trans3)
        num_features = num_features // 2


        # denseblock 4 
        num_features = num_features                
        block4 = _DenseBlock(
            num_layers=num_layers,
            num_input_features=num_features,
            bn_size=bn_size,
            growth_rate=growth_rate,
            drop_rate=drop_rate)
        self.denseblock4 = nn.Sequential(
                                OrderedDict([
                                                ('denseblock4', block4)
                                            ]))        
        num_features = num_features + num_layers * growth_rate
        # trans4 = _Transition(
        #     num_input_features=num_features,
        #     num_output_features=num_features // 2)
        # self.denseblock4.add_module('transition4', trans4)
        # num_features = num_features // 2



        self.features_end = nn.Sequential(
                                OrderedDict([
                                                # ('conv0', nn.Conv3d(num_features, num_features, kernel_size=3, stride=2, padding=0, bias=False)),
                                                ('norm1', nn.BatchNorm3d(num_features)),
                                                ('relu1', nn.ReLU(inplace=True)),
                                                ('pool1', nn.MaxPool3d(kernel_size=(4, 5, 4), stride=2))
                                            ])) 



        # Linear layer
        self.classifier = nn.Linear(1*1*1*num_features, num_classes) 

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        #1
        features_mri = self.features0(x)
        output1_mri = self.denseblock1(features_mri)

        #2
        output2_mri = self.denseblock2(output1_mri)

        #3 
        output3_mri = self.denseblock3(output2_mri)

        #4
        output4_mri = self.denseblock4(output3_mri)

        output_end = self.features_end(output4_mri)
        output = output_end.view(output_end.size(0), -1)
        output = self.classifier(output)
        
        return output1_mri, output2_mri, output3_mri, output_end


class PT_DCN(nn.Module):
    
    def __init__(self):
        super(PT_DCN, self).__init__()
        growth_rate=16
        num_layers = 2 
        num_init_features=16
        bn_size=4
        drop_rate=0
        num_classes=2

        # First convolution
        self.features0 = nn.Sequential(
            OrderedDict([
                ('conv0',
                 nn.Conv3d(
                     1,
                     num_init_features,
                     kernel_size=3,     #7
                     stride=(1, 1, 1),  #2
                     padding=(1, 1, 1), #3
                     bias=False)),
                ('norm0', nn.BatchNorm3d(num_init_features)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
            ]))

        # pathwise transferblock1
        self.transeferblock1 = nn.Sequential(
            OrderedDict([            
                ('conv0',
                 nn.Conv3d(
                     num_init_features * 2,
                     num_init_features * 2,
                     kernel_size=1,     
                     stride=1,  
                     padding=0,
                     bias=False)),
                ('norm0', nn.BatchNorm3d(num_init_features * 2)),
                ('relu0', nn.ReLU(inplace=True)),
                ('conv1',
                 nn.Conv3d(
                     num_init_features * 2,
                     num_init_features,
                     kernel_size=3,     
                     stride=1,  
                     padding=1,
                     bias=False)),
            ]))        

        # denseblock 1 
        num_features = num_init_features                 
        block1 = _DenseBlock(
            num_layers=num_layers,
            num_input_features=num_features,
            bn_size=bn_size,
            growth_rate=growth_rate,
            drop_rate=drop_rate)
        self.denseblock1 = nn.Sequential(
                                OrderedDict([
                                                ('denseblock1', block1)
                                            ]))         
        num_features = num_features + num_layers * growth_rate
        trans1 = _Transition(
            num_input_features=num_features,
            num_output_features=num_features // 2)
        self.denseblock1.add_module('transition1', trans1)
        num_features = num_features // 2

        # pathwise transferblock2
        self.transeferblock2 = nn.Sequential(
            OrderedDict([
                ('conv0',
                 nn.Conv3d(
                     num_features * 2,
                     num_features * 2,
                     kernel_size=1,     
                     stride=1,  
                     padding=0,
                     bias=False)),
                ('norm0', nn.BatchNorm3d(num_features * 2)),
                ('relu0', nn.ReLU(inplace=True)),
                ('conv1',
                 nn.Conv3d(
                     num_features * 2,
                     num_features,
                     kernel_size=3,     
                     stride=1,  
                     padding=1,
                     bias=False)),
            ])) 
 
        # denseblock 2 
        num_features = num_features * 2               
        block2 = _DenseBlock(
            num_layers=num_layers,
            num_input_features=num_features,
            bn_size=bn_size,
            growth_rate=growth_rate,
            drop_rate=drop_rate)
        self.denseblock2 = nn.Sequential(
                                OrderedDict([
                                                ('denseblock2', block2)
                                            ]))         
        num_features = num_features + num_layers * growth_rate
        trans2 = _Transition(
            num_input_features=num_features,
            num_output_features=num_features // 2)
        self.denseblock2.add_module('transition2', trans2)
        num_features = num_features // 2
        
        # pathwise transferblock3
        self.transeferblock3 = nn.Sequential(
            OrderedDict([            
                ('conv0',
                 nn.Conv3d(
                     num_features * 2,
                     num_features * 2,
                     kernel_size=1,     
                     stride=1,  
                     padding=0,
                     bias=False)),
                ('norm0', nn.BatchNorm3d(num_features * 2)),
                ('relu0', nn.ReLU(inplace=True)),
                ('conv1',
                 nn.Conv3d(
                     num_features * 2,
                     num_features,
                     kernel_size=3,     
                     stride=1,  
                     padding=1,
                     bias=False)),
            ])) 

        # denseblock 3 
        # num_features = num_features * 2              
        block3 = _DenseBlock(
            num_layers=num_layers,
            num_input_features=68,
            bn_size=bn_size,
            growth_rate=growth_rate,
            drop_rate=drop_rate)
        self.denseblock3 = nn.Sequential(
                                OrderedDict([
                                                ('denseblock2', block3)
                                            ]))         
        # num_features = 68 + num_layers * growth_rate
        trans3 = _Transition(
            num_input_features=100,
            num_output_features=100 // 2)
        self.denseblock3.add_module('transition3', trans3)
        num_features = 100 // 2


        # denseblock 4 
        # num_features = num_features * 2               
        block4 = _DenseBlock(
            num_layers=num_layers,
            num_input_features=80,
            bn_size=bn_size,
            growth_rate=growth_rate,
            drop_rate=drop_rate)
        self.denseblock4 = nn.Sequential(
                                OrderedDict([
                                                ('denseblock4', block4)
                                            ]))        
        # num_features = num_features + num_layers * growth_rate



        self.features_end = nn.Sequential(
                                OrderedDict([
                                                # ('conv0', nn.Conv3d(num_features, num_features, kernel_size=3, stride=2, padding=0, bias=False)),
                                                ('norm1', nn.BatchNorm3d(112)),
                                                ('relu1', nn.ReLU(inplace=True)),
                                                ('pool1', nn.MaxPool3d(kernel_size=(4, 5, 4), stride=2))
                                            ])) 



        # Linear layer
        self.classifier = nn.Linear(1*1*1*174, num_classes) 

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        self.sigmoid = nn.Sigmoid()


    def forward(self, x, y):
        #1
        # features_mri = self.features0(x)
        features_pet = self.features0(y)

        # output1 = self.sigmoid(features_pet) * features_mri * 0.3
        # mri1 = features_mri + output1
        # concat1 = torch.cat([features_mri, mri1], 1)
        # concat1_ = torch.cat([features_pet, features_pet], 1)

        # output1_mri = self.denseblock1(mri1)
        output1_pet = self.denseblock1(features_pet)

        #2
        # output2 = self.sigmoid(output1_pet) * output1_mri * 0.3
        # mri2 = output1_mri + output2
        concat2 = torch.cat([x[0], output1_pet], 1)
        # concat2_ = torch.cat([output1_pet, output1_pet], 1)


        # output2_mri = self.denseblock2(mri2)
        output2_pet = self.denseblock2(concat2)

        #3 
        # output3 = self.sigmoid(output2_pet) * output2_mri * 0.3
        # mri3 = output2_mri + output3
        concat3 = torch.cat([x[1], output2_pet], 1)
        # concat3_ = torch.cat([output2_pet, output2_pet], 1)

        # output3_mri = self.denseblock3(mri3)
        output3_pet = self.denseblock3(concat3)

        #4
        concat4 = torch.cat([x[2], output3_pet], 1)
        # output1 = self.features_end(output3_mri)
        output4_pet = self.denseblock4(concat4)
        output = self.features_end(output4_pet)

        concat = torch.cat([x[3], output], 1)


        output = concat.view(concat.size(0), -1)
        output = self.classifier(output)
        
        return output
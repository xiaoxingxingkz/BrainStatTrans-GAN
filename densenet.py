import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math

__all__ = ['DenseNet', 'densenet21', 'densenet121', 'densenet169', 'densenet201', 'densenet264', 'DiscriminatorC']

def DiscriminatorC(**kwargs):
    model = DiscriminatorT()
    return model


def densenet21(**kwargs):
    model = DenseNet(
        num_classes=2,
        num_init_features=16,
        growth_rate=16,
        #drop_rate=0.1,
        block_config=(2, 2, 2, 2),
        **kwargs)
    return model


def densenet121(**kwargs):
    model = DenseNet(
        num_init_features=64,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        **kwargs)
    return model


def densenet169(**kwargs):
    model = DenseNet(
        num_init_features=64,
        growth_rate=32,
        block_config=(6, 12, 32, 32),
        **kwargs)
    return model


def densenet201(**kwargs):
    model = DenseNet(
        num_init_features=64,
        growth_rate=32,
        block_config=(6, 12, 48, 32),
        **kwargs)
    return model


def densenet264(**kwargs):
    model = DenseNet(
        num_init_features=64,
        growth_rate=32,
        block_config=(6, 12, 64, 48),
        **kwargs)
    return model


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('denseblock{}'.format(i))
        ft_module_names.append('transition{}'.format(i))
    ft_module_names.append('norm5')
    ft_module_names.append('classifier')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


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
            # new_features = F.dropout(
            #     new_features, p=self.drop_rate, training=self.training)
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
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))  # nn.MaxPool3d nn.AvgPool3d


class DenseNet(nn.Module):
    """Densenet-BC model class
    Args:
        growth_rate (int) - how many filters to add each layer (k in paper)
        block_config (li05st of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

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

        #self.sample_size = sample_size
        #self.sample_duration = sample_duration

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
        out = features
        out = out.view(features.size(0), -1)
        out = self.classifier(out)
        return out

class DiscriminatorT(nn.Module):

    def __init__(self, conv_dim=32, out_class=2):
        super(DiscriminatorT, self).__init__()
        self.conv_dim =conv_dim

        self.conv1 = nn.Conv3d(1, conv_dim, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn1 =nn.BatchNorm3d(conv_dim)

        self.conv2 = nn.Conv3d(conv_dim, conv_dim*2, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn2 = nn.BatchNorm3d(conv_dim*2)

        self.conv3 = nn.Conv3d(conv_dim*2, conv_dim*4, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn3 = nn.BatchNorm3d(conv_dim*4)

        self.conv4 = nn.Conv3d(conv_dim*4, conv_dim*8, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn4 = nn.BatchNorm3d(conv_dim*8) 

        self.conv5 = nn.Conv3d(conv_dim*8, conv_dim*16, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn5 = nn.BatchNorm3d(conv_dim*16) 

        self.conv6 = nn.Conv3d(conv_dim*16, conv_dim*16, kernel_size=3, stride=1, padding=0, bias=True)


        self.classifier = nn.Linear(conv_dim*16, out_class)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        h = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.2)
        h = F.leaky_relu(self.bn2(self.conv2(h)), negative_slope=0.2)
        h = F.leaky_relu(self.bn3(self.conv3(h)), negative_slope=0.2)
        h = F.leaky_relu(self.bn4(self.conv4(h)), negative_slope=0.2)
        h = F.leaky_relu(self.bn5(self.conv5(h)), negative_slope=0.2)
        h = self.conv6(h)
        output = h.view(h.size()[0], -1)
        predict = self.classifier(output)

        return predict
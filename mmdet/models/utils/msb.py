# coding=utf-8
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _pair

class _ConvNd(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

class MyConv2d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(MyConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    def forward(self, input, padding=0, dilation=1):
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        padding, dilation, self.groups)

class SharedConv(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(SharedConv, self).__init__()
        self.atrous_conv = MyConv2d(inplanes, planes,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=padding,
                                    dilation=dilation,
                                    bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self._init_weight()

    def forward(self, x, padding=0, dilation=1):
        x = self.atrous_conv(x, padding=padding, dilation=dilation)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x
    
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)
        return x * scale
    
class MSB(nn.Module):
    def __init__(self, in_channels, out_channels, dilations, channel_attention=True, sptial_attention=True):
        super(MSB, self).__init__()
        self.dilations = dilations
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU(inplace=True))
        self.sc = SharedConv(in_channels, out_channels, 3, padding=1, dilation=1)
#         self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=dilations[0], dilation=dilations[0], bias=False),
#                                     nn.BatchNorm2d(out_channels),
#                                     nn.ReLU(inplace=True))
#         self.conv5 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=dilations[1], dilation=dilations[1], bias=False),
#                                     nn.BatchNorm2d(out_channels),
#                                     nn.ReLU(inplace=True))
#         self.conv7 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=dilations[2], dilation=dilations[2], bias=False),
#                                     nn.BatchNorm2d(out_channels),
#                                     nn.ReLU(inplace=True))
        
        self.fusion = nn.Sequential(nn.Conv2d(out_channels * 4, out_channels, 1, stride=1, bias=False),
                                    nn.BatchNorm2d(out_channels))
        self.channel_attention = channel_attention
        if self.channel_attention:
            self.se = SEModule(out_channels * 4, 16)
        self.sptial_attention = sptial_attention
        if self.sptial_attention:
            self.sptial = SpatialGate()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        outputs = [x1,]
        for dilation in self.dilations:
            output = self.sc(x, padding=dilation, dilation=dilation)
            outputs.append(output)
#         outputs.append(self.conv3(x)) 
#         outputs.append(self.conv5(x)) 
#         outputs.append(self.conv5(x)) 
        
        outputs = torch.cat(outputs, dim=1)
        if self.channel_attention:
            outputs = self.se(outputs)
        outputs = self.fusion(outputs)
        if self.sptial_attention:
            outputs = self.sptial(outputs)
        outputs = self.relu(outputs)

        return outputs

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

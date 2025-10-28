import math
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
from PIL import Image
from torch.cuda import amp

from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords, increment_path, xyxy2xywh
from utils.plots import color_list, plot_one_box
from utils.torch_utils import time_synchronized

from timm.models.layers import DropPath
from models.SE import SEAttention
from models.cooratt import CoordAtt
from models.attention import CBAM

##### basic ####

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class MP(nn.Module):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)


class SP(nn.Module):
    def __init__(self, k=3, s=1):
        super(SP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=s, padding=k // 2)

    def forward(self, x):
        return self.m(x)
    
    
class ReOrg(nn.Module):
    def __init__(self):
        super(ReOrg, self).__init__()

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        # print("ĐÂY NÀY",len(x), x[0].size(), x[1].size())
        return torch.cat(x, self.d)

class ConcatSE(nn.Module):
    def __init__(self, channel,dimension=1):
        super(ConcatSE, self).__init__()
        self.d = dimension
        self.se = CBAM(channel)

    def forward(self, x):
        return self.se(torch.cat(x, self.d))
    

class Chuncat(nn.Module):
    def __init__(self, dimension=1):
        super(Chuncat, self).__init__()
        self.d = dimension

    def forward(self, x):
        x1 = []
        x2 = []
        for xi in x:
            xi1, xi2 = xi.chunk(2, self.d)
            x1.append(xi1)
            x2.append(xi2)
        return torch.cat(x1+x2, self.d)


class Shortcut(nn.Module):
    def __init__(self, dimension=0):
        super(Shortcut, self).__init__()
        self.d = dimension

    def forward(self, x):
        return x[0]+x[1]


class Foldcut(nn.Module):
    def __init__(self, dimension=0):
        super(Foldcut, self).__init__()
        self.d = dimension

    def forward(self, x):
        x1, x2 = x.chunk(2, self.d)
        return x1+x2

class AddCoords(nn.Module):
    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret
### Convolution layers ###

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
    

class RobustConv(nn.Module):
    # Robust convolution (use high kernel size 7-11 for: downsampling and other layers). Train for 300 - 450 epochs.
    def __init__(self, c1, c2, k=7, s=1, p=None, g=1, act=True, layer_scale_init_value=1e-6):  # ch_in, ch_out, kernel, stride, padding, groups
        super(RobustConv, self).__init__()
        self.conv_dw = Conv(c1, c1, k=k, s=s, p=p, g=c1, act=act)
        self.conv1x1 = nn.Conv2d(c1, c2, 1, 1, 0, groups=1, bias=True)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(c2)) if layer_scale_init_value > 0 else None

    def forward(self, x):
        x = x.to(memory_format=torch.channels_last)
        x = self.conv1x1(self.conv_dw(x))
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1)) 
        return x


class RobustConv2(nn.Module):
    # Robust convolution 2 (use [32, 5, 2] or [32, 7, 4] or [32, 11, 8] for one of the paths in CSP).
    def __init__(self, c1, c2, k=7, s=4, p=None, g=1, act=True, layer_scale_init_value=1e-6):  # ch_in, ch_out, kernel, stride, padding, groups
        super(RobustConv2, self).__init__()
        self.conv_strided = Conv(c1, c1, k=k, s=s, p=p, g=c1, act=act)
        self.conv_deconv = nn.ConvTranspose2d(in_channels=c1, out_channels=c2, kernel_size=s, stride=s, 
                                              padding=0, bias=True, dilation=1, groups=1
        )
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(c2)) if layer_scale_init_value > 0 else None

    def forward(self, x):
        x = self.conv_deconv(self.conv_strided(x))
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1)) 
        return x
    

def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)

class DWSBConv(nn.Module):
    """Depth-wise with batchnorm"""
    def __init__(self, c1, c2, k=3, s=1,act=True):
        super(DWSBConv, self).__init__()
        self.depthwise = nn.Conv2d(c1, c1, kernel_size=k, stride=s, padding=k//2, groups=c1, bias=False)
        self.pointwise = nn.Conv2d(c1, c2, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        out = self.act(self.depthwise(x))
        out = self.act(self.pointwise(out))
        out = self.bn(out)
        return out
    
    def fuseforward(self, x):
        out = self.act(self.depthwise(x))
        out = self.act(self.pointwise(out))
        return out

class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))

class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super(GhostConv, self).__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)
    
class Yolov7_Tiny_E_ELAN(nn.Module):
    def __init__(self, inc, ouc, hidc, act=True):
        super(Yolov7_Tiny_E_ELAN, self).__init__()
        
        self.conv1 = Conv(inc, hidc, k=1, act=act)
        self.conv2 = Conv(inc, hidc, k=1, act=act)
        self.conv3 = Conv(hidc, hidc, k=3, act=act)
        self.conv4 = Conv(hidc, hidc, k=3, act=act)
        self.conv5 = Conv(hidc * 4, ouc, k=1, act=act)
        # self.att = EMA(ouc)
    
    def forward(self, x):
        x1, x2 = self.conv1(x), self.conv2(x)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x_concat = torch.cat([x1, x2, x3, x4], dim=1)
        x_final = self.conv5(x_concat)
        return x_final

class Yolov7_Tiny_SPP(nn.Module):
    def __init__(self, inc, ouc, act=True):
        super(Yolov7_Tiny_SPP, self).__init__()
        
        self.conv1 = Conv(inc, ouc, k=1, act=act)
        self.conv2 = Conv(inc, ouc, k=1, act=act)
        self.sp_5x5, self.sp_9x9, self.sp_13x13 = SP(k=5), SP(k=9), SP(k=13)
        self.conv3 = Conv(ouc * 4, ouc, k=1, act=act)
        self.conv4 = Conv(ouc * 2, ouc, k=1, act=act)
        # self.att = BiLevelRoutingAttention(ouc)
    
    def forward(self, x):
        x1, x2 = self.conv1(x), self.conv2(x)
        sp_5x5 = self.sp_5x5(x2)
        sp_9x9 = self.sp_9x9(x2)
        sp_13x13 = self.sp_13x13(x2)
        sp_concat = torch.cat([x2, sp_5x5, sp_9x9, sp_13x13], dim=1)
        sp_concat = self.conv3(sp_concat)
        sp_concat = torch.cat([sp_concat, x1], dim=1)
        x_final = self.conv4(sp_concat)
        # return self.att(x_final)
        return x_final

class CoordConv(nn.Module):  ### Coordinate
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, with_r=False):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_channels += 2
        if with_r:
            in_channels += 1
        self.conv = Conv(in_channels, out_channels, k=kernel_size, s=stride)

    def forward(self, x):
        x = self.addcoords(x)
        x = self.conv(x)
        return x

class CoordConvATT(nn.Module):  ### Coordinate
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, with_r=False):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_channels += 2
        if with_r:
            in_channels += 1
        self.conv = Conv(in_channels, out_channels, k=kernel_size, s=stride)
        self.cooratt = CoordAtt(out_channels)
    def forward(self, x):
        x = self.addcoords(x)
        x = self.conv(x)
        return self.cooratt(x)

class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x
    
class PConv(nn.Module):
    def __init__(self, dim, ouc, n_div=4, forward='split_cat'):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        self.conv = Conv(dim, ouc, k=1)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        x = self.conv(x)
        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        x = self.conv(x)
        return x
    
class ConvCBAM(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(ConvCBAM, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.att=CBAM(c2)
    def forward(self, x):
        return self.att(self.act(self.bn(self.conv(x))))

    def fuseforward(self, x):
        return self.att(self.act(self.conv(x)))
    
    
class ConvSE(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(ConvSE, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.att=SEAttention(c2,4)
    def forward(self, x):
        return self.att(self.act(self.bn(self.conv(x))))

    def fuseforward(self, x):
        return self.att(self.act(self.conv(x)))

### Block ###

class DGC(nn.Module):
    def __init__(self, c1, c2, n=1, k=2):
        super(DGC, self).__init__()
        c_ = int(c1)  # hidden channels
        self.cv1 = Conv(c_, c2//2, 3, k,g=2)
        self.dwsconv = GhostConv(c_,c2//2,3,k,g=2)
        self.cv2 = Conv(c2,c2,1,1,None,1)
    def forward(self, x):
        return self.cv2(torch.cat((self.cv1(x), self.dwsconv(x)), dim=1))   
    
class MDGC(nn.Module):
    def __init__(self, c1, c2, n=1, k=2):
        super(MDGC, self).__init__()
        c_ = int(c1)  # hidden channels
        self.cv0= Conv(c_,c2//2,3,k,None,g=2)
        self.cv1 = GhostConv(c_, c2//2, 3, k,g=2)
        self.mp=nn.MaxPool2d(kernel_size=2, stride=2)
        cx=c1+c2
        self.cv2=Conv(cx,c2,1,1,None,1)
        
    def forward(self, x):
        return self.cv2(torch.cat((self.cv1(x),self.mp(x),self.cv0(x)),dim=1))



class Stem(nn.Module):
    # Stem
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Stem, self).__init__()
        c_ = int(c2/2)  # hidden channels
        self.cv1 = Conv(c1, c_, 3, 2)
        self.cv2 = Conv(c_, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 2)
        self.pool = torch.nn.MaxPool2d(2, stride=2)
        self.cv4 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x = self.cv1(x)
        return self.cv4(torch.cat((self.cv3(self.cv2(x)), self.pool(x)), dim=1))


class DownC(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, n=1, k=2):
        super(DownC, self).__init__()
        c_ = int(c1)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2//2, 3, k)
        self.cv3 = Conv(c1, c2//2, 1, 1)
        self.mp = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return torch.cat((self.cv2(self.cv1(x)), self.cv3(self.mp(x))), dim=1)


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))
    

class Bottleneck(nn.Module):
    # Darknet bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class BottleC2Fneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class Res(nn.Module):
    # ResNet bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Res, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c_, 3, 1, g=g)
        self.cv3 = Conv(c_, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))


class ResX(Res):
    # ResNet bottleneck
    def __init__(self, c1, c2, shortcut=True, g=32, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__(c1, c2, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels


class Ghost(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super(Ghost, self).__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),  # pw
                                  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
                                  GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)

class Faster_Block(nn.Module):
    def __init__(self,
                 inc,
                 dim,
                 n_div=4,
                 mlp_ratio=2,
                 drop_path=0.1,
                 layer_scale_init_value=0.0,
                 pconv_fw_type='split_cat'
                 ):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer = [
            Conv(dim, mlp_hidden_dim, 1),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            pconv_fw_type
        )
        
        self.adjust_channel = None
        if inc != dim:
            self.adjust_channel = Conv(inc, dim, 1)

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x):
        if self.adjust_channel is not None:
            x = self.adjust_channel(x)
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x):
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class Bottle2neck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, shortcut, baseWidth=8, scale = 5):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: basic width of conv3x3
            scale: number of scale.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth/64.0)))
        self.conv1 = Conv(inplanes, width*scale, k=1)
        
        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        convs = []
        for i in range(self.nums):
          convs.append(Conv(width, width, k=3))
        self.convs = nn.ModuleList(convs)

        self.conv3 = Conv(width*scale, planes * self.expansion, k=1, act=False)

        self.silu = nn.SiLU(inplace=True)
        self.scale = scale
        self.width  = width
        self.shortcut = shortcut

    def forward(self, x):
        
        if self.shortcut:
            residual = x
        out = self.conv1(x)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        if self.scale != 1:
          out = torch.cat((out, spx[self.nums]),1)

        out = self.conv3(out)
        if self.shortcut:
            out += residual
        out = self.silu(out)
        return out


class TridentBlock(nn.Module):
    def __init__(self, c1, c2, stride=1, c=False, e=0.5, padding=[1, 2, 3], dilate=[1, 2, 3], bias=False):
        super(TridentBlock, self).__init__()
        self.stride = stride
        self.c = c
        c_ = int(c2 * e)
        self.padding = padding
        self.dilate = dilate
        self.share_weightconv1 = nn.Parameter(torch.Tensor(c_, c1, 1, 1))
        self.share_weightconv2 = nn.Parameter(torch.Tensor(c2, c_, 3, 3))

        self.bn1 = nn.BatchNorm2d(c_)
        self.bn2 = nn.BatchNorm2d(c2)

        self.act = nn.SiLU()

        nn.init.kaiming_uniform_(self.share_weightconv1, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.share_weightconv2, nonlinearity="relu")

        if bias:
            self.bias = nn.Parameter(torch.Tensor(c2))
        else:
            self.bias = None

        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward_for_small(self, x):
        residual = x
        out = nn.functional.conv2d(x, self.share_weightconv1, bias=self.bias)
        out = self.bn1(out)
        out = self.act(out)

        out = nn.functional.conv2d(out, self.share_weightconv2, bias=self.bias, stride=self.stride, padding=self.padding[0],
                                   dilation=self.dilate[0])
        out = self.bn2(out)
        out += residual
        out = self.act(out)

        return out

    def forward_for_middle(self, x):
        residual = x
        out = nn.functional.conv2d(x, self.share_weightconv1, bias=self.bias)
        out = self.bn1(out)
        out = self.act(out)

        out = nn.functional.conv2d(out, self.share_weightconv2, bias=self.bias, stride=self.stride, padding=self.padding[1],
                                   dilation=self.dilate[1])
        out = self.bn2(out)
        out += residual
        out = self.act(out)

        return out

    def forward_for_big(self, x):
        residual = x
        out = nn.functional.conv2d(x, self.share_weightconv1, bias=self.bias)
        out = self.bn1(out)
        out = self.act(out)

        out = nn.functional.conv2d(out, self.share_weightconv2, bias=self.bias, stride=self.stride, padding=self.padding[2],
                                   dilation=self.dilate[2])
        out = self.bn2(out)
        out += residual
        out = self.act(out)

        return out

    def forward(self, x):
        xm = x
        base_feat = []
        if self.c is not False:
            x1 = self.forward_for_small(x)
            x2 = self.forward_for_middle(x)
            x3 = self.forward_for_big(x)
        else:
            x1 = self.forward_for_small(xm[0])
            x2 = self.forward_for_middle(xm[1])
            x3 = self.forward_for_big(xm[2])

        base_feat.append(x1)
        base_feat.append(x2)
        base_feat.append(x3)

        return base_feat

class RFEM(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5, stride=1):
        super(RFEM, self).__init__()
        c = True
        layers = []
        layers.append(TridentBlock(c1, c2, stride=stride, c=c, e=e))
        c1 = c2
        for i in range(1, n):
            layers.append(TridentBlock(c1, c2))
        self.layer = nn.Sequential(*layers)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        out = self.layer(x)
        out = out[0] + out[1] + out[2] + x
        out = self.act(self.bn(out))
        return out
##### end of basic #####

#### C3 module ####
class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3 module with options for channel count, bottleneck repetition, shortcut usage, group
        convolutions, and expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Performs forward propagation using concatenated outputs from two convolutions and a Bottleneck sequence."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))



class C3x(C3):
    # C3 module with cross-convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3x module with cross-convolutions, extending C3 with customizable channel dimensions, groups,
        and expansion.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3 module with TransformerBlock for enhanced feature extraction, accepts channel sizes, shortcut
        config, group, and expansion.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3SPP(C3):
    # C3 module with SPP()
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        """Initializes a C3 module with SPP layer for advanced spatial feature extraction, given channel sizes, kernel
        sizes, shortcut, group, and expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes YOLOv5's C3 module with Ghost Bottlenecks for efficient feature extraction."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))

class C3_Faster(C3):
    # C3 module with cross-convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Faster_Block(c_, c_) for _ in range(n)))

class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(BottleC2Fneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class C2f_Faster(C2f):
    def __init__(self,c1,c2,n=1,shortcut=False,g=1,e=0.5):
        super().__init__(c1,c2,n,shortcut,g,e)
        c_= int(c2 * e)
        self.m=nn.ModuleList(Faster_Block(c_, c_) for _ in range(n))

class C3_Res2Block(C3):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottle2neck(c_, c_, shortcut) for _ in range(n)))

class C3RFEM(C3):
    # C3 module with RFEM
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(RFEM(c_, c_, n=1, e=e) for _ in range(n)))

##### cspnet #####

class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))

class GhostSPPCSPC(SPPCSPC):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = GhostConv(c1, c_, 1, 1)
        self.cv2 = GhostConv(c1, c_, 1, 1)
        self.cv3 = GhostConv(c_, c_, 3, 1)
        self.cv4 = GhostConv(c_, c_, 1, 1)
        self.cv5 = GhostConv(4 * c_, c_, 1, 1)
        self.cv6 = GhostConv(c_, c_, 3, 1)
        self.cv7 = GhostConv(2 * c_, c2, 1, 1)


class GhostStem(Stem):
    # Stem
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__(c1, c2, k, s, p, g, act)
        c_ = int(c2/2)  # hidden channels
        self.cv1 = GhostConv(c1, c_, 3, 2)
        self.cv2 = GhostConv(c_, c_, 1, 1)
        self.cv3 = GhostConv(c_, c_, 3, 2)
        self.cv4 = GhostConv(2 * c_, c2, 1, 1)
        

class BottleneckCSPA(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSPA, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat((y1, y2), dim=1))


class BottleneckCSPB(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSPB, self).__init__()
        c_ = int(c2)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.m(x1)
        y2 = self.cv2(x1)
        return self.cv3(torch.cat((y1, y2), dim=1))


class BottleneckCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSPC, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 1, 1)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(torch.cat((y1, y2), dim=1))


class ResCSPA(BottleneckCSPA):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[Res(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class ResCSPB(BottleneckCSPB):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2)  # hidden channels
        self.m = nn.Sequential(*[Res(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class ResCSPC(BottleneckCSPC):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[Res(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class ResXCSPA(ResCSPA):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=32, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[Res(c_, c_, shortcut, g, e=1.0) for _ in range(n)])


class ResXCSPB(ResCSPB):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=32, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2)  # hidden channels
        self.m = nn.Sequential(*[Res(c_, c_, shortcut, g, e=1.0) for _ in range(n)])


class ResXCSPC(ResCSPC):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=32, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[Res(c_, c_, shortcut, g, e=1.0) for _ in range(n)])


class GhostCSPA(BottleneckCSPA):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[Ghost(c_, c_) for _ in range(n)])


class GhostCSPB(BottleneckCSPB):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2)  # hidden channels
        self.m = nn.Sequential(*[Ghost(c_, c_) for _ in range(n)])


class GhostCSPC(BottleneckCSPC):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[Ghost(c_, c_) for _ in range(n)])

##### end of cspnet #####


##### yolor #####

class ImplicitA(nn.Module):
    def __init__(self, channel, mean=0., std=.02):
        super(ImplicitA, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit + x
    

class ImplicitM(nn.Module):
    def __init__(self, channel, mean=1., std=.02):
        super(ImplicitM, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit * x
    
##### end of yolor #####


##### repvgg #####

class RepConv(nn.Module):
    # Represented convolution
    # https://arxiv.org/abs/2101.03697

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True, deploy=False):
        super(RepConv, self).__init__()

        self.deploy = deploy
        self.groups = g
        self.in_channels = c1
        self.out_channels = c2

        assert k == 3
        assert autopad(k, p) == 1

        padding_11 = autopad(k, p) - k // 2

        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        if deploy:
            self.rbr_reparam = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)

        else:
            self.rbr_identity = (nn.BatchNorm2d(num_features=c1) if c2 == c1 and s == 1 else None)

            self.rbr_dense = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2),
            )

            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d( c1, c2, 1, s, padding_11, groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2),
            )

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)
    
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return (
            kernel.detach().cpu().numpy(),
            bias.detach().cpu().numpy(),
        )

    def fuse_conv_bn(self, conv, bn):

        std = (bn.running_var + bn.eps).sqrt()
        bias = bn.bias - bn.running_mean * bn.weight / std

        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        weights = conv.weight * t

        bn = nn.Identity()
        conv = nn.Conv2d(in_channels = conv.in_channels,
                              out_channels = conv.out_channels,
                              kernel_size = conv.kernel_size,
                              stride=conv.stride,
                              padding = conv.padding,
                              dilation = conv.dilation,
                              groups = conv.groups,
                              bias = True,
                              padding_mode = conv.padding_mode)

        conv.weight = torch.nn.Parameter(weights)
        conv.bias = torch.nn.Parameter(bias)
        return conv

    def fuse_repvgg_block(self):    
        if self.deploy:
            return
        print(f"RepConv.fuse_repvgg_block")
                
        self.rbr_dense = self.fuse_conv_bn(self.rbr_dense[0], self.rbr_dense[1])
        
        self.rbr_1x1 = self.fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
        rbr_1x1_bias = self.rbr_1x1.bias
        weight_1x1_expanded = torch.nn.functional.pad(self.rbr_1x1.weight, [1, 1, 1, 1])
        
        # Fuse self.rbr_identity
        if (isinstance(self.rbr_identity, nn.BatchNorm2d) or isinstance(self.rbr_identity, nn.modules.batchnorm.SyncBatchNorm)):
            # print(f"fuse: rbr_identity == BatchNorm2d or SyncBatchNorm")
            identity_conv_1x1 = nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=self.groups, 
                    bias=False)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.to(self.rbr_1x1.weight.data.device)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.squeeze().squeeze()
            # print(f" identity_conv_1x1.weight = {identity_conv_1x1.weight.shape}")
            identity_conv_1x1.weight.data.fill_(0.0)
            identity_conv_1x1.weight.data.fill_diagonal_(1.0)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.unsqueeze(2).unsqueeze(3)
            # print(f" identity_conv_1x1.weight = {identity_conv_1x1.weight.shape}")

            identity_conv_1x1 = self.fuse_conv_bn(identity_conv_1x1, self.rbr_identity)
            bias_identity_expanded = identity_conv_1x1.bias
            weight_identity_expanded = torch.nn.functional.pad(identity_conv_1x1.weight, [1, 1, 1, 1])            
        else:
            # print(f"fuse: rbr_identity != BatchNorm2d, rbr_identity = {self.rbr_identity}")
            bias_identity_expanded = torch.nn.Parameter( torch.zeros_like(rbr_1x1_bias) )
            weight_identity_expanded = torch.nn.Parameter( torch.zeros_like(weight_1x1_expanded) )            
        

        #print(f"self.rbr_1x1.weight = {self.rbr_1x1.weight.shape}, ")
        #print(f"weight_1x1_expanded = {weight_1x1_expanded.shape}, ")
        #print(f"self.rbr_dense.weight = {self.rbr_dense.weight.shape}, ")

        self.rbr_dense.weight = torch.nn.Parameter(self.rbr_dense.weight + weight_1x1_expanded + weight_identity_expanded)
        self.rbr_dense.bias = torch.nn.Parameter(self.rbr_dense.bias + rbr_1x1_bias + bias_identity_expanded)
                
        self.rbr_reparam = self.rbr_dense
        self.deploy = True

        if self.rbr_identity is not None:
            del self.rbr_identity
            self.rbr_identity = None

        if self.rbr_1x1 is not None:
            del self.rbr_1x1
            self.rbr_1x1 = None

        if self.rbr_dense is not None:
            del self.rbr_dense
            self.rbr_dense = None


class RepBottleneck(Bottleneck):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__(c1, c2, shortcut=True, g=1, e=0.5)
        c_ = int(c2 * e)  # hidden channels
        self.cv2 = RepConv(c_, c2, 3, 1, g=g)


class RepBottleneckCSPA(BottleneckCSPA):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])


class RepBottleneckCSPB(BottleneckCSPB):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2)  # hidden channels
        self.m = nn.Sequential(*[RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])


class RepBottleneckCSPC(BottleneckCSPC):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])


class RepRes(Res):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__(c1, c2, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv2 = RepConv(c_, c_, 3, 1, g=g)


class RepResCSPA(ResCSPA):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[RepRes(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class RepResCSPB(ResCSPB):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2)  # hidden channels
        self.m = nn.Sequential(*[RepRes(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class RepResCSPC(ResCSPC):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[RepRes(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class RepResX(ResX):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=32, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__(c1, c2, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv2 = RepConv(c_, c_, 3, 1, g=g)


class RepResXCSPA(ResXCSPA):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=32, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[RepResX(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class RepResXCSPB(ResXCSPB):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=32, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2)  # hidden channels
        self.m = nn.Sequential(*[RepResX(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class RepResXCSPC(ResXCSPC):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=32, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[RepResX(c_, c_, shortcut, g, e=0.5) for _ in range(n)])

##### end of repvgg #####


##### transformer #####

class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)

        # remove input_resolution
        self.blocks = nn.Sequential(*[TransformerLayer(dim=c2, num_heads=num_heads) for i in range(num_layers)])

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        x = self.blocks(x)
        return x


class STCSPA(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(STCSPA, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        num_heads = c_ // 32
        self.m = SwinTransformerBlock(c_, c_, num_heads, n)
        #self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat((y1, y2), dim=1))


class STCSPB(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(STCSPB, self).__init__()
        c_ = int(c2)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        num_heads = c_ // 32
        self.m = SwinTransformerBlock(c_, c_, num_heads, n)
        #self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.m(x1)
        y2 = self.cv2(x1)
        return self.cv3(torch.cat((y1, y2), dim=1))


class STCSPC(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(STCSPC, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 1, 1)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        num_heads = c_ // 32
        self.m = SwinTransformerBlock(c_, c_, num_heads, n)
        #self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(torch.cat((y1, y2), dim=1))


##### end of swin transformer #####   

class SPPRFEM(nn.Module):
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, None, 2)
        self.cv2 = Conv(c_ * 5, c2, 1, 1, None, 2)
        # self.cv1 = Conv(c1, c_, 1, 1, None, 1) #test
        # self.cv2 = Conv(c_ * 5, c2, 1, 1, None, 1) #test
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.rfe = RFEM(c_, c_)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
    
        return self.cv2(torch.cat([x, y1, y2, self.m(y2),self.rfe(x)], 1))


class GhostSPPRFEM(nn.Module):
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = GhostConv(c1, c_, 1, 1)
        self.cv2 = GhostConv(c_ * 5, c2, 1, 1)
        self.rfe = RFEM(c_, c_)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
    
        return self.cv2(torch.cat([x, y1, y2, self.m(y2),self.rfe(x)], 1))

# SPP block with conv and groupnorm
class SPPGN(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13), g=16):
        super(SPPGN, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([
            nn.Sequential(
                nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2),
                nn.GroupNorm(g, c_)
            )
            for x in k
        ])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


# SPP block with Shift channel attention SCA
class SPPSCA(nn.Module):
        # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13), g=16):
        super(SPPGN, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([
            nn.Sequential(
                nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2),
                nn.GroupNorm(g, c_),
                nn.Conv2d(c_, c_, kernel_size=1, stride=1, groups=c_),
                nn.ReLU(inplace=True)
            )
            for x in k
        ])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))
    

# SPP block with efficient block architecture
class EffSPP(nn.Module):
        # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13), g=16):
        super(SPPGN, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([
            nn.Sequential(
                nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2),
                nn.GroupNorm(g, c_)
            )
            for x in k
        ])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))
    

class ECA(nn.Module):
    """Efficient Channel Attention module"""
    def __init__(self, channel, gamma=2, b=1):
        super(ECA, self).__init__()
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class ConvECA(nn.Module):
    # Standard convolution with ECA attention
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(ConvECA, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.att = ECA(c2)

    def forward(self, x):
        return self.att(self.act(self.bn(self.conv(x))))

    def fuseforward(self, x):
        return self.att(self.act(self.conv(x)))
    
class DilatedConv(nn.Module):
    """Dilated convolution for enhanced receptive field"""
    def __init__(self, c1, c2, k=3, s=1, d=2, act=True):
        super(DilatedConv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, padding=d, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MultiScaleAggregation(nn.Module):
    """Multi-scale feature aggregation for small objects"""
    def __init__(self, c1, c2):
        super(MultiScaleAggregation, self).__init__()
        self.conv1x1 = Conv(c1, c2//3, 1, 1)
        self.conv3x3 = Conv(c1, c2//3, 3, 1)
        self.dilated_conv = DilatedConv(c1, c2//3, 3, 1, 2)
        self.conv_out = Conv(c2, c2, 1, 1)

    def forward(self, x):
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.dilated_conv(x)
        return self.conv_out(torch.cat([x1, x2, x3], dim=1))


class FPNConnection(nn.Module):
    """FPN-style lateral connection for feature pyramid"""
    def __init__(self, c1, c2):
        super(FPNConnection, self).__init__()
        self.conv = Conv(c1, c2, 1, 1)

    def forward(self, x):
        return self.conv(x)


class DeformableConv2d(nn.Module):
    """Deformable convolution for handling object deformations"""
    def __init__(self, c1, c2, k=3, s=1, p=1, dilation=1, groups=1, bias=False):
        super(DeformableConv2d, self).__init__()
        self.conv_offset = nn.Conv2d(c1, 2 * k * k, k, s, p, dilation=dilation, groups=groups, bias=bias)
        self.conv = DeformConv2d(c1, c2, k, s, p, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        offset = self.conv_offset(x)
        return self.act(self.bn(self.conv(x, offset)))


class RotationInvariantConv(nn.Module):
    """Rotation-invariant convolution using multiple orientations"""
    def __init__(self, c1, c2, k=3, s=1):
        super(RotationInvariantConv, self).__init__()
        self.conv0 = Conv(c1, c2//4, k, s)
        self.conv45 = Conv(c1, c2//4, k, s)
        self.conv90 = Conv(c1, c2//4, k, s)
        self.conv135 = Conv(c1, c2//4, k, s)
        self.conv_out = Conv(c2, c2, 1, 1)

    def forward(self, x):
        # Apply rotations
        x0 = self.conv0(x)
        x45 = self.conv45(torch.rot90(x, 1, [2, 3]))
        x90 = self.conv90(torch.rot90(x, 2, [2, 3]))
        x135 = self.conv135(torch.rot90(x, 3, [2, 3]))

        # Rotate back
        x45 = torch.rot90(x45, 3, [2, 3])
        x90 = torch.rot90(x90, 2, [2, 3])
        x135 = torch.rot90(x135, 1, [2, 3])

        return self.conv_out(torch.cat([x0, x45, x90, x135], dim=1))


class IlluminationNormalizer(nn.Module):
    """Illumination normalization for varying lighting conditions"""
    def __init__(self, c1):
        super(IlluminationNormalizer, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_gamma = nn.Conv2d(c1, c1, 1, 1, bias=False)
        self.conv_beta = nn.Conv2d(c1, c1, 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Compute illumination statistics
        mean = self.global_avg_pool(x)
        gamma = self.sigmoid(self.conv_gamma(mean))
        beta = self.conv_beta(mean)

        # Apply illumination normalization
        return x * gamma + beta

class BiFPNFusion(nn.Module):
    """Bidirectional Feature Pyramid Network fusion"""
    def __init__(self, c1, c2):
        super(BiFPNFusion, self).__init__()
        self.conv_up = Conv(c1, c2, 1, 1)
        self.conv_down = Conv(c1, c2, 1, 1)
        self.conv_out = Conv(c2, c2, 1, 1)

    def forward(self, x_high, x_low):
        # Upsample high-level features
        x_high_up = F.interpolate(self.conv_up(x_high), scale_factor=2, mode='nearest')

        # Downsample low-level features
        x_low_down = F.adaptive_avg_pool2d(self.conv_down(x_low), x_high.shape[2:])

        # Fuse features
        fused = x_high_up + x_low_down
        return self.conv_out(fused)


class CrossScaleAggregation(nn.Module):
    """Cross-scale feature aggregation for small objects"""
    def __init__(self, channels):
        super(CrossScaleAggregation, self).__init__()
        self.channels = channels
        self.convs = nn.ModuleList([
            Conv(c, c, 1, 1) for c in channels
        ])
        self.fusion_conv = Conv(sum(channels), channels[-1], 1, 1)

    def forward(self, features):
        # Apply 1x1 convs to each scale
        scaled_features = [conv(feat) for conv, feat in zip(self.convs, features)]

        # Resize all features to the same size (smallest scale)
        target_size = features[-1].shape[2:]
        resized_features = []
        for feat in scaled_features[:-1]:
            resized_features.append(F.adaptive_avg_pool2d(feat, target_size))
        resized_features.append(scaled_features[-1])

        # Concatenate and fuse
        return self.fusion_conv(torch.cat(resized_features, dim=1))


class FeatureEnhancementModule(nn.Module):
    """Feature Enhancement Module for small object amplification"""
    def __init__(self, c1, c2):
        super(FeatureEnhancementModule, self).__init__()
        self.conv1 = Conv(c1, c2, 1, 1)
        self.conv3 = Conv(c1, c2, 3, 1)
        self.conv5 = Conv(c1, c2, 5, 1)
        self.fusion = Conv(c2*3, c2, 1, 1)
        self.attention = ECA(c2)

    def forward(self, x):
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)

        enhanced = self.fusion(torch.cat([x1, x3, x5], dim=1))
        return self.attention(enhanced)

class NonLocalBlock(nn.Module):
    """Non-local neural network for long-range dependencies"""
    def __init__(self, c1, c2=None):
        super(NonLocalBlock, self).__init__()
        c2 = c2 or c1
        self.theta = Conv(c1, c2//8, 1, 1)
        self.phi = Conv(c1, c2//8, 1, 1)
        self.g = Conv(c1, c2//2, 1, 1)
        self.out_conv = Conv(c2//2, c2, 1, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.shape
        theta = self.theta(x).view(b, -1, h*w).permute(0, 2, 1)  # (b, hw, c//8)
        phi = self.phi(x).view(b, -1, h*w)  # (b, c//8, hw)
        g = self.g(x).view(b, -1, h*w).permute(0, 2, 1)  # (b, hw, c//2)

        attn = self.softmax(torch.bmm(theta, phi))  # (b, hw, hw)
        out = torch.bmm(attn, g).permute(0, 2, 1).view(b, -1, h, w)  # (b, c//2, h, w)

        return self.out_conv(out) + x


class RelationNetwork(nn.Module):
    """Relation Network for modeling object relationships"""
    def __init__(self, c1, c2):
        super(RelationNetwork, self).__init__()
        self.conv1 = Conv(c1, c2, 1, 1)
        self.conv2 = Conv(c1, c2, 1, 1)
        self.relation_conv = nn.Sequential(
            Conv(c2*2, c2, 1, 1),
            Conv(c2, c2, 3, 1),
            Conv(c2, 1, 1, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        feat = self.conv1(x).view(b, -1, h*w).permute(0, 2, 1)  # (b, hw, c)

        # Compute pairwise relations
        feat_i = feat.unsqueeze(2)  # (b, hw, 1, c)
        feat_j = feat.unsqueeze(1)  # (b, 1, hw, c)

        relations = torch.cat([feat_i.expand(-1, -1, h*w, -1),
                              feat_j.expand(-1, h*w, -1, -1)], dim=-1)  # (b, hw, hw, 2c)

        relations = relations.permute(0, 3, 1, 2).view(b*relations.shape[-1], relations.shape[1], h, w)
        relation_scores = self.relation_conv(relations).view(b, 1, h*w, h*w)

        # Apply relation-based attention
        attended = torch.bmm(relation_scores.squeeze(1), feat).permute(0, 2, 1).view(b, -1, h, w)

        return self.conv2(attended) + x


class GraphNeuralNetwork(nn.Module):
    """Simple Graph Neural Network for scene understanding"""
    def __init__(self, c1, c2, num_nodes=64):
        super(GraphNeuralNetwork, self).__init__()
        self.num_nodes = num_nodes
        self.node_conv = Conv(c1, c2, 1, 1)
        self.edge_conv = Conv(c2*2, c2, 1, 1)
        self.update_conv = Conv(c2*2, c2, 1, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        # Extract node features (adaptive pooling to fixed number of nodes)
        nodes = F.adaptive_avg_pool2d(self.node_conv(x), (self.num_nodes, 1))
        nodes = nodes.squeeze(-1).permute(0, 2, 1)  # (b, num_nodes, c)

        # Build adjacency matrix (simple distance-based)
        coords = torch.rand(b, self.num_nodes, 2, device=x.device)  # Random positions for simplicity
        dist = torch.cdist(coords, coords)  # (b, num_nodes, num_nodes)
        adj = torch.exp(-dist / 0.1)  # Gaussian kernel

        # Graph convolution
        node_agg = torch.bmm(adj, nodes)  # (b, num_nodes, c)
        updated_nodes = self.update_conv(torch.cat([nodes, node_agg], dim=-1))

        # Project back to spatial domain
        spatial_feat = updated_nodes.permute(0, 2, 1).unsqueeze(-1)  # (b, c, num_nodes, 1)
        spatial_feat = F.interpolate(spatial_feat, size=(h, w), mode='bilinear', align_corners=False)

        return spatial_feat.squeeze(-1) + x

class VarifocalHead(nn.Module):
    """VarifocalNet head for improved classification and localization"""
    def __init__(self, c1, num_classes, num_anchors):
        super(VarifocalHead, self).__init__()
        self.cls_conv = nn.Sequential(
            Conv(c1, c1, 3, 1),
            Conv(c1, c1, 3, 1),
            Conv(c1, num_classes * num_anchors, 1, 1)
        )
        self.reg_conv = nn.Sequential(
            Conv(c1, c1, 3, 1),
            Conv(c1, c1, 3, 1),
            Conv(c1, 4 * num_anchors, 1, 1)
        )
        self.quality_conv = nn.Sequential(
            Conv(c1, c1, 3, 1),
            Conv(c1, c1, 3, 1),
            Conv(c1, 1 * num_anchors, 1, 1)
        )

    def forward(self, x):
        cls_pred = self.cls_conv(x)
        reg_pred = self.reg_conv(x)
        quality_pred = self.quality_conv(x)
        return cls_pred, reg_pred, quality_pred


class CenterNetHead(nn.Module):
    """CenterNet-style center prediction head"""
    def __init__(self, c1, num_classes):
        super(CenterNetHead, self).__init__()
        self.heatmap_head = nn.Sequential(
            Conv(c1, c1, 3, 1),
            Conv(c1, num_classes, 1, 1)
        )
        self.wh_head = nn.Sequential(
            Conv(c1, c1, 3, 1),
            Conv(c1, 2, 1, 1)
        )
        self.offset_head = nn.Sequential(
            Conv(c1, c1, 3, 1),
            Conv(c1, 2, 1, 1)
        )

    def forward(self, x):
        heatmap = self.heatmap_head(x)
        wh = self.wh_head(x)
        offset = self.offset_head(x)
        return heatmap, wh, offset


class MultiScaleInference(nn.Module):
    """Multi-scale testing module"""
    def __init__(self, model, scales=[0.5, 0.75, 1.0, 1.25, 1.5]):
        super(MultiScaleInference, self).__init__()
        self.model = model
        self.scales = scales

    def forward(self, x):
        predictions = []
        original_size = x.shape[2:]

        for scale in self.scales:
            # Resize input
            scaled_size = (int(original_size[0] * scale), int(original_size[1] * scale))
            scaled_x = F.interpolate(x, size=scaled_size, mode='bilinear', align_corners=False)

            # Forward pass
            pred = self.model(scaled_x)

            # Resize predictions back
            if isinstance(pred, list):
                scaled_pred = []
                for p in pred:
                    scaled_pred.append(F.interpolate(p, size=original_size, mode='bilinear', align_corners=False))
                predictions.append(scaled_pred)
            else:
                predictions.append(F.interpolate(pred, size=original_size, mode='bilinear', align_corners=False))

        # Ensemble predictions (simple averaging)
        if isinstance(predictions[0], list):
            ensemble_pred = []
            for i in range(len(predictions[0])):
                ensemble_pred.append(torch.mean(torch.stack([p[i] for p in predictions]), dim=0))
        else:
            ensemble_pred = torch.mean(torch.stack(predictions), dim=0)

        return ensemble_pred

class SoftNMS(nn.Module):
    """Soft Non-Maximum Suppression for dense scenes"""
    def __init__(self, sigma=0.5, thresh=0.001, method='linear'):
        super(SoftNMS, self).__init__()
        self.sigma = sigma
        self.thresh = thresh
        self.method = method

    def forward(self, boxes, scores):
        """
        boxes: (N, 4) tensor of boxes
        scores: (N,) tensor of scores
        """
        # Sort by scores
        _, order = scores.sort(0, descending=True)
        boxes = boxes[order]
        scores = scores[order]

        # Apply soft NMS
        for i in range(len(scores)):
            if scores[i] == 0:
                continue

            # Calculate IoU with subsequent boxes
            iou = self.box_iou(boxes[i:i+1], boxes[i+1:])[0]

            if self.method == 'linear':
                decay = 1 - iou
            elif self.method == 'gaussian':
                decay = torch.exp(-iou**2 / self.sigma)
            else:
                raise ValueError(f"Unknown method: {self.method}")

            scores[i+1:] *= decay

            # Remove low-confidence detections
            scores[scores < self.thresh] = 0

        return boxes, scores

    @staticmethod
    def box_iou(box1, box2):
        """Calculate IoU between boxes"""
        def box_area(box):
            return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])

        area1 = box_area(box1)
        area2 = box_area(box2)

        lt = torch.max(box1[:, None, :2], box2[:, :2])
        rb = torch.min(box1[:, None, 2:], box2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]

        union = area1[:, None] + area2 - inter
        iou = inter / union
        return iou


class RepulsionLoss(nn.Module):
    """Repulsion loss for dense object detection"""
    def __init__(self, sigma=0.5):
        super(RepulsionLoss, self).__init__()
        self.sigma = sigma

    def forward(self, pred_boxes, gt_boxes):
        """
        pred_boxes: (N, 4) predicted boxes
        gt_boxes: (M, 4) ground truth boxes
        """
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            return torch.tensor(0.0, device=pred_boxes.device)

        # Calculate pairwise IoU
        iou = SoftNMS.box_iou(pred_boxes, gt_boxes)

        # Repulsion loss: penalize high IoU between different objects
        repulsion = torch.exp(-iou / self.sigma)
        loss = torch.sum(repulsion) / (len(pred_boxes) * len(gt_boxes))

        return loss


class InstanceSegmentationHead(nn.Module):
    """Instance segmentation head for dense scenes"""
    def __init__(self, c1, num_classes):
        super(InstanceSegmentationHead, self).__init__()
        self.mask_conv = nn.Sequential(
            Conv(c1, c1, 3, 1),
            Conv(c1, c1, 3, 1),
            Conv(c1, num_classes, 1, 1)
        )
        self.boundary_conv = nn.Sequential(
            Conv(c1, c1, 3, 1),
            Conv(c1, 1, 1, 1)
        )

    def forward(self, x):
        masks = self.mask_conv(x)
        boundaries = self.boundary_conv(x)
        return masks, boundaries


class AnchorClustering(nn.Module):
    """Anchor clustering for VisDrone dataset optimization"""
    def __init__(self, num_anchors=9, num_clusters=5):
        super(AnchorClustering, self).__init__()
        self.num_anchors = num_anchors
        self.num_clusters = num_clusters

    @staticmethod
    def kmeans_anchors(dataset_boxes, n=9, img_size=640):
        """K-means clustering to find optimal anchors"""
        from scipy.cluster.vq import kmeans

        # Convert boxes to width, height format
        boxes = np.array([[box[2] - box[0], box[3] - box[1]] for box in dataset_boxes])

        # Normalize by image size
        boxes = boxes / img_size

        # K-means clustering
        centroids, _ = kmeans(boxes, n, iter=30)

        # Sort by area
        areas = centroids[:, 0] * centroids[:, 1]
        sorted_indices = np.argsort(areas)
        centroids = centroids[sorted_indices]

        return centroids

    def forward(self, x):
        # This is just a placeholder - actual anchor generation happens during training
        return x


class AnchorFreeHead(nn.Module):
    """Anchor-free detection head for VisDrone small objects"""
    def __init__(self, c1, num_classes):
        super(AnchorFreeHead, self).__init__()
        # Heatmap for object centers
        self.center_heatmap = nn.Sequential(
            Conv(c1, c1, 3, 1),
            Conv(c1, num_classes, 1, 1)
        )

        # Size prediction (width, height)
        self.size_pred = nn.Sequential(
            Conv(c1, c1, 3, 1),
            Conv(c1, 2, 1, 1)
        )

        # Offset prediction for center refinement
        self.offset_pred = nn.Sequential(
            Conv(c1, c1, 3, 1),
            Conv(c1, 2, 1, 1)
        )

    def forward(self, x):
        heatmap = self.center_heatmap(x)
        size = self.size_pred(x)
        offset = self.offset_pred(x)
        return heatmap, size, offset


class VisDroneAnchorGenerator(nn.Module):
    """VisDrone-specific anchor generator with optimized sizes"""
    def __init__(self, strides=[8, 16, 32], anchor_sizes=None):
        super(VisDroneAnchorGenerator, self).__init__()
        self.strides = strides

        # VisDrone-optimized anchor sizes (width, height) normalized by stride
        if anchor_sizes is None:
            # Based on VisDrone dataset statistics - smaller anchors for tiny objects
            self.anchor_sizes = {
                8: [[0.05, 0.05], [0.1, 0.1], [0.15, 0.15]],  # Small objects
                16: [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]],     # Medium objects
                32: [[0.2, 0.2], [0.4, 0.4], [0.6, 0.6]]      # Large objects
            }
        else:
            self.anchor_sizes = anchor_sizes

    def forward(self, feature_maps):
        """Generate anchors for each feature map"""
        anchors = []
        for i, feat_map in enumerate(feature_maps):
            stride = self.strides[i]
            h, w = feat_map.shape[2:]

            # Generate grid
            y_coords = torch.arange(h, device=feat_map.device) * stride
            x_coords = torch.arange(w, device=feat_map.device) * stride
            y_grid, x_grid = torch.meshgrid(y_coords, x_coords)

            # Generate anchors for each position and size
            for anchor_size in self.anchor_sizes[stride]:
                w_a, h_a = anchor_size
                x1 = x_grid - w_a * stride / 2
                y1 = y_grid - h_a * stride / 2
                x2 = x_grid + w_a * stride / 2
                y2 = y_grid + h_a * stride / 2

                anchor = torch.stack([x1.flatten(), y1.flatten(),
                                    x2.flatten(), y2.flatten()], dim=1)
                anchors.append(anchor)

        return torch.cat(anchors, dim=0)


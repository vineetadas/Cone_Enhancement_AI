import os
import torch
import torch.nn as n
import torch.nn.functional as f
import numpy as np
import torch.optim as optim
from torchvision import models, datasets
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook,tqdm
import cv2
import torchvision.utils as vutils
from matplotlib import pyplot
# from dataset import get_training_data_RF 
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from torch import einsum
import torch.nn.functional as F
import scipy.io

# -------------define transformer layers
class PreNorm(n.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = n.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class GELU(n.Module):
    def forward(self, x):
        return F.gelu(x)


class WMSA(n.Module):
    def __init__(
        self,
        dim,
        window_size=(8,8),
        heads = 8,
        shift_size = (0,0)
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        dim_head = dim // heads
        self.scale = dim_head ** -0.5
        self.window_size = window_size
        self.shift_size = shift_size
        

        # position embedding
        seq_l = window_size[0]*window_size[1]
        self.pos_emb = n.Parameter(torch.Tensor(1, heads, seq_l, seq_l))
        trunc_normal_(self.pos_emb)

        inner_dim = dim_head * heads
        
        self.to_q = n.Linear(dim, inner_dim, bias=False)
        self.to_kv = n.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = n.Linear(inner_dim, dim)

    def forward(self,x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b,h,w,c = x.shape
        w_size = self.window_size

        assert h % w_size[0] == 0 and w % w_size[1] == 0 , 'fmap dimensions must be divisible by the window size'
        
        

        if self.shift_size[0] > 0:
            x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2)) 

        x_inp = x.view(x.shape[0]*x.shape[1]//w_size[0]*x.shape[2]//w_size[1], w_size[0]*w_size[1], x.shape[3])

        q = self.to_q(x_inp)
        k, v = self.to_kv(x_inp).chunk(2, dim=-1) 

        q, k, v = map(lambda t: t.contiguous().view(t.shape[0],self.heads,t.shape[1],t.shape[2]//self.heads), (q, k, v))
        q *= self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        sim = sim + self.pos_emb
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = out.view(out.shape[0], out.shape[2], -1)
        out = self.to_out(out)
        out = out.view(out.shape[0] // (h // w_size[0]) // (w // w_size[1]), h, w, c)
        if self.shift_size[0] > 0:
            out = torch.roll(out, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))

        return out

class FeedForward(n.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = n.Sequential(
            n.Conv2d(dim, dim * mult,1,1,bias=False),
            n.LeakyReLU(negative_slope=0.2),
            n.Conv2d(dim*mult, dim*mult, 3, 1, 1, bias=False, groups=dim*mult),
            n.LeakyReLU(negative_slope=0.2),
            n.Conv2d(dim * mult, dim,1,1,bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0,3,1,2))
        return out.permute(0,2,3,1)


class TransformerLayer(n.Module):
    def __init__(
            self,
            dim,
            window_size=(5, 5),
            heads=4,
            num_blocks = 2,
    ):
        super().__init__()
        self.blocks = n.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(n.ModuleList([
                PreNorm(dim, WMSA(dim=dim,window_size=window_size,heads=heads,
                                          shift_size=(0,0) if (_%2==0) else (window_size[0]//2,window_size[1]//2))),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0,2,3,1)
        for (attn, ff) in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0,3,1,2)
        return out   
    
class ResidualDenseBlockTransformer(n.Module):
    def __init__(self, dim=64, beta = 0.2):
        super().__init__()
        self.conv1 = TransformerLayer(dim = dim, window_size = (5,5), heads = 8, num_blocks = 2)   
        self.conv2 = TransformerLayer(dim = 2*dim,  window_size = (5,5), heads = 8, num_blocks = 2)   
        self.conv3 = TransformerLayer(dim = 4*dim,  window_size = (5,5), heads = 8, num_blocks = 2) 
        # self.conv4 = TransformerLayer(dim = 8*dim,  window_size = (5,5), heads = 4, num_blocks = 2) 
        self.conv5 = n.Conv2d(8 * dim,  dim, 3, 1, 1)
        self.lrelu = n.LeakyReLU(negative_slope=0.2)
        self.b = beta
        
    def forward(self, x):
        block1 = self.lrelu(self.conv1(x))
        block2 = self.lrelu(self.conv2(torch.cat((block1, x), dim = 1)))
        block3 = self.lrelu(self.conv3(torch.cat((block2, block1, x), dim = 1)))
        # block4 = self.lrelu(self.conv4(torch.cat((block3, block2, block1, x), dim = 1)))
        out = self.conv5(torch.cat((block3, block2, block1, x), dim = 1))
        
        return x + self.b * out
    
class ResidualInResidualDenseBlockTransformer(n.Module):
    def __init__(self, beta = 0.2):
        super().__init__()
        self.RDB = ResidualDenseBlockTransformer()
        self.b = beta
    
    def forward(self, x):
        out = self.RDB(x)
        out = self.RDB(out)
        out = self.RDB(out)
        
        return x + self.b * out    
    
class Generator(n.Module):
    def __init__(self,in_channel = 1, out_channel = 1, noRRDBBlockTrans = 5, scale_fac = 4):
        super().__init__()   
        self.conv1 = n.Conv2d(1, 64, 3, 1, 1)
        
        # self.conv2 = n.Conv2d(64, 128, 1, 1, 1)
        
        # self.upsample = n.UpsamplingNearest2d(scale_factor = (1, scale_fac))
        
        # self.test = TransformerLayer(dim = 64, window_size = (5,5), heads = 4, num_blocks = 2)   

        # RRDB = ResidualInResidualDenseBlock()
        # RRDB_layer = []
        # for i in range(noRRDBBlockConv):
        #     RRDB_layer.append(RRDB)
        # self.RRDB_block =  n.Sequential(*RRDB_layer)

        RRDBT = ResidualInResidualDenseBlockTransformer()
        RRDBT_layer = []
        for i in range(noRRDBBlockTrans):
            RRDBT_layer.append(RRDBT)
        self.RRDBT_block =  n.Sequential(*RRDBT_layer)

        self.RRDB_conv2 = n.Conv2d(64, 64, 3, 1, 1)
        self.upconv = n.Conv2d(64, 64, 3, 1, 1)

        self.out_conv = n.Conv2d(64, 1, 3, 1, 1)
        self.tanh = n.Tanh()
    
    def forward(self, x):
        first_conv = self.conv1(x)
        
        # second_conv = self.conv2(first_conv)
        
        # rrdb_conv_block = self.RRDB_block(first_conv)
                
        rrdb_trans_block = self.RRDBT_block(first_conv)
        
        # RRDB_full_block = torch.cat ((rrdb_conv_block,rrdb_trans_block),dim=1)
        # rrdb_trans_block = self.RRDBT_block(rrdb_conv_block)
        # RRDB_full_block = torch.add(rrdb_trans_block,first_conv)
        
        RRDB_full_block = torch.add(self.RRDB_conv2(rrdb_trans_block),first_conv)
        upconv_block1 = self.upconv(f.interpolate(RRDB_full_block,scale_factor = (1,4),mode= 'bicubic'))
        # upconv_block2 = self.upconv(f.interpolate(upconv_block1, scale_factor = (1,2),  mode= 'bicubic'))
        # # upconv_block2 = self.upsample(RRDB_full_block)
        out = self.tanh(self.out_conv(upconv_block1))
        
        return out
            

class Discriminator(n.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.features = n.Sequential(
            # input size. (3) x 128 x 128
            n.Conv2d(1, 32, (3, 3), (1, 1), (1, 1), bias=True),
            n.LeakyReLU(0.2, True),
            # state size. (64) x 64 x 64
            n.Conv2d(32, 64, (4, 4), (2, 2), (1, 1), bias=False),
            n.BatchNorm2d(64),
            n.LeakyReLU(0.2, True),
            n.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), bias=False),
            n.BatchNorm2d(128),
            n.LeakyReLU(0.2, True),
            # state size. (128) x 32 x 32
            n.Conv2d(128, 256, (4, 4), (2, 2), (1, 1), bias=False),
            n.BatchNorm2d(256),
            n.LeakyReLU(),
            n.Conv2d(256, 512, (4, 4), (2, 2), (1, 1), bias=False),
            n.BatchNorm2d(512),
            n.LeakyReLU(),
            n.AdaptiveAvgPool2d(1)
        )
        
        self.fc1 = n.Linear(512,1)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.shape[0], -1)
        out = self.fc1(out)
        # out = torch.flatten(out, 1)
        # out = self.classifier(out)

        return out
#!/usr/bin/env python
# -*- coding:utf-8 -*-

import math
import numpy as np
import torch
from torch import nn


class ConvBNReLU(nn.Module):
    def __init__(self,
                 c_in,
                 c_out,
                 kernel_size,
                 stride=1,
                 padding=1,
                 activation=True):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(c_in,
                              c_out,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU()
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = self.relu(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(cin, cout, 3, 1, 1),
            ConvBNReLU(cout, cout, 3, 1, 1, activation=False))
        self.conv1 = nn.Conv2d(cout, cout, 1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = self.conv(x)
        h = x
        x = self.conv1(x)
        x = self.bn(x)
        x = h + x # skip connection
        x = self.relu(x)
        return x


class U_encoder(nn.Module):
    def __init__(self):
        super(U_encoder, self).__init__()
        self.res1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.res2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.res3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        #self.pool3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=14, bias=False)

    def forward(self, x):
        features = []
        x = self.res1(x)
        features.append(x)  # (224, 224, 64)
        x = self.pool1(x)

        x = self.res2(x)
        features.append(x)  # (112, 112, 128)
        x = self.pool2(x)

        x = self.res3(x)
        features.append(x)  # (56, 56, 256)
        x = self.pool3(x)
        
        return x, features


class U_decoder(nn.Module):
    def __init__(self):
        super(U_decoder, self).__init__()
        # dilation=1(default): controls the spacing between the kernel points; also known as the à trous algorithm -> 3x3 kernel
        # stride=2: 1 zeros (spaces) between each input pixel
        self.trans1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.res1 = DoubleConv(512, 256)
        self.trans2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.res2 = DoubleConv(256, 128)
        self.trans3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.res3 = DoubleConv(128, 64)

    def forward(self, x, feature):

        x = self.trans1(x)  # (56, 56, 256)
        x = torch.cat((feature[2], x), dim=1) # skip connection
        x = self.res1(x)  # (56, 56, 256)
        x = self.trans2(x)  # (112, 112, 128)
        x = torch.cat((feature[1], x), dim=1)
        x = self.res2(x)  # (112, 112, 128)
        x = self.trans3(x)  # (224, 224, 64)
        x = torch.cat((feature[0], x), dim=1)
        x = self.res3(x)
        return x

# external attention
class MEAttention(nn.Module):
    def __init__(self, dim, configs):
        super(MEAttention, self).__init__()
        self.num_heads = configs["head"]
        self.coef = 4 # enlarge Q for improving the representation learning ability
        self.query_liner = nn.Linear(dim, dim * self.coef)
        self.num_heads = self.coef * self.num_heads
        self.k = 256 // self.coef
        self.key_linear = nn.Linear(dim * self.coef // self.num_heads, self.k) # shared memory units M_K
        self.value_linear = nn.Linear(self.k, dim * self.coef // self.num_heads) # shared memory units M_V

        self.proj = nn.Linear(dim * self.coef, dim)

    def forward(self, x):
        B, N, C = x.shape # N: flatten featuresmaps
        x = self.query_liner(x) # (B, N, C*self.coef)
        x = x.view(B, N, self.num_heads, -1).permute(0, 2, 1, 3) #(B, heads, N, features)
        # Q and M_K are distributed represention
        # similarity of Q and M_K
        attn = self.key_linear(x) #(B, heads, N, k) 
        # for each concept (distributed feature), similarities are compared between all flatten featuremaps
        # softmax along N axis -> attention weights between each featuremap (sum of attention is 1 along N axis)
        attn = nn.Softmax(dim=-2)(attn) #(B, heads, N, k) 
        # V is distributed representation
        # for each flatten featuremap, all concepts (distributed features) are weighted sum into final represeantion
        # normalized in each featuremap, just like layer normalization
        attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True)) # (B, heads, N, k) 

        x = self.value_linear(attn).permute(0, 2, 1, 3).reshape(B, N, -1) #(B, heads, N, features) -> (B, N, heads, features) -> (B, N, C*self.coef)

        x = self.proj(x) # (B, N, C)

        return x


class Attention(nn.Module):
    def __init__(self, dim, configs, axial=False):
        super(Attention, self).__init__()
        self.axial = axial
        self.dim = dim
        self.num_head = configs["head"]
        self.attention_head_size = int(self.dim / configs["head"])
        self.all_head_size = self.num_head * self.attention_head_size

        self.query_layer = nn.Linear(self.dim, self.all_head_size)
        self.key_layer = nn.Linear(self.dim, self.all_head_size)
        self.value_layer = nn.Linear(self.dim, self.all_head_size)

        self.out = nn.Linear(self.all_head_size, self.dim)
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_head, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x

    def forward(self, x):
        if self.axial:
            # global self-attention
            b, h, w, c = x.shape # h: height, w: width, c: channel
            # transform into distributed representation
            mixed_query_layer = self.query_layer(x)
            mixed_key_layer = self.key_layer(x)
            mixed_value_layer = self.value_layer(x)
            
            # row attention (single head attention)
            # attention of width in each height
            query_layer_x = mixed_query_layer.view(b * h, w, -1)
            key_layer_x = mixed_key_layer.view(b * h, w, -1).transpose(-1, -2)
            attention_scores_x = torch.matmul(query_layer_x,
                                              key_layer_x)  # (b*h, w, w)
            attention_scores_x = attention_scores_x.view(b, -1, w,
                                                         w)  # (b, h, w, w)

            # col attention  (single head attention)
            # attention height in each width
            query_layer_y = mixed_query_layer.permute(0, 2, 1,
                                                      3).contiguous().view(
                                                          b * w, h, -1)
            key_layer_y = mixed_key_layer.permute(
                0, 2, 1, 3).contiguous().view(b * w, h, -1).transpose(-1, -2)
            attention_scores_y = torch.matmul(query_layer_y,
                                              key_layer_y)  # (b*w, h, h)
            attention_scores_y = attention_scores_y.view(b, -1, h,
                                                         h)  # (b, w, h, h)

            return attention_scores_x, attention_scores_y, mixed_value_layer

        else:
            # local self-attention
            # x: (b, height, width, window*window, channel)
            # transform into distributed representation
            mixed_query_layer = self.query_layer(x)
            mixed_key_layer = self.key_layer(x)
            mixed_value_layer = self.value_layer(x)

            query_layer = self.transpose_for_scores(mixed_query_layer).permute(
                0, 1, 2, 4, 3, 5).contiguous()  # (b, h', w', head, window*window, c) 
            key_layer = self.transpose_for_scores(mixed_key_layer).permute(
                0, 1, 2, 4, 3, 5).contiguous() # (b, h', w', head, window*window, c')
            value_layer = self.transpose_for_scores(mixed_value_layer).permute(
                0, 1, 2, 4, 3, 5).contiguous() # (b, h', w', head, window*window, c')

            # attention scores of each window to all windows
            attention_scores = torch.matmul(query_layer,
                                            key_layer.transpose(-1, -2)) # (b, h', w', head, window*window, window*window)
            attention_scores = attention_scores / math.sqrt(
                self.attention_head_size)
            atten_probs = self.softmax(attention_scores) # (b, h', w', head, window*window, window*window)

            # attended value in x window = weighted sum of attention_xj and value_i
            # where attention_xj is attention score of x window attends to j window
            # and where value_i is the i-th dimension in c'-axis (feature dims)
            # => information of each window is comprised of itself attends to all window
            context_layer = torch.matmul(
                atten_probs, value_layer)  # (b, h', w', head, window*window, c)
            context_layer = context_layer.permute(0, 1, 2, 4, 3,
                                                  5).contiguous()  # (b, h', w', window*window, head, c)
            new_context_layer_shape = context_layer.size()[:-2] + (
                self.all_head_size, ) 
            context_layer = context_layer.view(*new_context_layer_shape) # (b, h', w', window*window, c)
            attention_output = self.out(context_layer) # (b, h', w', window*window, self.dim=channel)

        return attention_output

# local self-attention
class WinAttention(nn.Module):
    def __init__(self, configs, dim):
        super(WinAttention, self).__init__()
        self.window_size = configs["win_size"]
        self.attention = Attention(dim, configs)

    def forward(self, x):
        b, n, c = x.shape # n: flatten featuremap (height*width)
        h, w = int(np.sqrt(n)), int(np.sqrt(n))
        x = x.permute(0, 2, 1).contiguous().view(b, c, h, w)
        if h % self.window_size != 0:
            # padding (repeated duplicating)
            right_size = h + (self.window_size - h % self.window_size)
            new_x = torch.zeros((b, c, right_size, right_size))
            new_x[:, :, 0:x.shape[2], 0:x.shape[3]] = x[:]
            new_x[:, :, x.shape[2]:,
                  x.shape[3]:] = x[:, :, (x.shape[2] - right_size):,
                                   (x.shape[3] - right_size):]
            x = new_x
            b, c, h, w = x.shape
        x = x.view(b, c, h // self.window_size, self.window_size,
                   w // self.window_size, self.window_size) # (b,c,h',win,w',win)
        x = x.permute(0, 2, 4, 3, 5,
                      1).contiguous().view(b, h // self.window_size,
                                           w // self.window_size,
                                           self.window_size * self.window_size,
                                           c).cuda() # (b,h',w',win*win,c)
        # each window is comprise of all windows
        # while Q*K gives attention scores between the window itself and all windows, V derives from the window itself
        # thus, attention scores mean the correlation between the window itself and all windows
        # attended V still depends on the value information of window itself
        x = self.attention(x)  # (b, h', w', win*win, c)
        return x

# convolve whole input by weighted-sum all window features 
# aggregate after local attention
class DlightConv(nn.Module):
    def __init__(self, dim, configs, aggregate=False):
        super(DlightConv, self).__init__()
        self.aggregate = aggregate
        self.linear = nn.Linear(dim, configs["win_size"] * configs["win_size"])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        h = x # (b, h', w', win*win, c)
        avg_x = torch.mean(x, dim=-2)  # (b, h', w', c) 
        # all windows give information to each window 
        # attended V may be large because the value information of the window itself
        # but to whole image (all windows), it may not important
        # thus, here learns a overall attention of whole image to each window
        x_prob = self.softmax(self.linear(avg_x))  # (b, h', w', win*win)

        # inportance of each window in whole image (all windows) 
        x = torch.mul(h,
                      x_prob.unsqueeze(-1))  # (b, h', w', win*win, c)*(b, h', w', win*win, 1) -> (b, h', w', win*win, c)
        if self.aggregate:
            x = torch.sum(x, dim=-2)  # (b, h', w', c)
        return x

# LSA (local self-attention) module
class LSAttention(nn.Module):
    def __init__(self, dim, configs):
        super(LSAttention, self).__init__()
        # local attention
        self.win_atten = WinAttention(configs, dim)
        # aggragate
        self.dlightconv = DlightConv(dim, configs)
        
    def forward(self, x):
        '''
        :param x: size(b, n, c)
        :return:
        '''
        origin_size = x.shape
        _, origin_h, origin_w, _ = origin_size[0], int(np.sqrt(
            origin_size[1])), int(np.sqrt(origin_size[1])), origin_size[2]
        # local attention
        x = self.win_atten(x)  # (b, h', w', win*win, c)
        # aggregate
        x = self.dlightconv(x)  # (b, h', w', win*win, c)
        b, p, p, win, c = x.shape # (b, h', w', win*win, c)
        x = x.view(b, p, p, int(np.sqrt(win)), int(np.sqrt(win)),
                   c).permute(0, 1, 3, 2, 4, 5).contiguous() # (b, h', w', win, win, c) -> (b, h', win, w', win, c)
        x = x.view(b, p * int(np.sqrt(win)), p * int(np.sqrt(win)),
                   c).contiguous()  # (b, h'*win, w'*win, c) = (b, h, w, c)
        x = x[:, :origin_h, :origin_w, :].contiguous()
        x = x.view(b, -1, c)

        return x
    
#  Gaussian-Weighted Axial Attention
class GaussianTrans(nn.Module):
    def __init__(self):
        super(GaussianTrans, self).__init__()
        self.bias = nn.Parameter(-torch.abs(torch.randn(1)))
        self.shift = nn.Parameter(torch.abs(torch.randn(1)))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x, atten_x_full, atten_y_full, value_full = x  # x(b, h, w, c) atten_x_full(b, h, w, w) atten_y_full(b, w, h, h) value_full(b, h, w, c) 
        new_value_full = torch.zeros_like(value_full)

        for r in range(x.shape[1]):  # row (height)
            for c in range(x.shape[2]):  # col (width)
                atten_x = atten_x_full[:, r, c, :]  # (b, w)
                atten_y = atten_y_full[:, c, r, :]  # (b, h)

                dis_x = torch.tensor([(h - c)**2 for h in range(x.shape[2])
                                      ]).cuda()  # (b, w)
                dis_y = torch.tensor([(w - r)**2 for w in range(x.shape[1])
                                      ]).cuda()  # (b, h)

                dis_x = -(self.shift * dis_x + self.bias).cuda() # (b, w)
                dis_y = -(self.shift * dis_y + self.bias).cuda() # (b, h)
                # adding position information to attention
                atten_x = self.softmax(dis_x + atten_x) # (b, w)
                atten_y = self.softmax(dis_y + atten_y) # (b, h)

                new_value_full[:, r, c, :] = torch.sum(
                    atten_x.unsqueeze(dim=-1) * value_full[:, r, :, :] +
                    atten_y.unsqueeze(dim=-1) * value_full[:, :, c, :],
                    dim=-2) # (b, c)
        return new_value_full # (b, h, w, c) 

# LGG-SA
class CSAttention(nn.Module):
    def __init__(self, dim, configs):
        super(CSAttention, self).__init__()
        # local attention
        self.win_atten = WinAttention(configs, dim)
        # aggragate
        self.dlightconv = DlightConv(dim, configs, aggregate=True)
        # global attention
        self.global_atten = Attention(dim, configs, axial=True)
        self.gaussiantrans = GaussianTrans()
        #self.conv = nn.Conv2d(dim, dim, 3, padding=1)
        #self.maxpool = nn.MaxPool2d(2)
        self.up = nn.UpsamplingBilinear2d(scale_factor=4)
        self.queeze = nn.Conv2d(2 * dim, dim, 1)

    def forward(self, x):
        '''
        :param x: size(b, n, c)
        :return:
        '''
        origin_size = x.shape
        _, origin_h, origin_w, _ = origin_size[0], int(np.sqrt(
            origin_size[1])), int(np.sqrt(origin_size[1])), origin_size[2]
        # local attention
        x = self.win_atten(x)  # (b, h', w', win*win, c)
        b, p, p, win, c = x.shape # (b, h', w', win*win, c)
        h = x.view(b, p, p, int(np.sqrt(win)), int(np.sqrt(win)),
                   c).permute(0, 1, 3, 2, 4, 5).contiguous() # (b, h', w', win, win, c) -> (b, h', win, w', win, c)
        h = h.view(b, p * int(np.sqrt(win)), p * int(np.sqrt(win)),
                   c).permute(0, 3, 1, 2).contiguous()  # (b, h'*win, w'*win, c) -> (b, c, h, w)
        # aggregate
        x = self.dlightconv(x)  # (b, h', w', c)
        # global attention
        atten_x, atten_y, mixed_value = self.global_atten(
            x)  # (atten_x, atten_y, value)
        gaussian_input = (x, atten_x, atten_y, mixed_value)
        x = self.gaussiantrans(gaussian_input)  # (b, h', w', c)
        x = x.permute(0, 3, 1, 2).contiguous()
        # upsample
        x = self.up(x)
        # concat LSA and GLA then squeeze
        x = self.queeze(torch.cat((x, h), dim=1)).permute(0, 2, 3,
                                                          1).contiguous()
        x = x[:, :origin_h, :origin_w, :].contiguous()
        x = x.view(b, -1, c)

        return x

# mixed transformer module
class EAmodule(nn.Module):
    def __init__(self, dim, global_SA=False):
        super(EAmodule, self).__init__()
        self.SlayerNorm = nn.LayerNorm(dim, eps=1e-6)
        self.ElayerNorm = nn.LayerNorm(dim, eps=1e-6)
        if global_SA:
            self.SAttention = CSAttention(dim, configs)
        else:
             self.SAttention = LSAttention(dim, configs)
        self.EAttention = MEAttention(dim, configs)

    def forward(self, x):
        h = x  # (B, flattened_featuremap, channels)
        x = self.SlayerNorm(x)

        x = self.SAttention(x)  # padding 到right_size
        x = h + x

        h = x
        x = self.ElayerNorm(x)

        x = self.EAttention(x)
        x = h + x

        return x


class DecoderStem(nn.Module):
    def __init__(self):
        super(DecoderStem, self).__init__()
        self.block = U_decoder()

    def forward(self, x, features):
        x = self.block(x, features)
        return x


class Stem(nn.Module):
    def __init__(self):
        super(Stem, self).__init__()
        self.model = U_encoder()
        self.trans_dim = ConvBNReLU(256, 256, 1, 1, 0)  #out_dim, model_dim
        self.position_embedding = nn.Parameter(torch.zeros((1, 784, 256)))
        # self.position_embedding = nn.Parameter(torch.zeros((1, 4096, 256)))

    def forward(self, x):

        x, features = self.model(x)  # (B, 256, 28, 28)
        x = self.trans_dim(x)  # (B, C, H, W) (B, 256, 28, 28)
        x = x.flatten(2)  # (B, C, H*W)  (B, 256, 28*28)
        x = x.transpose(-2, -1)  #  (B, H*W, C)
        x = x + self.position_embedding
        return x, features  #(B, H*W, C), [(B, C, H, W)]

# made up of mixed transformer modules and conv/deconv
class encoder_block(nn.Module):
    def __init__(self, dim, global_SA=False, dilation=1):
        super(encoder_block, self).__init__()
        self.block = nn.ModuleList([
            EAmodule(dim, global_SA=global_SA),
            EAmodule(dim, global_SA=global_SA),
            ConvBNReLU(dim, dim * 2, 2, stride=2, padding=0)
            #nn.Conv2d(dim, dim*2, kernel_size=4, stride=1, dilation=dilation, padding=1, bias=False) if dilation==3 else nn.Conv2d(dim, dim*2, kernel_size=3, stride=1, dilation=dilation, bias=False)
        ])

    def forward(self, x):
        x = self.block[0](x)
        x = self.block[1](x)
        B, N, C = x.shape
        h, w = int(np.sqrt(N)), int(np.sqrt(N))
        x = x.view(B, h, w, C).permute(0, 3, 1,
                                       2)  # (1, 256, 28, 28) B, C, H, W
        skip = x
        x = self.block[2](x)  # (14, 14, 256)
        return x, skip


class decoder_block(nn.Module):
    def __init__(self, dim, flag, global_SA=False):
        super(decoder_block, self).__init__()
        self.flag = flag
        if not self.flag:
            self.block = nn.ModuleList([
                nn.ConvTranspose2d(dim,
                                   dim // 2,
                                   kernel_size=2,
                                   stride=2, # upsampling 2-times
                                   padding=0),
                nn.Conv2d(dim, dim // 2, kernel_size=1, stride=1),
                EAmodule(dim // 2, global_SA=global_SA),
                EAmodule(dim // 2, global_SA=global_SA)
            ])
        else:
            self.block = nn.ModuleList([
                nn.ConvTranspose2d(dim,
                                   dim // 2,
                                   kernel_size=2,
                                   stride=2,
                                   padding=0),
                EAmodule(dim, global_SA=global_SA),
                EAmodule(dim, global_SA=global_SA)
            ])

    def forward(self, x, skip):
        if not self.flag:
            x = self.block[0](x)
            x = torch.cat((x, skip), dim=1)
            x = self.block[1](x)
            x = x.permute(0, 2, 3, 1)
            B, H, W, C = x.shape
            x = x.view(B, -1, C)
            x = self.block[2](x)
            x = self.block[3](x)
        else:
            x = self.block[0](x)
            x = torch.cat((x, skip), dim=1)
            x = x.permute(0, 2, 3, 1)
            B, H, W, C = x.shape
            x = x.view(B, -1, C)
            x = self.block[1](x)
            x = self.block[2](x)
        return x


class MTUNet(nn.Module):
    def __init__(self, out_ch=2, global_SA=True):
        super(MTUNet, self).__init__()
        """
        complete architecture
        args: 
            out_ch: number of output classes, default=2 (0/1)
            global_SA: bool, use global self-attention if True (learns global contextual information)
        """
        self.stem = Stem()
        self.encoder = nn.ModuleList()
        self.bottleneck = nn.Sequential(EAmodule(configs["bottleneck"], global_SA=global_SA),
                                        EAmodule(configs["bottleneck"], global_SA=global_SA))
        self.decoder = nn.ModuleList()

        self.decoder_stem = DecoderStem()
        for i in range(len(configs["encoder"])):
            dim = configs["encoder"][i]
            dilation=[7,3]
            self.encoder.append(encoder_block(dim, global_SA=global_SA ,dilation=dilation[i]))
        for i in range(len(configs["decoder"]) - 1):
            dim = configs["decoder"][i]
            self.decoder.append(decoder_block(dim, False))
        self.decoder.append(decoder_block(configs["decoder"][-1], True))
        self.SegmentationHead = nn.Conv2d(64, out_ch, 1) # out_ch = num_class , kernel_size=1

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, features = self.stem(x)  #(B, N, C) (1, 196, 256)
        skips = []
        for i in range(len(self.encoder)):
            x, skip = self.encoder[i](x)
            skips.append(skip)
            B, C, H, W = x.shape  #  (1, 512, 8, 8)
            x = x.permute(0, 2, 3, 1).contiguous().view(B, -1, C)  # (B, N, C)
        x = self.bottleneck(x)  # (1, 25, 1024)
        B, N, C = x.shape
        x = x.view(B, int(np.sqrt(N)), -1, C).permute(0, 3, 1, 2)
        for i in range(len(self.decoder)):
            x = self.decoder[i](x,
                                skips[len(self.decoder) - i - 1])  # (B, N, C)
            B, N, C = x.shape
            x = x.view(B, int(np.sqrt(N)), int(np.sqrt(N)),
                       C).permute(0, 3, 1, 2)

        x = self.decoder_stem(x, features)
        x = self.SegmentationHead(x)
        return x

class MTUNet_Pretrain(nn.Module):
    def __init__(self, out_ch=2, global_SA=True):
        super(MTUNet_Pretrain, self).__init__()
        """
        complete architecture
        args: 
            out_ch: number of output classes, default=2 (0/1)
            global_SA: bool, use global self-attention if True (learns global contextual information)
        """
        self.stem = Stem()
        self.encoder = nn.ModuleList()
        self.bottleneck = nn.Sequential(EAmodule(configs["bottleneck"], global_SA=global_SA),
                                        EAmodule(configs["bottleneck"], global_SA=global_SA))
        for i in range(len(configs["encoder"])):
            dim = configs["encoder"][i]
            self.encoder.append(encoder_block(dim))
        
    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, features = self.stem(x)  #(B, N, C) (1, 196, 256)
        skips = []
        for i in range(len(self.encoder)):
            x, skip = self.encoder[i](x)
            skips.append(skip)
            B, C, H, W = x.shape  #  (1, 512, 8, 8)
            x = x.permute(0, 2, 3, 1).contiguous().view(B, -1, C)  # (B, N, C)
        x = self.bottleneck(x)  # (1, 25, 1024)
        return x

    
configs = {
    "win_size": 4,
    "head": 8,
    "axis": [28, 16, 8],
    "encoder": [256, 512],
    "bottleneck": 1024,
    "decoder": [1024, 512],
    "decoder_stem": [(256, 512), (256, 256), (128, 64), 32]
}

import torch.nn as nn
import functools
import torch
import math
import torch.nn.functional as F
from math import sqrt

class NetworkBase(nn.Module):
    def __init__(self):
        super(NetworkBase, self).__init__()
        self._name = 'BaseNetwork'

    @property
    def name(self):
        return self._name

    def init_weights(self):
        self.apply(self._weights_init_fn)

    def _weights_init_fn(self, m):
        classname = m.__class__.__name__

        if classname.find('Conv') != -1 and classname.find('ConvLSTM') == -1:
            if m.weight.requires_grad:
                pass
            else:
                m.weight.data.normal_(0.0, 0.02)
                if hasattr(m.bias, 'data'):
                    m.bias.data.fill_(0)
        elif classname.find('BatchNorm2d') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def _get_norm_layer(self, norm_type='batch'):
        if norm_type == 'batch':
            norm_layer = functools.partial(nn.BatchNorm2d,
                                           affine=True)  # looks appealing
        elif norm_type == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        elif norm_type == 'batchnorm2d':
            norm_layer = nn.BatchNorm2d
        else:
            raise NotImplementedError(
                'normalization layer [%s] is not found' % norm_type)

        return norm_layer


class DeConvNetwork(NetworkBase):
    def __init__(self, input_dim=None):
        super(DeConvNetwork, self).__init__()
        self.relu = nn.LeakyReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.deconv1 = nn.ConvTranspose2d(input_dim, 512, kernel_size=4,
                                          stride=2, padding=1, bias=False)
        self.deconv1_bn = torch.nn.BatchNorm2d(512)

        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2,
                                          padding=1, bias=False)
        self.deconv2_bn = torch.nn.BatchNorm2d(256)

        self.deconv3 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2,
                                          padding=1, bias=False)
        self.deconv3_bn = torch.nn.BatchNorm2d(256)

        self.deconv4 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2,
                                          padding=1, bias=False)
        self.deconv4_bn = torch.nn.BatchNorm2d(128)

        self.deconv5 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2,
                                          padding=1, bias=False)
        self.deconv5_bn = torch.nn.BatchNorm2d(64)

        self.conv6 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1,
                               bias=False)

    def forward(self, x):
        x = self.deconv1(x)
        x = self.deconv1_bn(x)
        x = self.relu(x)  # 512, 32

        x = self.deconv2(x)
        x = self.deconv2_bn(x)
        x = self.relu(x)  # 256, 64

        x = self.deconv3(x)
        x = self.deconv3_bn(x)
        x = self.relu(x)  # 128, 128

        x = self.deconv4(x)
        x = self.deconv4_bn(x)
        x = self.relu(x)  # 64, 256

        x = self.deconv5(x)
        x = self.deconv5_bn(x)
        x = self.relu(x)  # 64, 224,224

        x = self.conv6(x)
        out1 = self.sigmoid(x)  # 16,1,224,224

        return out1


class ResidualBlock(nn.Module):
    def __init__(self, ChannelNum=512):
        super(ResidualBlock, self).__init__()
        self.residual_cnn = nn.Sequential(
            nn.Conv2d(ChannelNum, ChannelNum, kernel_size=1, stride=1,
                      padding=0),
            nn.BatchNorm2d(ChannelNum),
            nn.ReLU(inplace=True),
            nn.Conv2d(ChannelNum, ChannelNum, kernel_size=1, stride=1,
                      padding=0),
            nn.BatchNorm2d(ChannelNum))

    def forward(self, x):
        residual = x
        out = self.residual_cnn(x)
        out += residual
        return out


class FullyConnectCls(nn.Module):
    def __init__(self, InputDim=512, OutputDim=10):
        super(FullyConnectCls, self).__init__()
        self.classification = nn.Sequential(
            nn.Linear(InputDim, 4096),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.Dropout(),
            nn.Linear(4096, OutputDim))

    def forward(self, x):
        x = self.classification(x)
        return x


class Self_Attention(nn.Module):
    # input : batch_size * seq_len * input_dim
    # q : batch_size * input_dim * dim_k
    # k : batch_size * input_dim * dim_k
    # v : batch_size * input_dim * dim_v
    def __init__(self,input_dim,dim_k,dim_v):
        super(Self_Attention,self).__init__()
        self.q = nn.Linear(input_dim,dim_k)
        self.k = nn.Linear(input_dim,dim_k)
        self.v = nn.Linear(input_dim,dim_v)
        self._norm_fact = 1 / sqrt(dim_k)
        
    
    def forward(self,x):
        Q = self.q(x) # Q: batch_size * seq_len * dim_k
        K = self.k(x) # K: batch_size * seq_len * dim_k
        V = self.v(x) # V: batch_size * seq_len * dim_v
         
        atten = nn.Softmax(dim=-1)(torch.bmm(Q,K.permute(0,2,1))) * self._norm_fact # Q * K.T() # batch_size * seq_len * seq_len
        
        output = torch.bmm(atten,V) # Q * K.T() * V # batch_size * seq_len * dim_v
        
        return output
 

# Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1,
                          padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1,
                          padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x, y):
        y = self.avg_pool(y)
        y = self.conv_du(y)
        return x * y
    
    
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
    
class ChannelLinearAttention(nn.Module):
    "https://blog.csdn.net/DD_PP_JJ/article/details/103318617"
    def __init__(self, channel, reduction=16): 
        super(ChannelLinearAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # batch不受影响，只针对channel
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class ChannelConvAttention(nn.Module):
    "https://blog.csdn.net/DD_PP_JJ/article/details/103318617"
    def __init__(self, in_planes, ratio=16):
        super(ChannelConvAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    "https://blog.csdn.net/DD_PP_JJ/article/details/103318617"
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        # x = self.conv1(x)
        return x

class SpatialAttention_backup(nn.Module):
    "https://blog.csdn.net/DD_PP_JJ/article/details/103318617"
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
 
    
class Conv_1_1_BnReluLayer(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,dirate=1):
        super(Conv_1_1_BnReluLayer,self).__init__()

        # self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
        # self.bn_s1 = nn.BatchNorm2d(out_ch)
        # self.relu_s1 = nn.ReLU(inplace=True)
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size=1,stride=1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            )
        
    def forward(self,x):
        xout = self.conv1_1(x)

        return xout

class Conv_3_3_and_1_1_BnReluLayer(nn.Module):
    def __init__(self,in_ch=3,out_ch=3):
        super(Conv_3_3_and_1_1_BnReluLayer,self).__init__()

        # self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
        # self.bn_s1 = nn.BatchNorm2d(out_ch)
        # self.relu_s1 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Sequential(
            nn.Conv2d(in_ch,in_ch,kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            )
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size=1,stride=1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            )
        
    def forward(self,x):
        x = self.conv3_3(x)
        xout = self.conv1_1(x)

        return xout
    

class Conv_1_1_BnLayer(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,dirate=1):
        super(Conv_1_1_BnLayer,self).__init__()

        # self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
        # self.bn_s1 = nn.BatchNorm2d(out_ch)
        # self.relu_s1 = nn.ReLU(inplace=True)
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size=1,stride=1, padding=0),
            nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True),
            )
        
    def forward(self,x):
        xout = self.conv1_1(x)

        return xout

class Conv_3_3_BnReluLayer(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,dirate=1):
        super(Conv_3_3_BnReluLayer,self).__init__()

        # self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
        # self.bn_s1 = nn.BatchNorm2d(out_ch)
        # self.relu_s1 = nn.ReLU(inplace=True)
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            )
        
    def forward(self,x):
        xout = self.conv1_1(x)

        return xout

class Conv_3_3_BnLayer(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,dirate=1):
        super(Conv_3_3_BnReluLayer,self).__init__()

        # self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
        # self.bn_s1 = nn.BatchNorm2d(out_ch)
        # self.relu_s1 = nn.ReLU(inplace=True)
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True),
            )
        
    def forward(self,x):
        xout = self.conv1_1(x)

        return xout
    
class UpSampleLayer(nn.Module):
    def __init__(self,in_ch,out_ch):
    # 512-1024-512
    # 1024-512-256
    # 512-256-128
    # 256-128-64
        super(UpSampleLayer, self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
        nn.Conv2d(in_channels=in_ch, out_channels=out_ch*2, kernel_size=3, stride=1,padding=1),
        nn.BatchNorm2d(out_ch*2),
        nn.ReLU(),
        nn.Conv2d(in_channels=out_ch*2, out_channels=out_ch*2, kernel_size=3, stride=1,padding=1),
        nn.BatchNorm2d(out_ch*2),
        nn.ReLU())
        self.upsample=nn.Sequential(
        nn.ConvTranspose2d(in_channels=out_ch*2,out_channels=out_ch,kernel_size=3,stride=2,padding=1,output_padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU())

    def forward(self,x,out):
        '''
        :param x: 输入卷积层
        :param out:与上采样层进行cat
        :return:
        '''
        x_out=self.Conv_BN_ReLU_2(x)
        x_out=self.upsample(x_out)
        cat_out=torch.cat((x_out,out),dim=1)
        return cat_out

class MSAC(nn.Module):
    
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(MSAC, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=3 * rate, dilation=3 * rate, bias=True),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, 3, 1, padding=7 * rate, dilation=7 * rate, bias=True),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=11 * rate, dilation=11 * rate, bias=True),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out)
        self.branch5_relu = nn.ReLU(inplace=True)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [b, c, row, col] = x.size()
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        # global_feature = self.branch5_bn(global_feature)
        # global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        # feature_sum = conv3x3_1 + conv3x3_2 + conv3x3_3
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        #		feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3], dim=1)
        result = self.conv_cat(feature_cat)
        return result
    
class ResSkipBlock_one(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 4, 1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels // 4, 3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels))

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))


    def forward(self, x):
        # x = F.interpolate(x, size=skip.size()[2:], mode=self.mode)
        # residual = x
        x = self.shortcut(x)
        # decoder_after_se = SELayer(channel=512)(x)
        residual = x
        # skip = self.conv2d3x3(skip)
        # encoder_after_se = SELayer(channel=512)(skip)

        # fm_after_concat = torch.cat([x, skip], dim=1)

        # print(fm_after_concat.size())

        # fm_after_se = SELayer(channel=512)(fm_after_concat)
        # residual = self.residual(fm_after_se)
        
        residual = self.residual(residual)

        # fm_after_concat = self.conv512(fm_after_concat)
        return F.relu(x + residual, inplace=True)
    
class ResSkipBlock_two(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels, stride=1, mode='bilinear'):
        super().__init__()
        self.mode = mode
        self.residual = nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels // 4, 1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels // 4, 3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels))

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels1, out_channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        self.conv2d3x3 = nn.Sequential(
            nn.Conv2d(in_channels2, out_channels, 3, stride=stride, bias=False, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.size()[2:], mode=self.mode)

        x = self.shortcut(x)
        # decoder_after_se = SELayer(channel=512)(x)

        skip = self.conv2d3x3(skip)
        # encoder_after_se = SELayer(channel=512)(skip)

        fm_after_concat = torch.cat([x, skip], dim=1)

        # print(fm_after_concat.size())

        # fm_after_se = SELayer(channel=512)(fm_after_concat)
        # residual = self.residual(fm_after_se)
        
        residual = self.residual(fm_after_concat)

        # fm_after_concat = self.conv512(fm_after_concat)
        return F.relu(x + residual, inplace=True)
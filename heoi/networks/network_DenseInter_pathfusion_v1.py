from turtle import forward
import cv2
import torch
import torch.nn as nn
from networks.ResNet import resnet18
from networks.base import NetworkBase

from options.train_opt import TrainOptions


class AttentionPrediction(NetworkBase):
    """Attention_model. """
    def __init__(self):
        super(AttentionPrediction, self).__init__()
        self.opt = TrainOptions().parse()
        
        self.gpu_num = len(self.opt.gpu_ids)  # gpu_num = 2
    
        "modality"
        self.fm_extractor_3channel = resnet18(pretrained=True, channel=3)
        self.fm_extractor_4channel = resnet18(pretrained=True, channel=4)
        
        "channel attention module"
        self.channel_attention = channel_attention(in_channel=512)
          
        "spatial attention module"
        self.spatial_attention = spatial_attention(kernel_size=7)
        
        "MLP"
        self.mlp = nn.Sequential(
            nn.Linear(512*2*7*7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),    # dropout = 0.5
            nn.Linear(512, 1),
            nn.ReLU(inplace=True),
            # nn.Dropout()
            )
      
        "Sigmoid"
        self.sigmoid = nn.Sigmoid()
        
   
    def forward(self, image, human_box_mask, obj_mask, human_skeleton_mask, human_body_part_mask):
        
        result = {'score': None}
        "human mask is 3*224*224  gray = 255"
        "human skeleton is 3*224*224  gray = 255"
        "obj_mask is 3*224*224  gray = 255"
        "human_body_part_mask is 1*224*224  gray = 255"
       
        box = human_box_mask[:,0:1,:,:]
        obj = obj_mask[:,0:1,:,:]
        skeleton = human_skeleton_mask[:,0:1,:,:]
        bodypart = human_body_part_mask
        
        box_obj = box + obj
        skeleton_obj = skeleton + obj
        bodypart_obj = bodypart + obj
        image_box_obj = torch.cat([image, box_obj],dim=1)
        image_skeleton_obj = torch.cat([image, skeleton_obj],dim=1)
        image_bodypart_obj = torch.cat([image, bodypart_obj], dim=1)
        
        "featue extraction"
        image_fm = self.fm_extractor_3channel.fm_extract(image)
        fm_in1 = self.fm_extractor_4channel.fm_extract(image_box_obj)
        fm_in2 = self.fm_extractor_4channel.fm_extract(image_skeleton_obj)
        fm_in3 = self.fm_extractor_4channel.fm_extract(image_bodypart_obj)
        
        fm_in = fm_in1 + fm_in2 + fm_in3
        
        hoi_fm = self.spatial_attention(fm_in)
        image_fm = self.channel_attention(image_fm)
        
        output = torch.cat([image_fm, hoi_fm],dim = 1)
        output = output.view(output.size(0), -1)
        output = self.mlp(output)
        output = self.sigmoid(output)
        result['score'] = output

        return result
    
    
    
class spatial_attention(nn.Module):
    # 初始化，卷积核大小为7*7
    def __init__(self, kernel_size=7):
        # 继承父类初始化方法
        super(spatial_attention, self).__init__()
        
        # 为了保持卷积前后的特征图shape相同，卷积时需要padding
        padding = kernel_size // 2
        # 7*7卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        # self.conv = nn.Conv2d(in_channels=2, out_channels=512, kernel_size=kernel_size,
                              # padding=padding, bias=False)
        self.conv = nn.Conv2d(in_channels=2, out_channels=512, kernel_size=kernel_size,
                              padding=padding, bias=False)

        # sigmoid函数
        self.sigmoid = nn.Sigmoid()
    
    # 前向传播
    def forward(self, inputs):
        
        # 在通道维度上最大池化 [b,512,h,w]  keepdim保留原有深度
        # 返回值是在某维度的最大值和对应的索引
        x_maxpool, _ = torch.max(inputs, dim=1, keepdim=True)   # 512,1,7
        
        # 在通道维度上平均池化 [b,512,h,w]
        x_avgpool = torch.mean(inputs, dim=1, keepdim=True)   # 512,1,7
        # 池化后的结果在通道维度上堆叠 [b,1024,h,w]
        x = torch.cat([x_maxpool, x_avgpool], dim=1)   # 512,2,7
        
        # 卷积融合通道信息 [b,2,h,w]==>[b,512,h,w]
        x = self.conv(x)
        # 空间权重归一化
        x = self.sigmoid(x)
        # 输入特征图和空间权重相乘
        outputs = inputs * x
        return outputs
    
    
class channel_attention(nn.Module):
    def __init__(self, in_channel, ratio=16):
        """ 通道注意力机制 同最大池化和平均池化两路分别提取信息,后共用一个多层感知机mlp,再将二者结合

        :param in_channel: 输入通道
        :param ratio: 通道降低倍率
        """
        super(channel_attention, self).__init__()
        # 平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 最大池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 通道先降维后恢复到原来的维数
        self.fc1 = nn.Conv2d(in_channel, in_channel // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channel // ratio, in_channel, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 平均池化
        # avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # 最大池化
        # max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
		# out = avg_out + max_out
        # return x*self.sigmoid(out)
        
        # 平均池化一支 (2,512,8,8) -> (2,512,1,1) -> (2,512/ration,1,1) -> (2,512,1,1)
        # (2,512,8,8) -> (2,512,1,1)
        avg = self.avg_pool(x)
        # 多层感知机mlp (2,512,8,8) -> (2,512,1,1) -> (2,512/ration,1,1) -> (2,512,1,1)
        # (2,512,1,1) -> (2,512/ratio,1,1)
        avg = self.fc1(avg)
        avg = self.relu1(avg)
        # (2,512/ratio,1,1) -> (2,512,1,1)
        avg_out = self.fc2(avg)

        # 最大池化一支
        # (2,512,8,8) -> (2,512,1,1)
        max = self.max_pool(x)
        # 多层感知机
        # (2,512,1,1) -> (2,512/ratio,1,1)
        max = self.fc1(max)
        max = self.relu1(max)
        # (2,512/ratio,1,1) -> (2,512,1,1)
        max_out = self.fc2(max)

        # (2,512,1,1) + (2,512,1,1) -> (2,512,1,1)
        out = avg_out + max_out
        return x*self.sigmoid(out)

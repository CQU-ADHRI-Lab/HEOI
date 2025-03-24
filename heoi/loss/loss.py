from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
from options.train_opt import TrainOptions
from torch.nn.functional import binary_cross_entropy, \
    binary_cross_entropy_with_logits, cross_entropy
# loss function: seven probability map --- 6 scale + 1 fuse
class Loss(nn.Module):
    def __init__(self, weight=[1.0] * 7):
        super(Loss, self).__init__()
        self.weight = weight

    def forward(self, x_list, label):
        loss = self.weight[0] * F.binary_cross_entropy(x_list[0], label)
        for i, x in enumerate(x_list[1:]):
            loss += self.weight[i + 1] * F.binary_cross_entropy(x, label)
        return loss

def CrossEntropyLoss(prediction, gt):
    loss = nn.CrossEntropyLoss(reduction='mean')
    return loss(prediction, gt)

def BinaryCrossEntropyLoss(prediction, gt):
    loss = nn.BCELoss(reduction='mean')
    return loss(prediction, gt)

class FocalLoss(nn.Module):
    
    def __init__(self,gamma=2.0,alpha=0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self,y_pred,y_true):
        bce = torch.nn.BCELoss(reduction = "none")(y_pred,y_true)
        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
        alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        modulating_factor = torch.pow(1.0 - p_t, self.gamma)
        loss = torch.mean(alpha_factor * modulating_factor * bce)
        return loss

def L2Loss(prediction, gt):
    loss = nn.MSELoss().cuda()
    return loss(prediction, gt)

def L1Loss(prediction, gt):
    loss = nn.L1Loss().cuda()
    return loss(prediction, gt)

# L2正则化
def L2LossRegular(model,alpha):
    l2_loss = torch.tensor(0.0, requires_grad=True)
    for name, param in model.named_parameters():
        if 'bias' not in name: #一般不对偏置项使用正则
            l2_loss = l2_loss + (0.5 * alpha * torch.sum(torch.pow(param, 2)))
    return l2_loss


# L1正则化
def L1LossRegular(model,beta):
    l1_loss = torch.tensor(0.0, requires_grad=True)
    for name, param in model.named_parameters():
        if 'bias' not in name:
            l1_loss = l1_loss +  beta * torch.sum(torch.abs(param))
    return l1_loss

# 将L2正则和L1正则添加到FocalLoss损失，一起作为目标函数
def focal_loss_with_regularization(model, y_pred,y_true):
    focal = FocalLoss()(y_pred,y_true) 
    l2_loss = L2LossRegular(model,0.001) #注意设置正则化项系数
    l1_loss = L1LossRegular(model,0.001)
    total_loss = focal + l2_loss + l1_loss
    return total_loss


def outside_loss(gt_map_batch, pre_map_batch):
    # heat_outside = 0
    # gt_map_batch = gt_map_batch.squeeze()
    # pre_map_batch = pre_map_batch.squeeze()
    # from (batch, 1, 224, 224) to (batch, 224, 224)
    # for i in range(batch_size):
    common_outside = pre_map_batch * (1.0 - gt_map_batch)
    heat_outside = common_outside.sum() / (1.0 - gt_map_batch).sum()
    return heat_outside


def inside_loss(gt_map_batch, pre_map_batch):
    # heat_outside = 0
    # gt_map_batch = gt_map_batch.squeeze()
    # pre_map_batch = pre_map_batch.squeeze()
    # from (batch, 1, 224, 224) to (batch, 224, 224)
    # for i in range(batch_size):
    # common_outside = pre_map_batch * (1.0 - gt_map_batch)
    common_inside = pre_map_batch * gt_map_batch
    # heat_outside = common_outside.sum() / (1.0 - gt_map_batch).sum()
    heat_inside = common_inside.sum() / gt_map_batch.sum
    return heat_inside

class WeightedBCE(nn.Module):
    
    def __init__(self, weights=[0.4, 0.6]):
        super(WeightedBCE, self).__init__()
        self.weights = weights

    def forward(self, logit_pixel, truth_pixel):
        # print("====",logit_pixel.size())
        logit = logit_pixel.view(-1)
        truth = truth_pixel.view(-1)
        assert(logit.shape==truth.shape)
        loss = F.binary_cross_entropy(logit, truth, reduction='none')
        pos = (truth>0.5).float()
        neg = (truth<0.5).float()
        pos_weight = pos.sum().item() + 1e-12
        neg_weight = neg.sum().item() + 1e-12
        loss = (self.weights[0]*pos*loss/pos_weight + self.weights[1]*neg*loss/neg_weight).sum()

        return loss
    
class WeightedDiceLoss(nn.Module):
    def __init__(self, weights=[0.5, 0.5]): # W_pos=0.8, W_neg=0.2
        super(WeightedDiceLoss, self).__init__()
        self.weights = weights

    def forward(self, logit, truth, smooth=1e-5):
        batch_size = len(logit)
        logit = logit.view(batch_size,-1)
        truth = truth.view(batch_size,-1)
        assert(logit.shape==truth.shape)
        p = logit.view(batch_size,-1)
        t = truth.view(batch_size,-1)
        w = truth.detach()
        w = w*(self.weights[1]-self.weights[0])+self.weights[0]
        # p = w*(p*2-1)  #convert to [0,1] --> [-1, 1]
        # t = w*(t*2-1)
        p = w*(p)
        t = w*(t)
        intersection = (p * t).sum(-1)
        union =  (p * p).sum(-1) + (t * t).sum(-1)
        dice  = 1 - (2*intersection + smooth) / (union +smooth)
        # print "------",dice.data

        loss = dice.mean()
        return loss
    
        
def weight_CELoss_batch(gt_map_batch, pre_map_batch):
    # gt_map_batch = gt_map_batch.squeeze()
    # pre_map_batch = pre_map_batch.squeeze()
    # from (batch, 1, 224, 224) to (batch, 224, 224)
    # loss = 0
    # for i in range(batch_size):
    attention_area = gt_map_batch.sum()
    weight = attention_area / (224 * 224)
    pre_map_batch = pre_map_batch.clamp(min=0.01, max=0.99)
    weight_loss = (1 - weight) * gt_map_batch * pre_map_batch.log() + weight * (
                1 - gt_map_batch) * (1 - pre_map_batch).log()
    loss = weight_loss.sum()
    return -loss


def weight_CELoss_Mean(gt_map_batch, pre_map_batch):
    assert gt_map_batch.size() == pre_map_batch.size()

    dim = list(gt_map_batch.size())
    dim_num = len(dim)
    pixel_num = 1
    for i in range(dim_num):
        pixel_num *= dim[i]

    attention_area = gt_map_batch.sum()
    weight = attention_area / pixel_num

    pre_map_batch = pre_map_batch.clamp(min=0.01, max=0.99)
    weight_loss = (1 - weight) * gt_map_batch * pre_map_batch.log() + weight * (
                1 - gt_map_batch) * (1 - pre_map_batch).log()

    # loss = weight_loss.sum()
    loss = weight_loss.mean()

    return -loss


def weight_CELoss_Sum(gt_map_batch, pre_map_batch):
    assert gt_map_batch.size() == pre_map_batch.size()

    dim = list(gt_map_batch.size())
    dim_num = len(dim)
    pixel_num = 1
    for i in range(dim_num):
        pixel_num *= dim[i]

    attention_area = gt_map_batch.sum()
    weight = attention_area / pixel_num

    pre_map_batch = pre_map_batch.clamp(min=0.01, max=0.99)
    weight_loss = (1 - weight) * gt_map_batch * pre_map_batch.log() + weight * (
                1 - gt_map_batch) * (1 - pre_map_batch).log()

    loss = weight_loss.sum()
    # loss = weight_loss.mean()

    return -loss


def vae_loss(x, x_reconst, mu, log_var):
    dim = list(x.size())
    dim_num = len(dim)
    pixel_num = 1
    for i in range(dim_num):
        pixel_num *= dim[i]

    xent_loss = F.binary_cross_entropy(x_reconst.detach(), x.detach())
    kl_div_loss = - 0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    kl_div_loss = kl_div_loss.mean()
    xent_loss = xent_loss.mean()
    return xent_loss + kl_div_loss


def dice_coefficient_loss(gt_tensor, pre_tensor):
    # dice = 2*(a and b)/(a or b)

    common = gt_tensor * pre_tensor
    gt_and_pre = common.sum()

    # gt_mask = gt_tensor>0
    square_gt = gt_tensor * gt_tensor
    square_pre = pre_tensor * pre_tensor
    square_sum_gt = square_gt.sum()
    square_sum_pre = square_pre.sum()

    loss = gt_and_pre / (square_sum_gt + square_sum_pre)

    return 1 - 2 * loss


if __name__ == '__main__':
    # a = np.random.random((3,4))
    # b = torch.from_numpy(a).float().cuda()

    a = np.zeros((3, 4))
    c = -torch.from_numpy(a)
    print(c.log())
    # print(torch.log(c))
    # print(weighted_cross_entropy_loss_for_heat_map(b,c))

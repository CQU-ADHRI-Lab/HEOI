from re import S
import torch
import torch.nn as nn
from collections import OrderedDict
from torch import optim
import torch.nn.functional as F
from model.base import BaseModel
from networks.a_factory_networks import NetworksFactory
from loss.loss import L1Loss, L1LossRegular

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


class CustomModel(BaseModel):
    def __init__(self, opt):
        super(CustomModel, self).__init__(opt)
        self._loss = None

        # create networks(encoder and decoder)
        self._init_create_networks()
        self._temporal_state = opt.temporal_state

        # init optimizer
        if self._is_train:
            self._init_train_optimizer()

        # load networks and optimizers
        if not self._is_train or self._opt.load_epoch > 0:
            self.load()

        self.result = {}
        
    def _init_create_networks(self):
        # generator network
        self._gpu_ids = [0]
        self._G = self._create_generator()
        # print(self._G)
        self._G = self._G.cuda(self._gpu_ids[0])

        if len(self._gpu_ids) == 1:
            self._G = self._G.to('cuda')
            self._G = self._G.to(self._gpu_ids[0])

        elif len(self._gpu_ids) > 1:
            self._G = nn.DataParallel(self._G, device_ids=self._gpu_ids)

    def _create_generator(self):
        # here, return is a network
        return NetworksFactory.get_by_name(self._opt.network_name)

    def _init_train_optimizer(self):
        self._current_lr_G = self._opt.lr_G
        # initialize optimizers
        self._optimizer_G = optim.Adam(filter(lambda p: p.requires_grad,
                                              self._G.parameters()),
                                       lr=self._current_lr_G,
                                       betas=(self._opt.G_adam_b1,
                                              self._opt.G_adam_b2))


    def set_input(self, input):  # input is train_batch
        self._input_frame = input['image']
        self._input_label = input['label']
        self.frame_name = input['frame_name']
        self.human_box_mask = input['human_box_mask']
        self.object_box_mask = input['object_box_mask']
        self.human_skeleton_mask = input['human_skeleton_mask']
        self.human_body_part_mask = input['human_body_part_mask']
        # self.index = input['index']
        self.box = input['box']
        
        
        self._input_label = self._input_label.float().cuda()
        self._input_frame = self._input_frame.float().cuda()
        self.human_box_mask = self.human_box_mask.float().cuda()
        self.object_box_mask = self.object_box_mask.float().cuda()
        self.human_body_part_mask = self.human_body_part_mask.float().cuda()
        self.human_skeleton_mask = self.human_skeleton_mask.float().cuda()
        return True

    def set_train(self):
        self._G.train()
        self._is_train = True

    def set_eval(self):
        self._G.eval()
        self._is_train = False

    def train_forward_backward(self):  # for training
        if self._is_train:            
            self._optimizer_G.zero_grad()
            result = self.G_forward_and_loss_compute()
            result['loss'].backward(retain_graph=True)
            self._optimizer_G.step()

    def test_forward(self):  # for testing
        if not self._is_train:
            self.G_forward_and_loss_compute()

    def G_forward_and_loss_compute(self):  # return training loss
        " for one output "
        # print(next(self._G.parameters()).device)
        self._loss = self._Tensor([0])
        self.l1_loss = self._Tensor([0])
        
        
        "object, human box, human skeleton, human body part"
        self.output = self._G.forward(self._input_frame, self.human_box_mask,self.object_box_mask, self.human_skeleton_mask, self.human_body_part_mask) 
        
        pre = self.output['score']
        
        label = self._input_label
    
        label = label.view(-1,1)
        
        a = torch.isnan(pre).sum().item()
        if a>0:
            print(f"is nan {a}")
        else:
            self.l1_loss += L1Loss(pre,label) + L1LossRegular(model=self._G, beta=0.001)
            
            self._loss = self.l1_loss
       
        "infer"
        if self._is_train:
            pass
        else:
            "Subject3_rgbd_images_arranging_objects_0510143426-0000-x1-y1-x2-y2-class-flag"
            for i in range(pre.size(0)):
                if a==0:
                    frame_name = self.frame_name[i]
                    box = self.box[i]
                    score = format(float(pre[i]), '.3f') #######  one outputs 
                    tem_result = f"{score}_{box}"
                    if frame_name not in self.result:
                        self.result[frame_name] = []
                        self.result[frame_name].append(tem_result)
                    else:
                        self.result[frame_name].append(tem_result)

        result = {'loss': self._loss,
                  'result': self.result}
        return result

    def get_current_scalars(self):  # current learning rate
        return OrderedDict([('lr_G', self._current_lr_G)])

    def save(self, label):
        # save networks
        self._save_network(self._G, 'G', label)

        # save optimizers
        self._save_optimizer(self._optimizer_G, 'G', label)

    def load(self):
        load_epoch = self._opt.load_epoch

        # load G
        self._load_network(self._G, 'G', load_epoch)

        if self._is_train:
            self._load_optimizer(self._optimizer_G, 'G', load_epoch)

    def update_learning_rate(self):
        # updated learning rate G, for example from 0.2 to 0.2-0.2/10
        lr_decay = self._opt.lr_G / self._opt.nepochs_decay  #
        self._current_lr_G -= lr_decay
        for param_group in self._optimizer_G.param_groups:
            param_group['lr'] = self._current_lr_G
        print('update G learning rate: %f -> %f' % (
            self._current_lr_G + lr_decay, self._current_lr_G))

    def get_current_errors(self):
        loss_dict = OrderedDict()
        loss_dict['gaze_loss'] = self.l1_loss.item()
        loss_dict['total_loss'] = self._loss.item()
        return loss_dict

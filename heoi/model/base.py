import os
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from collections import OrderedDict


class BaseModel(object):
# class BaseModel(nn.Module):

    def __init__(self, opt):
        # super(BaseModel, self).__init__()
        self._name = 'BaseModel'

        self._opt = opt
        self._gpu_ids = opt.gpu_ids
        self._is_train = opt.is_train

        self._Tensor = torch.cuda.FloatTensor if self._gpu_ids else torch.Tensor
        self._save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    @property
    def name(self):
        return self._name

    @property
    def is_train(self):
        return self._is_train

    def set_input(self, input):
        assert False, "set_input not implemented"

    def set_train(self):
        assert False, "set_train not implemented"

    def set_eval(self):
        assert False, "set_eval not implemented"

    def forward(self, keep_data_for_visuals=False):
        assert False, "forward not implemented"

    # used in test time, no backprop
    def test(self):
        assert False, "test not implemented"

    def get_image_paths(self):
        return {}

    def optimize_parameters(self):
        assert False, "optimize_parameters not implemented"

    def get_current_visuals(self):
        return {}

    def get_current_errors(self):
        return {}

    def get_current_scalars(self):
        return {}

    def save(self, label):
        assert False, "save not implemented"

    def load(self):
        assert False, "load not implemented"

    def _save_optimizer(self, optimizer, optimizer_label, epoch_label):
        save_filename = 'opt_epoch_%s_id_%s.pth' % (
        epoch_label, optimizer_label)
        save_path = os.path.join(self._save_dir, save_filename)
        torch.save(optimizer.state_dict(), save_path)
        print('saved optimizer: %s' % save_path)

    def _load_optimizer(self, optimizer, optimizer_label, epoch_label):
        load_filename = 'opt_epoch_%s_id_%s.pth' % (
        epoch_label, optimizer_label)
        load_path = os.path.join(self._save_dir, load_filename)
        assert os.path.exists(
            load_path), 'Weights file not found. Have you trained a model!? ' \
                        'We are not providing one' % load_path

        optimizer.load_state_dict(torch.load(load_path))
        print('loaded optimizer: %s' % load_path)

    def _save_network(self, network, network_label, epoch_label):
        save_filename = 'net_epoch_%s_id_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self._save_dir, save_filename)
        torch.save(network.state_dict(), save_path)
        print('saved net: %s' % save_path)

    def _load_network(self, network, network_label, epoch_label):
        load_filename = 'net_epoch_%s_id_%s.pth' % (epoch_label, network_label)
        load_path = os.path.join(self._save_dir, load_filename)
        print('load network of:', load_path)
        assert os.path.exists(
            load_path), 'Weights file not found. Have you trained a model!? ' \
                        'We are not providing one' % load_path

        # network.load_state_dict(torch.load(load_path))

        module = torch.load(load_path)
        # D = OrderedDict()
        if len(self._gpu_ids) == 1:
            D = OrderedDict()
            for layer_weight in module:
                layer_weight = layer_weight.split('.')
                if layer_weight[0] == 'module':
                    layer_weight_new = '.'.join(layer_weight[1:])
                else:
                    layer_weight_new = '.'.join(layer_weight)
                layer_weight = '.'.join(layer_weight)
                D[layer_weight_new] = module[layer_weight]
                D[layer_weight_new] = D[layer_weight_new].to(self._gpu_ids[0])
            network.load_state_dict(D)
        else:
            D = OrderedDict()
            for layer_weight in module:
                layer_weight = layer_weight.split('.')
                if layer_weight[0] != 'module':
                    layer_weight_new = '.'.join(['module'] + layer_weight)
                    # layer_weight_new = '.'.join(layer_weight)
                else:
                    layer_weight_new = '.'.join(layer_weight)
                layer_weight = '.'.join(layer_weight)
                D[layer_weight_new] = module[layer_weight]
            network.load_state_dict(D)

        print('loaded net: %s' % load_path)

    def update_learning_rate(self):
        pass

    def print_network(self, network):
        num_params = 0
        for param in network.parameters():
            num_params += param.numel()
        print(network)
        print('Total number of parameters: %d' % num_params)

    def _get_scheduler(self, optimizer, opt):
        # TODO: Add scheduler into the model
        if opt.lr_policy == 'lambda':
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter
                                 ) / float(opt.niter_decay + 1)
                return lr_l

            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif opt.lr_policy == 'step':
            scheduler = lr_scheduler.StepLR(optimizer,
                                            step_size=opt.lr_decay_iters,
                                            gamma=0.1)
        elif opt.lr_policy == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       mode='min',
                                                       factor=0.2,
                                                       threshold=0.01,
                                                       patience=5)
        else:
            return NotImplementedError(
                'learning rate policy [%s] is not implemented', opt.lr_policy)
        return scheduler

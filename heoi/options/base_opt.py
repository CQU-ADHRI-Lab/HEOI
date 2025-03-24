import argparse


class BaseOptions(object):
    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self._initialized = False

    def initialize(self):
          self._parser.add_argument('--load_epoch', type=int, default=-1,
                                   help='which epoch to load? set to -1 to use '
                                        'latest cached model')
          
          # GPU ID, Batch Size, Image Size
          self._parser.add_argument('--batch_size', type=int, default=64,
                                   help='input batch size')
          self._parser.add_argument('--image_size', type=int, default=224,
                                   help='input image size')
          self._parser.add_argument('--gpu_ids', type=str, default='0',
                                   help='gpu ids: e.g. 0  0,1,2, 0,2. '
                                        'use -1 for CPU')

          # check point
          self._parser.add_argument('--checkpoints_dir', type=str,
                                   default='',
                                   help='models are saved here')
          self._parser.add_argument('--name', type=str,
                                   default='CAD_DenseFusion_B_64_LR0001',
                                   help='name of the experiment. It decides '
                                        'where to store samples and models')

          # network
          self._parser.add_argument('--network_name', type=str,
                                   default='Dense_fusion',
                                   help= 'network name')
          self._parser.add_argument('--backbone', type=str,
                                   default='resnet18',
                                   help='resnet34, resnet50, vgg16, vgg19')

          # model
          self._parser.add_argument('--model_name', type=str,
                                   default='Dense_V1',
                                   help='model name')

          # dataset
          self._parser.add_argument('--dataset_mode', type=str,
                                   default='CAD',
                                   help='TIA, CAD')
          self._parser.add_argument('--data_root', type=str,
                                   default='/home/cqu/nzx/',
                                   help='path to dataset root')
          self._parser.add_argument('--n_threads_train', default=8, type=int,
                                   help='# threads for loading data')
          self._parser.add_argument('--n_threads_test', default=8, type=int,
                                   help='# threads for loading data')
  
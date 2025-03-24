from options.base_opt import BaseOptions
import os
from utils import util


class TrainOptions(BaseOptions):
    def __init__(self):
        super().__init__()
        self.is_train = True

    def initialize(self):
        BaseOptions.initialize(self)
        self._parser.add_argument('--training_mode', type=str, default='train',
                                  help='training_mode')
        self._parser.add_argument('--num_iters_validate', type=int, default=5,
                                  help='# batches to use when validating')
        self._parser.add_argument('--print_freq_s', type=int, default=20,
                                  help='frequency of showing training results '
                                       'on console')
        self._parser.add_argument('--display_freq_s', type=int, default=100,
                                  help='frequency [s] of showing training '
                                       'results on screen')
        self._parser.add_argument('--save_latest_freq_s', type=int, default=240,
                                  help='frequency of saving the latest results')

        # epoch
        self._parser.add_argument('--nepochs_no_decay', type=int, default=5,
                                  help='# of epochs at starting learning rate')
        self._parser.add_argument('--nepochs_decay', type=int, default=5,
                                  help='# of times to linearly decay learning '
                                       'rate to zero')

        # optimizer
        self._parser.add_argument('--poses_g_sigma', type=float, default=0.06,
                                  help='initial learning rate for adam')
        self._parser.add_argument('--lr_G', type=float, default=0.0001,
                                  help='initial learning rate for G adam')
        self._parser.add_argument('--G_adam_b1', type=float, default=0.9,
                                  help='beta1 for G adam')
        self._parser.add_argument('--G_adam_b2', type=float, default=0.999,
                                  help='beta2 for G adam')
        self._parser.add_argument('--lr_policy', type=str, default='lambda',
                                  help='lr policy, you can choose lambda, '
                                       'step, plateau')

    def parse(self):
        if not self._initialized:
            self.initialize()
        self._opt = self._parser.parse_args()

        # set is train or set
        self._opt.is_train = self.is_train

        # set and check load_epoch
        self._set_and_check_load_epoch()

        # get and set gpus
        self._get_set_gpus()

        args = vars(self._opt)

        # save args to file
        self._save(args)

        return self._opt

    def _set_and_check_load_epoch(self):
        # default of self._opt.load_epoch is -1, if -1, we find whether
        # there are save model, becasue the saved model is with epoch index,
        # then we can find the epoch index that has been trained

        # if we do not set default value, we try to find the epoch number
        # which is set, if found, that is fine
        # and self._opt.load_epoch will be the set value, if not found,
        # print error information

        models_dir = os.path.join(self._opt.checkpoints_dir, self._opt.name)
        if os.path.exists(models_dir):
            if self._opt.load_epoch == -1:
                load_epoch = 0
                for file in os.listdir(models_dir):
                    if file.startswith("net_epoch_"):
                        load_epoch = max(load_epoch, int(file.split('_')[2]))
                self._opt.load_epoch = load_epoch
            else:
                found = False
                for file in os.listdir(models_dir):
                    if file.startswith("net_epoch_"):
                        found = int(file.split('_')[2]) == self._opt.load_epoch
                        if found:
                            break
                assert found, f'Model for epoch {self._opt.load_epoch} ' \
                              f'not found'
        else:
            assert self._opt.load_epoch < 1, f'Model for epoch ' \
                                             f'{self._opt.load_epoch} not found'
            self._opt.load_epoch = 0

    def _get_set_gpus(self):
        # get gpu ids
        str_ids = self._opt.gpu_ids.split(',')
        self._opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self._opt.gpu_ids.append(id)

    def _print(self, args):
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

    def _save(self, args):
        expr_dir = os.path.join(self._opt.checkpoints_dir, self._opt.name)
        util.mkdir(expr_dir)
        expr_dir =  os.path.join(self._opt.checkpoints_dir, self._opt.name,'exp_result')
        util.mkdir(expr_dir)
        file_name = os.path.join(expr_dir, 'opt_%s.txt' % (
            'train' if self.is_train else 'test'))
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')

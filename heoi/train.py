import time
import os,cv2
import random
import matplotlib.pyplot as plt
from collections import OrderedDict

import numpy
from options.train_opt import TrainOptions
from data_n.factory_dataloader import CustomDatasetDataLoader
from model.factory_models import ModelsFactory
from utils.tb_visualizer import TBVisualizer
from torch.optim import lr_scheduler

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
ids = [0]

class Train:
    def __init__(self):
        self._opt = TrainOptions().parse()
        data_loader_train = CustomDatasetDataLoader(self._opt,
                                                    is_for_train=True)
        data_loader_test = CustomDatasetDataLoader(self._opt,
                                                   is_for_train=False)


        self._dataset_train = data_loader_train.load_data()
        self._dataset_test = data_loader_test.load_data()

        self._dataset_train_size = len(data_loader_train)
        self._dataset_test_size = len(data_loader_test)
        
        self.plot_step = int(self._dataset_train_size // (self._opt.batch_size*30))
        print(f"plot step = {self.plot_step}")
        print(f'train samples: {self._dataset_train_size}\n')
        print(f'test samples: {self._dataset_test_size}\n')

        self.batch_num = self._dataset_train_size // self._opt.batch_size
        print(f'Train Batches: 'f'{self.batch_num}\n')
        print(f'Test Batches: 'f'{self._dataset_test_size // self._opt.batch_size}\n')

        self._model = ModelsFactory.get_by_name(self._opt.model_name, self._opt)
        self._tb_visualizer = TBVisualizer(self._opt)  # log here

        #  save and plot loss, PDF
        self.global_loss = []
        self.val_loss = []
        self.lr = []
        self.plot_loss_init()
        
        self.loss_save_name = os.path.join(self._opt.checkpoints_dir,self._opt.name, 'exp_result','train_loss.pdf')
        self.lr_save_name = os.path.join(self._opt.checkpoints_dir,self._opt.name, 'exp_result','lr.txt')
        
        self._train()

    def _train(self):
        self._total_steps = self._opt.load_epoch * self._dataset_train_size
        self._iters_per_epoch = self._dataset_train_size / self._opt.batch_size
        self._last_display_time = None
        self._last_save_latest_time = None
        self._last_print_time = time.time()

        # test_error_epochs = []
        
        for i_epoch in range(self._opt.load_epoch + 1,self._opt.nepochs_no_decay + self._opt.nepochs_decay + 1):
            epoch_start_time = time.time()
            self.i_epoch = i_epoch
            # validation
            self.test_errors = self._test_epoch(i_epoch)
            self.val_loss.append(self.test_errors)
            # test_error_epochs.append(test_error)
            
            # train epoch
            self._train_epoch(i_epoch)

            # print each epoch's index number and time consumption
            time_epoch = time.time() - epoch_start_time
            print(f'End of epoch {i_epoch} / 'f'{self._opt.nepochs_no_decay + self._opt.nepochs_decay} \t 'f'Time Taken: {time_epoch} sec\n')

            # update learning rate            
            self.update_lr_policy_warmup()
            # self.record_lr()

            # display train
            "torch vision"
            self._display_visualizer_train(self._total_steps)

    def _train_epoch(self, i_epoch):
        batch_num = 0
        self._model.set_train()

        train_errors = OrderedDict()

        for i_train_batch, train_batch in enumerate(self._dataset_train):
            # print(i_train_batch)
            iter_start_time = time.time()

            do_visuals = self._last_display_time is None or time.time() - self._last_display_time > self._opt.display_freq_s
            do_print_terminal = time.time() - self._last_print_time > self._opt.print_freq_s

            # train model
            if not self._model.set_input(train_batch):
                continue

            self._model.train_forward_backward()

            # update epoch info
            self._total_steps += self._opt.batch_size
            batch_num += 1

            # update error information
            errors = self._model.get_current_errors()

            # store current batch errors
            for k, v in errors.items():
                if k in train_errors:
                    train_errors[k] += v
                else:
                    train_errors[k] = v

            #  display terminal
            if do_print_terminal:
                self._display_terminal(iter_start_time, i_epoch,
                                       i_train_batch, do_visuals)
                self._last_print_time = time.time()

            # reset time
            if do_visuals:
                self._last_display_time = time.time()

            # TODO: Plot loss
            if i_train_batch % self.plot_step == 0:
                self.plot_loss(errors, save_flag=1)

            # save model
            # print(f"{i_train_batch}-{self._dataset_train_size}")
            if i_train_batch == self.batch_num-1:
            # if self._last_save_latest_time is None or time.time() - \
                    # self._last_save_latest_time > self._opt.save_latest_freq_s:
                self._model.save(i_epoch)
                # self._last_save_latest_time = time.time()
                # normalize errors
            
        " draw loss segment"
        self.plot_loss_seg()
            
        for k in train_errors:
            train_errors[k] /= batch_num
            print('(T, epoch: %d) %s:%s \n' % (i_epoch, k, train_errors[k]))
       
    
    def update_lr_policy_warmup(self):
          
        mile_stone_epochs = [self._opt.nepochs_no_decay, self._opt.nepochs_decay]
        big_lr = self._opt.lr_G
        # each_epoch_add_lr = big_lr / mile_stone_epochs[0]
        each_epoch_minus_lr = big_lr / ( mile_stone_epochs[1] + 1)
                
        for param_group in self._model._optimizer_G.param_groups:
            current_lr_G= param_group['lr']
        self.lr.append(current_lr_G)
        
        if self.i_epoch<= mile_stone_epochs[0]:
            # lr = self.i_epoch*each_epoch_add_lr
            lr = current_lr_G   
            
        if self.i_epoch> mile_stone_epochs[0]:
            lr = current_lr_G - each_epoch_minus_lr
            
        for param_group in self._model._optimizer_G.param_groups:
            param_group['lr'] = lr
            current_lr_G= param_group['lr']
            
            
        print(f'epoch = {self.i_epoch}, last_epoch_lr = {self.lr[-1]:.7f}, current_epoch_lr = {current_lr_G:.7f},  last_epoch_valloss = {self.test_errors}\n')
        self.f_w = open(self.lr_save_name , 'a')
        self.f_w.write(f'epoch = {self.i_epoch}, last_epoch_lr = {self.lr[-1]:.7f}, current_epoch_lr = {current_lr_G:.7f},  last_epoch_valloss = {self.test_errors}\n')
        self.f_w.close()
        
    
    def plot_loss_init(self):
        plt.ion()
        fig = plt.figure()

        self.ax1 = fig.add_subplot(211)
        self.ax1.set_title('Global Loss')
      

    def plot_loss(self, loss_dict, save_flag):
        for k, v in loss_dict.items():
            if k == 'total_loss':
                self.global_loss.append(v)
        if len(self.global_loss) > 0:
            self.ax1.plot(range(1, len(self.global_loss) + 1), self.global_loss)
        if save_flag:
            plt.savefig(self.loss_save_name, dpi=30)

        
    def plot_loss_seg(self, ):
        
        if len(self.global_loss) > 0:
            self.ax1.axvline(len(self.global_loss))
        save_flag = 1
        if save_flag:
            plt.savefig(self.loss_save_name, dpi=30)
            
            
    def _display_terminal(self, iter_start_time, i_epoch, i_train_batch,
                          visuals_flag):
        errors = self._model.get_current_errors()
        t = (time.time() - iter_start_time) / self._opt.batch_size
        self._tb_visualizer.print_current_train_errors(i_epoch, i_train_batch,
                                                       self._iters_per_epoch,
                                                       errors, t, visuals_flag)

    def _display_visualizer_train(self, total_steps):
        self._tb_visualizer.display_current_results(
            self._model.get_current_visuals(), total_steps, is_train=True, save_visuals= True)

        self._tb_visualizer.plot_scalars(self._model.get_current_errors(),
                                         total_steps, is_train=True)
        # learning rate
        self._tb_visualizer.plot_scalars(self._model.get_current_scalars(),
                                         total_steps, is_train=True)

    def _test_epoch(self, i_epoch):
        val_start_time = time.time()

        self._model.set_eval()
        val_errors = OrderedDict()
        val_total_error = 0
        for i_val_batch, val_batch in enumerate(self._dataset_test):
            if i_val_batch > self._opt.num_iters_validate:
                break
            # evaluate model
            if not self._model.set_input(val_batch):
                continue
            self._model.test_forward()
            errors = self._model.get_current_errors()

            # store current batch errors
            message = ''
            for k, v in errors.items():
                # errors is a dict, each value is a for a batch size
                # print(v)
                if k in val_errors:
                    val_errors[k] += v
                else:
                    val_errors[k] = v
                    
                " add_flag 20220530 "
                if k == 'total_loss':
                    val_total_error += v
                
                message += '%s:%.3f ' % (k, v)
            print('Validation Batch: ', i_val_batch, '/',
                  self._dataset_test_size // self._opt.batch_size, message)

        # normalize errors
        for k in val_errors:
            val_errors[k] /= self._opt.num_iters_validate

        # print loss in the terminal and write the loss into txt
        t = (time.time() - val_start_time)
        self._tb_visualizer.print_current_validate_errors(i_epoch, val_errors, t)

        # set model back to train
        self._model.set_train()
        
        return round(val_total_error,3)


if __name__ == "__main__":
    print('okay')
    random.seed(27)
    Train()

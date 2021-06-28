


import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch.nn as nn
from torch.optim import lr_scheduler


from collections import OrderedDict

from network.u2net import U2NET

from utils.losses import muti_ce_loss_fusion
from utils.metrics import IoU

import utils.utils as utils

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor


class Trainer():
    def __init__(self, opt):
        
        self.opt = opt
        self.device = self.opt['device']

        self.metric = IoU(self.opt['network']['n_classes'])
        
        self.model = U2NET(self.opt['network']['input_channel'], self.opt['network']['n_classes']).to(self.device)
#         util.init_weights(self.G, init_type='kaiming', scale=0.1)
        if self.opt['path']['pretrain_model']:
            self.model.load_state_dict(torch.load(
                self.opt['path']['pretrain_model']), strict=True)
            
        self.model.train()
        
        self.log_dict = OrderedDict()

        self.optim_params = [
            v for k, v in self.model.named_parameters() if v.requires_grad]
        self.opt_model = torch.optim.Adam(self.optim_params, lr=self.opt['train']['lr'], betas=(
            self.opt['train']['b1'], self.opt['train']['b2']))

        self.scheduler = lr_scheduler.MultiStepLR(
            self.opt_model, self.opt['train']['lr_steps'], self.opt['train']['lr_gamma'])

    def update_learning_rate(self):
        self.scheduler.step()

    def get_current_log(self):
        return self.log_dict

    def get_current_learning_rate(self):
        return self.scheduler.get_last_lr()[0]

    def load_model(self, step, strict=True):
        self.model.load_state_dict(torch.load(
            f"{self.opt['path']['checkpoints']['models']}/{step}.pth"), strict=strict)
   
    def resume_training(self, resume_state):
        '''Resume the optimizers and schedulers for training'''

        resume_optimizer = resume_state['optimizer']
        resume_scheduler = resume_state['scheduler']

        self.opt_model.load_state_dict(resume_optimizer)
        self.scheduler.load_state_dict(resume_scheduler)

    def save_network(self, network, network_label, iter_step):

        utils.mkdir(self.opt['path']['checkpoints']['models'])
        save_filename = '{}_{}.pth'.format(iter_step, network_label)
        save_path = os.path.join(
            self.opt['path']['checkpoints']['models'], save_filename)

        if isinstance(network, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def save_model(self, epoch, current_step):
        self.save_network(self.model, 'model', current_step)
        self.save_training_state(epoch, current_step)

    def save_training_state(self, epoch, iter_step):
        '''Saves training state during training, which will be used for resuming'''
        state = {'epoch': epoch, 'iter': iter_step,
                 'scheduler': [], 'optimizer': []}
        state['scheduler'].append(self.scheduler.state_dict())
        state['optimizer'].append(self.opt_model.state_dict())
        save_filename = '{}.state'.format(iter_step)
        utils.mkdir(self.opt['path']['checkpoints']['states'])
        save_path = os.path.join(
            self.opt['path']['checkpoints']['states'], save_filename)
        torch.save(state, save_path)

    def train(self, train_batch, step):

        self.images = train_batch['image'].to(self.device)
        self.masks = train_batch['mask'].to(self.device)
        


        self.opt_model.zero_grad()

        d0, d1, d2, d3, d4, d5, d6 = self.model(self.images)
        
        loss2, loss = muti_ce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, self.masks)


        loss.backward()
        self.opt_model.step()

        # set log
        self.log_dict['loss2'] = loss2.item()
        self.log_dict['loss'] = loss.item()
  
    def validate(self, val_batch, current_step):
        avg_dice_coef = 0.0
        idx = 0
        self.metric.reset()
        for _, val_data in enumerate(val_batch):
            idx += 1
            img_name = os.path.splitext(
                os.path.basename(val_data['image_path'][0]))[0]
            img_dir = os.path.join(
                self.opt['path']['checkpoints']['val_image_dir'], img_name)
            utils.mkdir(img_dir)

            self.val_image = val_data['image'].to(self.device)
            self.val_mask = val_data['mask'].to(self.device)

            self.model.eval()
            with torch.no_grad():
                d0, d1, d2, d3, d4, d5, d6 = self.model(self.val_image)
            self.model.train()
            
            self.metric.add(d0, self.val_mask)
            
            pred = torch.argmax(d0, 1)
            
#             out_mask = utils.tensor2img(pred)  # uint8
#             gt_mask = utils.tensor2img(self.val_mask)  # uint8
#             border = np.ones((out_mask.shape[0], 5, 3))*255
#             imgs_comb = np.hstack((out_mask, border.astype(np.uint8), gt_mask))
#             im = Image.fromarray(imgs_comb)
#             im.save(os.path.join(img_dir, f'{img_name}__{current_step}.png'))

#             avg_dice_coef += dice_coeff(d0[:, [1], :, :], self.val_mask[:, [1], :, :])

#         avg_dice_coef = avg_dice_coef / idx
        
        return self.metric.value()

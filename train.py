
import sys
import torch
import math
import random
import numpy as np

from torch.utils.tensorboard import SummaryWriter
import logging

import utils.utils as utils
import utils.data as data
from utils.trainer import Trainer as Model
import wandb
wandb.login()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    
    wandb.init(project="U2NET_IOU")
    
    config_path = sys.argv[1]
    opt = utils.load_yaml(config_path)

    if opt['path']['resume_state']:  # resuming training
        resume_state = torch.load(opt['path']['resume_state'])

    else:
        resume_state = None
#         utils.mkdir(opt['path']['log'])


    logger = logging.getLogger('base')

    set_random_seed(0)

    # tensorboard log
    writer = SummaryWriter(log_dir= opt['path']['tb_logger'])

    torch.backends.cudnn.benckmark = True
    
    data_root = opt['path']['data_root']
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = data.create_dataset(dataset_opt, data_root, True)
            train_size = int(
                math.ceil(len(train_set) / dataset_opt['batch_size']))
            logger.info('Number of train images: {:,d}, iters per epoch: {:,d}'.format(
                len(train_set), train_size))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                total_epochs, total_iters))
            train_loader = data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'valid':
            val_set = data.create_dataset(dataset_opt, data_root, False)
            val_loader = data.create_dataloader(val_set, dataset_opt, phase)
            logger.info('Number of validation images in [{:s}]: {:d}'.format(dataset_opt['name'],
                                                                             len(val_set)))
        else:
            raise NotImplementedError(
                'Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    # create model

    model = Model(opt)

    # resume training
    if resume_state:
        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.load_model(current_step)
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    # training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(
        start_epoch, current_step))

    for epoch in range(start_epoch, total_epochs):
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break

            # training
            model.train(train_data, current_step)

            # update learning rate
            model.update_learning_rate()
            
            logs = model.get_current_log()
            
            wandb.log({"Loss/loss": logs['loss']})
            wandb.log({"Loss/loss2": logs['loss2']})
            
            
            # log
            if current_step % opt['train']['print_freq'] == 0:
                
                logs = model.get_current_log()
                message = '<epoch:{:3d}, iter:{:8d}, lr:{:.3e}> '.format(
                    epoch, current_step, model.get_current_learning_rate())
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    # tensorboard logger
                    writer.add_scalar(k, v, current_step)
   
                logger.info(message)
                
            if current_step % opt['train']['val_freq'] == 0:
                
                iou, miou = model.validate(val_loader, current_step)

                # log
                logger.info('# Validation # mIOU: {:.4f} # IOU BG: {:.4f}, Vessel: {:.4f}, Tool: {:.4f} Fetus: {:.4f}'.format(miou.item(), iou[0].item(), iou[1].item(), iou[2].item(), iou[3].item()))

                # tensorboard logger
                writer.add_scalar('Metrics/mIOU', miou.item(), current_step)
                writer.add_scalar('Metrics/IOU_BG', iou[0].item(), current_step)
                writer.add_scalar('Metrics/IOU_Vessel', iou[1].item(), current_step)
                writer.add_scalar('Metrics/IOU_Tool', iou[2].item(), current_step)
                writer.add_scalar('Metrics/IOU_Fetus', iou[3].item(), current_step)
                
                
                # wandb log
                wandb.log({"Metric/mIOU": miou.item()})

                wandb.log({"Metric/IOU_BG": iou[0].item()})
                wandb.log({"Metric/IOU_Vessel": iou[1].item()})
                wandb.log({"Metric/IOU_Tool": iou[2].item()})
                wandb.log({"Metric/IOU_Fetus": iou[3].item()})
                
                
             # save models and training states
            if current_step % opt['train']['save_step'] == 0:
                logger.info('Saving models and training states.')
                model.save_model(epoch, current_step)
      # Mark the run as finished
    wandb.finish()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    main()


import random
import numpy as np
import torch

import torch.nn.functional as F
import torchvision.transforms.functional as TF
import utils.utils as utils
import cv2



def create_dataset(dataset_opt, data_root, transform):

    dataset = Dataset(dataset_opt, data_root, transform)

    return dataset


def create_dataloader(dataset, dataset_opt, phase):

    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['shuffle'],
            num_workers=dataset_opt['n_workers'],
            drop_last=True)
    else:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True)
    
    
    
    
    
class Dataset(torch.utils.data.Dataset):

    def __init__(self, opt, data_root, transform=True):
        
        super(Dataset, self).__init__()
        
        self.opt = opt
        self.data_root = data_root
        self.transform = transform

        self.image_paths, self.mask_paths = utils.get_image_paths(opt['dir_list'], data_root)

        if self.image_paths and self.mask_paths:
            assert len(self.image_paths) == len(self.mask_paths), \
                'Images and Masks have different number of images - {}, {}.'.format(
                len(self.image_paths), len(self.mask_paths))
    
    
    def __getitem__(self, index):

        # get image
        image_path = self.image_paths[index]
        image = utils.read_img(image_path)

        mask_path = self.mask_paths[index]
        mask = utils.read_img(mask_path)

        if self.transform == True:
            
            
            image, mask = utils.transform(self.opt, [image, mask])
        image = TF.to_tensor(np.array(image))
        mask = torch.from_numpy(np.array(mask))

        return {'image': image, 'mask': mask, 'image_path': image_path, 'mask_path': mask_path}

    def __len__(self):
        return len(self.image_paths)
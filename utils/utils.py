





import os
import yaml
import glob
import random

import PIL.Image as Image
import numpy as np

import torch
from torchvision.utils import make_grid
from torchvision import transforms
import torchvision.transforms.functional as TF






IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

def load_yaml(load_path):
    """load yaml file"""
    with open(load_path, 'r') as f:
        loaded = yaml.load(f, Loader=yaml.Loader)

    return loaded

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
        
def get_image_paths(dir_list, data_root):
    
    assert os.path.isdir(data_root), '{:s} is not a valid directory'.format(data_root)
    
    image_path_list = []
    mask_path_list = []
    
    for path in dir_list:
        image_path = glob.glob(os.path.join(data_root, path, 'images', '*'))
        mask_path = glob.glob(os.path.join(data_root, path, 'labels', '*'))
        for img_fname, mask_fname in zip(image_path, mask_path):
            if is_image_file(img_fname) and is_image_file(mask_fname):
                image_path_list.append(img_fname)
                mask_path_list.append(mask_fname)
                
    assert image_path_list, '{:s} has no valid image file'.format(data_root)
    assert mask_path_list, '{:s} has no valid mask file'.format(data_root)
    
    return image_path_list, mask_path_list

    
    
    
def read_img(path):

    img = Image.open(path)
    
    return img


def augmentation(opt, images):
    
    image, mask = images

    hflip = opt['flip'] and random.random() < 0.5
    vflip = opt['flip'] and random.random() < 0.5
    rot = opt['rotation'] and random.random() < 0.5

    def _augment(img, mask):
        if hflip:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        if vflip:
            img = TF.vflip(img)
            mask = TF.vflip(mask)
        if rot:
            angle = torch.randint(-45, 45, (1, )).item()
            img = TF.rotate(img, angle)
            mask = TF.rotate(mask, angle)
        return img, mask

    return _augment(image, mask) 

def transform(opt, images, augment=True):
    
    image, mask = images
    resize = transforms.Resize(size=(opt['resize'], opt['resize']))
    image = resize(image)
    mask = resize(mask)
    
    if augment == True:
        image, mask = augmentation(opt, [image, mask])
    #Random crop
    i, j, h, w = transforms.RandomCrop.get_params(
        image, output_size=(opt['crop'], opt['crop']))
    image = TF.crop(image, i, j, h, w)
    mask = TF.crop(mask, i, j, h, w)

#     if augment == True:
#         image, mask = augmentation(opt, [image, mask])

    
    return image, mask

def get_colormap():
    """
    Returns FetReg colormap
    """
    colormap = np.asarray(
        [
            [0, 0, 0],   # 0 - background 
            [255, 0, 0], # 1 - vessel
            [0, 0, 255], # 2 - tool
            [0, 255, 0], # 3 - fetus

        ]
        )
    return colormap

def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    
    
    colormap = get_colormap()    
    n_dim = tensor.dim()
    
    tensor = tensor.squeeze().float().cpu()

    mask_rgb = np.zeros(tensor.shape[:2] + (3,), dtype=np.uint8)
    for cnt in range(len(colormap)):
        mask_rgb[tensor == cnt] = colormap[cnt]
        
    return mask_rgb.astype(out_type)
    
#     colormap = get_colormap()
    
#     tensor = tensor.squeeze().float().cpu().clamp_(*min_max)
#     tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])
    
#     n_dim = tensor.dim()
    
#     if n_dim == 4:
#         n_img = len(tensor)
#         img_np = make_grid(tensor, nrow=int(
#             math.sqrt(n_img)), normalize=False).numpy()
#         img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
#     elif n_dim == 3:
#         img_np = tensor.numpy()
#         img_np = np.transpose(img_np, (1, 2, 0))  # HWC, BGR
#         img_np = np.argmax(img_np, axis=2)
#     elif n_dim == 2:
#         img_np = tensor.numpy()
#     else:
#         raise TypeError(
#             'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
#     mask_rgb = np.zeros(img_np.shape[:2] + (3,), dtype=np.uint8)
#     for cnt in range(len(colormap)):
#         mask_rgb[img_np == cnt] = colormap[cnt]
        
#     return mask_rgb.astype(out_type)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
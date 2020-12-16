import os
import copy
import random
import numpy as np
import torch
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data.dataset import Dataset

def get_path(dir, frame, side):
    side = 2 if side == 'l' else 3
    return os.path.join('data', dir, 'image_{:02d}'.format(side), 'data', '{:010d}.png'.format(frame))

class KITTIDatasetTrain(Dataset):
    def __init__(self, split, config):
        super(KITTIDatasetTrain, self).__init__()
        self.config = config

        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        
        with open(split, 'r') as f:
            self.lines = [x.strip().split() for x in f.readlines()]

    def __getitem__(self, index):
        dirname, frame, side = self.lines[index]
        imgs = [Image.open(get_path(dirname, int(frame)+x, side)) for x in [-1, 0, 1]]
        imgs_aug = copy.deepcopy(imgs)
        
        flip_p = random.random()
        color_p = random.random()
        img_transform = transforms.Compose([
            transforms.Resize((self.config['img_hgt'], self.config['img_wid'])),
            transforms.RandomHorizontalFlip(p=1.0 if flip_p > 0.5 else 0.0),
            transforms.ToTensor()
        ])
        img_aug_transform = transforms.Compose([
            transforms.Resize((self.config['img_hgt'], self.config['img_wid'])),
            transforms.RandomHorizontalFlip(p=1.0 if flip_p > 0.5 else 0.0),
            transforms.RandomApply(
                [transforms.ColorJitter(
                    brightness = (0.8, 1.2),
                    contrast = (0.8, 1.2),
                    saturation = (0.8, 1.2),
                    hue = (-0.1, 0.1)
                )],
                p = 1.0 if color_p > 0.5 else 0.0
            ),
            transforms.ToTensor()
        ])

        imgs = [img_transform(x) for x in imgs]
        imgs_aug = [img_aug_transform(x) for x in imgs_aug]

        K = []
        invK = []
        for scale in range(self.config['num_scales']):
            k = self.K.copy()
            k[0, :] *= self.config['img_wid'] // (2 ** scale)
            k[1, :] *= self.config['img_hgt'] // (2 ** scale)
            invk = np.linalg.pinv(k)
            K.append(torch.from_numpy(k))
            invK.append(torch.from_numpy(invk))

        data = {
            'img': imgs[1],
            'img_aug': imgs_aug[1],
            'prev_img': imgs[0],
            'prev_img_aug': imgs_aug[0],
            'next_img': imgs[2],
            'next_img_aug': imgs_aug[2],
            'K': K,
            'invK': invK,
        }
        return data

    def __len__(self):
        return len(self.lines)

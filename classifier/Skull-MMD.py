# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 16:40:06 2023

@author: blair
"""

import os
import json
import argparse
import yaml
import glob
from tqdm import trange
import numpy as np
import pandas as pd
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torchvision import datasets, transforms

# let's import our own classes and functions!
os.chdir(r"C:\Users\blair\OneDrive - UBC\Skull-Adapt\classifier")
from util import init_seed
from model_cam import CustomResNet18


import torch
from tqdm import tqdm

from pytorch_adapt.containers import Models, Optimizers
from pytorch_adapt.datasets import (
    DataloaderCreator,
    CombinedSourceAndTargetDataset,
    SourceDataset,
    TargetDataset,
)
from pytorch_adapt.hooks import ClassifierHook, BNMHook, BSPHook
from pytorch_adapt.models import Discriminator, mnistC, mnistG
from pytorch_adapt.utils.common_functions import batch_to_device
from pytorch_adapt.validators import IMValidator

parser = argparse.ArgumentParser(description='Train deep learning model.')
parser.add_argument('--config', help='Path to config file', default='../configs/skull_MMD.yaml')
args = parser.parse_args()

# load config
print(f'Using config "{args.config}"')
cfg = yaml.safe_load(open(args.config, 'r'))

# init random number generator seed (set at the start)
init_seed(cfg.get('seed', None))
exp = cfg['experiment_name']

# check if GPU is available
device = cfg['device']
if device != 'cpu' and not torch.cuda.is_available():
    print(f'WARNING: device set to "{device}" but CUDA not available; falling back to CPU...')
    cfg['device'] = 'cpu'

# initialize data loaders for training and validation set

# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# import cv2

# Define augmentations
# transform_t = A.Compose([
#     A.Resize(224, 224),
#     A.HorizontalFlip(p=0.5),
#     A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.8),
#     A.GaussNoise(var_limit=(5.0, 10.0), p=0.8),  # Add Gaussian noise with specified mean and std
#     A.ShiftScaleRotate(shift_limit=0.1, border_mode=cv2.BORDER_REPLICATE),
#     ToTensorV2(),
# ])


# class RandomAffineWithNoise(transforms.RandomAffine):
#   def __init__(self, degrees, translate, scale, shear, fill, mean=0.0, std=0.01):
#     super().__init__(degrees, translate, scale, shear, fill)
#     self.mean = mean
#     self.std = std

#   def _get_params(self, img, degrees, translate, scale, shear):
#     params = super()._get_params(img, degrees, translate, scale, shear)
#     noise = torch.randn_like(img) * self.std + self.mean
#     return params, noise

# transform_t = transforms.Compose([
#   transforms.Resize((224, 224)),
#   transforms.RandomHorizontalFlip(),
#   transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
#   RandomAffineWithNoise(degrees=0, translate=(0.1, 0.1), fill=(48, 48, 48), shear = None, scale = None, mean=0.0, std=0.01),
#   transforms.ToTensor()
# ])

transform_t = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), fill=(48, 48, 48)),
    transforms.ToTensor()
])

transform_v = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Create the ImageFolder dataset for training data
src_train = SourceDataset(datasets.ImageFolder(root=cfg['src_train'],
                                     transform=transform_t))
src_val = SourceDataset(datasets.ImageFolder(root=cfg['src_val'],
                                     transform=transform_v))
target_train = TargetDataset(datasets.ImageFolder(root=cfg['target_train'],
                                     transform=transform_v))
target_val = TargetDataset(datasets.ImageFolder(root=cfg['target_val'],
                                     transform=transform_v))
target_val_acc = SourceDataset(datasets.ImageFolder(root=cfg['target_eval'],
                                     transform=transform_v))
target_tsne = SourceDataset(datasets.ImageFolder(root=cfg['target_val'],
                                     transform=transform_v))
train = CombinedSourceAndTargetDataset(src_train, target_train)


dc = DataloaderCreator(batch_size=32, num_workers=cfg['num_workers'])
dataloaders = dc(train = train,
                 src_train = src_train,
                 src_val = src_val,
                 target_train = target_train,
                 target_val = target_val)
eval_loader = dc(src_val = target_val_acc)


device = torch.device("cuda")

model = CustomResNet18(cfg['num_classes']) 
G = model.features_conv.to(device)
G.add_module('36', model.max_pool)
G.add_module('37', nn.Flatten())
C = model.classifier.to(device)
models = Models({"G": G, "C": C})

optimizers = Optimizers((torch.optim.Adam, {"lr": 0.0001}))
optimizers.create_with(models)
optimizers = list(optimizers.values())


from pytorch_adapt.hooks import AlignerPlusCHook
from pytorch_adapt.layers import MMDLoss
from pytorch_adapt.layers.utils import get_kernel_scales

kernel_scales = get_kernel_scales(low=-8, high=8, num_kernels=33)
loss_fn = MMDLoss(kernel_scales=kernel_scales)
hook = AlignerPlusCHook(opts=optimizers[:2], loss_fn=loss_fn)


src_t_loss = []
src_v_loss = []
src_v_acc = []
target_loss = []
target_acc = []

for epoch in range(100):

    # train loop
    criterion = nn.CrossEntropyLoss()   # we still need a criterion to calculate the validation loss

    # running averages
    loss_total, oa_total = 0.0, 0.0     # for now, we just log the loss and overall accuracy (OA)

    # iterate over dataLoader
    progressBar = trange(len(dataloaders["train"]), position=0, leave=True)
    tot_loss = 0
    models.train()
    for idx, data in enumerate(dataloaders["train"]):
        data = batch_to_device(data, device)
        _, loss = hook({**models, **data})
        tot_loss += loss['total_loss']['src_c_loss']
        progressBar.set_description(f"Epoch : {epoch}")
        progressBar.update(1)
    progressBar.close()
    
    avg_loss = tot_loss/(idx+1)
    src_t_loss.append(avg_loss)
    

    # eval loop
    models.eval()
    criterion = nn.CrossEntropyLoss()   # we still need a criterion to calculate the validation loss

    # running averages
    loss_total, oa_total = 0.0, 0.0     # for now, we just log the loss and overall accuracy (OA)

    # iterate over dataLoader
    progressBar = trange(len(dataloaders["src_val"]), position=0, leave=True)
    with torch.no_grad():
        for idx, data in enumerate(dataloaders["src_val"]):
            data = batch_to_device(data, device)
            # forward pass
            prediction = C(G(data['src_imgs']))
            labels = data['src_labels']

            # loss
            loss = criterion(prediction, labels)

            # log statistics
            loss_total += loss.item()

            pred_label = torch.argmax(prediction, dim=1)
            oa = torch.mean((pred_label == labels).float())
            oa_total += oa.item()
            
            avg_loss = loss_total/(idx+1)
            avg_oa = 100*oa_total/(idx+1)

            progressBar.set_description(
                '[Val ] Loss: {:.2f}; OA: {:.2f}%'.format(
                    avg_loss,
                    avg_oa
                )
            )
            progressBar.update(1)
    progressBar.close()
    
    src_v_loss.append(avg_loss)
    src_v_acc.append(avg_oa)

    models.eval()
    criterion = nn.CrossEntropyLoss()   # we still need a criterion to calculate the validation loss

    # running averages
    loss_total, oa_total = 0.0, 0.0     # for now, we just log the loss and overall accuracy (OA)

    # iterate over dataLoader
    progressBar = trange(len(eval_loader["src_val"]), position=0, leave=True)
    with torch.no_grad():
        for idx, data in enumerate(eval_loader["src_val"]):
            data = batch_to_device(data, device)
            # forward pass
            prediction = C(G(data['src_imgs']))
            labels = data['src_labels']

            # loss
            loss = criterion(prediction, labels)

            # log statistics
            loss_total += loss.item()
            
            pred_label = torch.argmax(prediction, dim=1)
            oa = torch.mean((pred_label == labels).float())
            oa_total += oa.item()
            
            avg_loss = loss_total/(idx+1)
            avg_oa = 100*oa_total/(idx+1)

            progressBar.set_description(
                '[Val ] Loss: {:.2f}; OA: {:.2f}%'.format(
                    avg_loss,
                    avg_oa
                )
            )
            progressBar.update(1)
    progressBar.close()
    target_loss.append(avg_loss)
    target_acc.append(avg_oa)
    
    for _ in range(3):
        try:
            torch.save(models, os.path.join(cfg['save_path'], f"{exp}_{epoch}.pth"))
            with open('src_v_loss.json', 'w') as f:
                json.dump(src_v_loss, f)

            with open('src_t_loss.json', 'w') as f:
                json.dump(src_t_loss, f)
                
            with open('src_v_acc.json', 'w') as f:
                json.dump(src_v_acc, f)

            with open('target_acc.json', 'w') as f:
                json.dump(target_acc, f)

            with open('target_loss.json', 'w') as f:
                json.dump(target_loss, f)
        except RuntimeError as e:
            print(f"Error saving model: {e}. Retrying...")
            time.sleep(1)  # Wait for 1 second before retrying
    



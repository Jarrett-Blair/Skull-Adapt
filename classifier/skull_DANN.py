# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 22:41:21 2023

@author: blair
"""

import os
import argparse
import yaml
import glob
from tqdm import trange

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torchvision import datasets, transforms

# let's import our own classes and functions!
os.chdir(r"C:\Users\blair\OneDrive - UBC\CV-eDNA-Hybrid\ct_classifier")
from util import init_seed
from dataset import CTDataset
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
parser.add_argument('--config', help='Path to config file', default='../configs/skull_DANN.yaml')
args = parser.parse_args()

# load config
print(f'Using config "{args.config}"')
cfg = yaml.safe_load(open(args.config, 'r'))

# init random number generator seed (set at the start)
init_seed(cfg.get('seed', None))

# check if GPU is available
device = cfg['device']
if device != 'cpu' and not torch.cuda.is_available():
    print(f'WARNING: device set to "{device}" but CUDA not available; falling back to CPU...')
    cfg['device'] = 'cpu'

# initialize data loaders for training and validation set

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust the size as needed
    transforms.ToTensor()
])

# Create the ImageFolder dataset for training data
src_train = SourceDataset(datasets.ImageFolder(root=cfg['src_train'],
                                     transform=transform,
                                     shuffle = True))
src_val = SourceDataset(datasets.ImageFolder(root=cfg['src_val'],
                                     transform=transform))
target_train = TargetDataset(datasets.ImageFolder(root=cfg['target_train'],
                                     transform=transform,
                                     shuffle = True))
target_val = TargetDataset(datasets.ImageFolder(root=cfg['target_val'],
                                     transform=transform))
target_val_acc = SourceDataset(datasets.ImageFolder(root=cfg['target_val'],
                                     transform=transform))
train = CombinedSourceAndTargetDataset(src_train, target_train)


dc = DataloaderCreator(batch_size=32, num_workers=cfg['num_workers'])
dataloaders = dc(train = train,
                 src_train = src_train,
                 src_val = src_val,
                 target_train = target_train,
                 target_val = target_val)
test_loader = dc(train = train,
                 src_train = src_train,
                 src_val = target_val_acc,
                 target_train = target_train,
                 target_val = target_val)


device = torch.device("cuda")

model = CustomResNet18(cfg['num_classes']) 
G = model.features_conv.to(device)
G.add_module('36', model.max_pool)
G.add_module('37', nn.Flatten())
C = model.classifier.to(device)
D = Discriminator(in_size=25088, h=4096).to(device)
models = Models({"G": G, "C": C, "D": D})

optimizers = Optimizers((torch.optim.Adam, {"lr": 0.0001}))
optimizers.create_with(models)
optimizers = list(optimizers.values())

hook = ClassifierHook(optimizers)
validator = IMValidator()


for epoch in range(50):

    # train loop
    criterion = nn.CrossEntropyLoss()   # we still need a criterion to calculate the validation loss

    # running averages
    loss_total, oa_total = 0.0, 0.0     # for now, we just log the loss and overall accuracy (OA)

    # iterate over dataLoader
    progressBar = trange(len(dataloaders["train"]), position=0, leave=True)
    models.train()
    for idx, data in enumerate(dataloaders["train"]):
        data = batch_to_device(data, device)
        _, loss = hook({**models, **data})
        progressBar.set_description(f"Epoch : {epoch}")
        progressBar.update(1)
    progressBar.close()

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

            progressBar.set_description(
                '[Val ] Loss: {:.2f}; OA: {:.2f}%'.format(
                    loss_total/(idx+1),
                    100*oa_total/(idx+1)
                )
            )
            progressBar.update(1)
    progressBar.close()

    models.eval()
    criterion = nn.CrossEntropyLoss()   # we still need a criterion to calculate the validation loss

    # running averages
    loss_total, oa_total = 0.0, 0.0     # for now, we just log the loss and overall accuracy (OA)

    # iterate over dataLoader
    progressBar = trange(len(test_loader["src_val"]), position=0, leave=True)
    with torch.no_grad():
        for idx, data in enumerate(test_loader["src_val"]):
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

            progressBar.set_description(
                '[Val ] Loss: {:.2f}; OA: {:.2f}%'.format(
                    loss_total/(idx+1),
                    100*oa_total/(idx+1)
                )
            )
            progressBar.update(1)
    progressBar.close()
    



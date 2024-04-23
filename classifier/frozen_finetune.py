# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 17:11:03 2024

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
from pytorch_adapt.adapters import Finetuner
from pytorch_adapt.hooks import FinetunerHook, ClassifierHook, BNMHook, BSPHook
from pytorch_adapt.models import Discriminator, mnistC, mnistG
from pytorch_adapt.utils.common_functions import batch_to_device
from pytorch_adapt.validators import IMValidator

parser = argparse.ArgumentParser(description='Train deep learning model.')
parser.add_argument('--config', help='Path to config file', default='../configs/subset.yaml')
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
target_val_acc = SourceDataset(datasets.ImageFolder(root=cfg['target_val'],
                                     transform=transform_v))


dc = DataloaderCreator(batch_size=32, num_workers=cfg['num_workers'])
dataloaders = dc(train = src_train,
                 src_val = src_val)
eval_loader = dc(src_val = target_val_acc)

device = torch.device("cuda")

model_path = os.path.join(cfg['save_path'], "mmd/skull_MMD_49.pth")
model_weights = torch.load(model_path)
G = model_weights['G'].to(device)
C = model_weights['C'].to(device)

models = Models({"G": G, "C": C})

adapter = Finetuner(models)


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
        loss = adapter.training_step(data)
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


  
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 14:58:35 2024

@author: blair
"""

import os
import argparse
import yaml
import glob
from tqdm import trange
import numpy as np
import random
import cv2
import itertools

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torchvision import datasets, transforms

# let's import our own classes and functions!
os.chdir(r"C:\Users\blair\OneDrive - UBC\Skull-Adapt\classifier")
from util import init_seed

import torch.nn as nn
from torchvision.models import vgg19

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

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


def save_cams(loader, idxs, preds, species_list, model, target_layer, filepath):
    progressBar = trange(len(loader['src_val']), position=0, leave=True)
    for batch_idx, data in enumerate(loader['src_val']):
        if batch_idx in idxs:
           
            data = batch_to_device(data, device)
            image = data['src_imgs']
    
            img = image.cpu()
            pred_lab = preds[batch_idx]
            labels = [ClassifierOutputTarget(pred_lab)]
            species = species_list[pred_lab]
            
            gradcam = GradCAM(model, target_layers, use_cuda=True)
            
            
            mask = gradcam(image, labels, aug_smooth=True, eigen_smooth=True)
            mask = mask.squeeze()
            img = img.squeeze()
            img = img.permute(1,2,0)
            
            visualisation = show_cam_on_image(np.array(img), mask, use_rgb=True)
            bgr_vis = cv2.cvtColor(visualisation, cv2.COLOR_RGB2BGR)
            
            filename = os.path.basename(src_val_torch.samples[batch_idx][0])
            cv2.imwrite(filepath+f'{species}_'+filename, bgr_vis)
        progressBar.update(1)
    progressBar.close()



parser = argparse.ArgumentParser(description='Train deep learning model.')
parser.add_argument('--config', help='Path to config file', default='../configs/supplemented.yaml')
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

src_val_torch = datasets.ImageFolder(root=cfg['src_val'], 
                                     transform=transform_v)
target_val_torch = datasets.ImageFolder(root=cfg['target_val'], 
                                        transform=transform_v)

# Create the ImageFolder dataset for training data
src_train = SourceDataset(datasets.ImageFolder(root=cfg['target_val'],
                                     transform=transform_t))
src_val = SourceDataset(src_val_torch)
target_val_acc = SourceDataset(target_val_torch)


dc = DataloaderCreator(batch_size=1, num_workers=cfg['num_workers'])
dataloaders = dc(src_val = src_val)
eval_loader = dc(src_val = target_val_acc)

model_path = os.path.join(cfg['save_path'], "supplemented/supplemented_23.pth")
model_weights = torch.load(model_path)
G = model_weights['G'].to(device)
C = model_weights['C'].to(device)
models = Models({"G": G, "C": C})

combined_model = nn.Sequential(
    *G,
    *C
)

combined_model.eval()

# for now, we just log the loss and overall accuracy (OA)

# iterate over dataLoader

preds = []
probs = []
true_labs = []
features = []

# Using dataloaders
progressBar = trange(len(eval_loader['src_val']), position=0, leave=True)
with torch.no_grad():
    for idx, data in enumerate(eval_loader['src_val']):
        data = batch_to_device(data, device)
        
        # forward pass
        probs.append(combined_model(data['src_imgs']))
        true_labs.append(data['src_labels'])

        preds.append(torch.argmax(probs[idx], dim=1))

        progressBar.update(1)
progressBar.close()


preds = torch.cat(preds, dim=0)
probs = torch.cat(probs, dim=0)
true_labs = torch.cat(true_labs, dim=0)

preds = preds.cpu()
probs = probs.cpu()
true_labs = true_labs.cpu()

preds = preds.numpy()
probs = probs.numpy()
true_labs = true_labs.numpy()


init_seed(cfg.get('seed', None))
species_list = [
    "Canis-latrans",
    "Canis-lupus",
    "Gulo-gulo",
    "Lontra-canadensis",
    "Lynx-canadensis",
    "Lynx-rufus",
    "Martes-americana",
    "Martes-pennanti",
    "Mephitis-mephitis",
    "Neovison-vison",
    "Procyon-lotor",
    "Ursus-americanus",
    "Ursus-arctos",
    "Ursus-maritimus",
    "Vulpes-lagopus",
    "Vulpes-vulpes"
]


target_layers = [combined_model[-9]]
filepath = r'C:\Users\blair\OneDrive - UBC\Skulls\CAMs-Supplemented/'


dataset = eval_loader['src_val']
random_numbers = [random.randint(0, len(dataset)) for _ in range(100)]

save_cams(eval_loader, random_numbers, preds, species_list, combined_model, target_layers, filepath)

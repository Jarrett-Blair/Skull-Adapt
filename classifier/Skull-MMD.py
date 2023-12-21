# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 16:40:06 2023

@author: blair
"""

import os
import argparse
import yaml
import glob
from tqdm import trange
import numpy as np
import pandas as pd

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

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

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
tsne_loader = dc(train = train,
                 src_train = src_train,
                 src_val = target_tsne,
                 target_train = target_train,
                 target_val = target_val)
eval_loader = dc(train = train,
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


from pytorch_adapt.hooks import AlignerPlusCHook
from pytorch_adapt.layers import MMDLoss
from pytorch_adapt.layers.utils import get_kernel_scales

kernel_scales = get_kernel_scales(low=-8, high=8, num_kernels=33)
loss_fn = MMDLoss(kernel_scales=kernel_scales)
hook = AlignerPlusCHook(opts=optimizers[:2], loss_fn=loss_fn)


for epoch in range(100):

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
    


models.eval()
 
# for now, we just log the loss and overall accuracy (OA)

# iterate over dataLoader

preds = []
probs = []
true_labs = []
features = []

progressBar = trange(len(tsne_loader['src_val']), position=0, leave=True)
with torch.no_grad():
    for idx, data in enumerate(tsne_loader['src_val']):
        data = batch_to_device(data, device)
        # forward pass
        features.append(G(data['src_imgs']))
        probs.append(C(features[idx]))
        true_labs.append(data['src_labels'])

        preds.append(torch.argmax(probs[idx], dim=1))

        progressBar.update(1)
progressBar.close()

progressBar = trange(len(dataloaders['src_val']), position=0, leave=True)
with torch.no_grad():
    for idx, data in enumerate(dataloaders['src_val']):
        data = batch_to_device(data, device)
        # forward pass
        features.append(G(data['src_imgs']))
        probs.append(C(features[idx]))
        true_labs.append(data['src_labels'])

        preds.append(torch.argmax(probs[idx], dim=1))

        progressBar.update(1)
progressBar.close()


preds = torch.cat(preds, dim=0)
probs = torch.cat(probs, dim=0)
true_labs = torch.cat(true_labs, dim=0)
features = torch.cat(features, dim=0)

preds = preds.cpu()
probs = probs.cpu()
true_labs = true_labs.cpu()
features = features.cpu()

preds = preds.numpy()
probs = probs.numpy()
true_labs = true_labs.numpy()
features = features.numpy()

target_array = np.full(len(target_tsne), 'Target')
source_array = np.full(len(src_val), 'Source')

# Concatenate the arrays to create the final array
domain = np.concatenate((target_array, source_array))

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


tsne = TSNE(n_components=2, random_state=42)
tsne_embeddings = tsne.fit_transform(features)

# markers = {'Source': '.', 'Target': '1'}

species = os.listdir(cfg['src_train'])
sp_labs = np.array(species)[true_labs]

tsne_df = pd.DataFrame({'tsne1': tsne_embeddings[:,0],
                        'tsne2': tsne_embeddings[:,1],
                        'labels': sp_labs,
                        'domain': domain})

c25 = ["#1C86EE", "#E31A1C", "#008B00", "#6A3D9A", "#FF7F00", "#000000", "#FFD700", 
       "#7EC0EE", "#FB9A99", "#90EE90", "#CAB2D6", "#FDBF6F", "#B3B3B3", "#EEE685", 
       "#B03060", "#FF83FA", "#FF1493", "#0000FF", "#36648B", "#00CED1", "#00FF00", 
       "#8B8B00", "#CDCD00", "#8B4500", "#A52A2A"]

def tsne_plot(data, palette = 'viridis', domains = False, title = None,
              xlim = None, ylim = None):
    
    style = None
    if domains:
        style = 'domain'
    
    xmin = min(data['tsne1']) - 5
    xmax = max(data['tsne1']) + 5
    ymin = min(data['tsne2']) - 5
    ymax = max(data['tsne2']) + 5
    
    if xlim != None:
        xmin = xlim[0]
        xmax = xlim[1]
    
    if ylim != None:
        ymin = ylim[0]
        ymax = ylim[1]
    
    sns.scatterplot(x='tsne1', 
                    y='tsne2', 
                    hue='labels', 
                    palette=palette,
                    style = style,
                    data=data)
    
    plt.xlim(xmin, xmax)  # Set the x-axis bounds
    plt.ylim(ymin, ymax)  # Set the y-axis bounds
    plt.legend(bbox_to_anchor=(1.05, 1), ncol=1, loc='upper left')
    plt.title(title)
    plt.show()    

xlim = (min(tsne_df['tsne1']) - 5, max(tsne_df['tsne1']) + 5)
ylim = (min(tsne_df['tsne2']) - 5, max(tsne_df['tsne2']) + 5)


tsne_plot(data = tsne_df, 
          palette = c25, 
          domains = True,
          title = "Target and Source",
          xlim = xlim,
          ylim = ylim)

just_target = tsne_df[tsne_df['domain'] == 'Target']
just_source = tsne_df[tsne_df['domain'] == 'Source']

tsne_plot(data = just_target, 
          palette = c25,
          title = "Target",
          xlim = xlim,
          ylim = ylim)

tsne_plot(data = just_source, 
          palette = c25,
          title = "Source",
          xlim = xlim,
          ylim = ylim)



# Visualize the t-SNE embeddings
# plt.scatter(tsne_embeddings[:, 0], 
#             tsne_embeddings[:, 1], 
#             c=domain, 
#             cmap='viridis',
#             marker=domain.map(markers))
# plt.title('t-SNE')
# plt.colorbar()
# plt.show()

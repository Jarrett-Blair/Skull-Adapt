# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 18:06:29 2024

@author: blair
"""
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

import os
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

transform_v = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


src_val = SourceDataset(datasets.ImageFolder(root=cfg['src_val'],
                                     transform=transform_v))
target_tsne = SourceDataset(datasets.ImageFolder(root=cfg['target_train'],
                                     transform=transform_v))


dc = DataloaderCreator(batch_size=32, num_workers=cfg['num_workers'])
dataloaders = dc(src_val = src_val)
tsne_loader = dc(src_val = target_tsne)

device = torch.device("cuda")

model_path = os.path.join(cfg['save_path'], "mmd_fine_tune/skull_MMD_30.pth")
model_weights = torch.load(model_path)
G = model_weights['G'].to(device)
C = model_weights['C'].to(device)

models = Models({"G": G, "C": C})


def conf_table(conf_matrix, Y, prop = True):
    """
    Creates a confusion matrix as a pandas data frame pivot table

    Parameters:
    - conf_matrix (Array): The standard confusion matrix output from sklearn
    - Y (list): The unique labels of the classifier (i.e. the classes of the output layer). Will be used as conf_table labels
    - prop (bool): Should the conf table use proportions (i.e. Recall) or total values?

    Returns:
    DataFrame: conf_table
    """
    
    # Convert conf_matrix to list
    conf_data = []
    for i, row in enumerate(conf_matrix):
        for j, count in enumerate(row):
            conf_data.append([Y[i], Y[j], count])
    
    # Convert list to 
    conf_df = pd.DataFrame(conf_data, columns=['Reference', 'Prediction', 'Count'])
    
    # If prop = True, calculate proportions
    if prop:
        conf_df['prop'] = conf_df['Count'] / (conf_df.groupby('Reference')['Count'].transform('sum') + 0.1)
        
        # Create conf_table
        conf_table = conf_df.pivot_table(index='Reference', columns='Prediction', values='prop', fill_value=0)
        
    else:
        # Create conf_table
        conf_table = conf_df.pivot_table(index='Reference', columns='Prediction', values='Count', fill_value=0)
    
    
    return conf_table



def plt_conf(table, Y, report, domain):
    """
    Plots a confusion matrix

    Parameters:
    - table (DataFrame): The conf_table
    - Y (list): The class labels to be plotted on the conf_tab
    - report (dict): From sklearn.metrics.classification_report

    Returns:
    Saves plot to directory
    """
      
    accuracy = round(report["accuracy"], 3)
    
    custom_gradient = ["#201547", "#00BCE1"]
    n_bins = 100  # Number of bins for the gradient

    custom_cmap = LinearSegmentedColormap.from_list("CustomColormap", custom_gradient, N=n_bins)

    plt.figure(figsize=(16, 12))
    sns.set(font_scale=1)
    sns.set_style("white")
    
    ax = sns.heatmap(table, cmap=custom_cmap, cbar_kws={'label': 'Proportion'})
    
    ax.set_title(f"Accuracy = {accuracy} ; Domain = {domain}", fontsize=24)
    # Customize the axis labels and ticks
    ax.set_xlabel("Predicted", fontsize=20)
    ax.set_ylabel("Actual", fontsize=20)
    ax.set_xticks(np.arange(len(Y)) + 0.5)
    ax.set_yticks(np.arange(len(Y)) + 0.5)
    ax.set_xticklabels(Y, fontsize=12)
    ax.set_yticklabels(Y, rotation=0, fontsize=12)
    
    # Add annotation
    ax.annotate("Predicted", xy=(0.5, -0.2), xytext=(0.5, -0.5), ha='center', va='center',
                 textcoords='axes fraction', arrowprops=dict(arrowstyle="-", lw=1))
    
    
    # Customize the appearance directly on the Axes object
    ax.set_xticklabels(Y, rotation=45, ha='right')
    
    return plt.gcf()


preds = []
probs = []
true_labs = []
features = []

# Run this first to get target data features
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

# Run this next to append source data features. This way both will be used in the tsne
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

species = os.listdir(cfg['src_train'])
sp_labs = np.array(species)[true_labs]
pred_labs = np.array(species)[preds]


conf_matrix = confusion_matrix(sp_labs,
                      pred_labs,
                      labels = species)

conf_tab = conf_table(conf_matrix, species)

report = classification_report(sp_labs,
                      pred_labs,
                      output_dict=True,
                      zero_division = 1)

plt_conf(conf_tab, species, report, domain="Target")
plt_conf(conf_tab, species, report, domain="Source")


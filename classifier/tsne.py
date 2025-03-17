import os
import argparse
import yaml
from tqdm import trange
import numpy as np
import pandas as pd

import torch
from torchvision import datasets, transforms

# let's import our own classes and functions!
os.chdir(r"C:\Users\blair\OneDrive - UBC\Skull-Adapt\classifier")
from util import init_seed

from pytorch_adapt.containers import Models
from pytorch_adapt.datasets import (
    DataloaderCreator,
    SourceDataset,
)
from pytorch_adapt.utils.common_functions import batch_to_device

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
target_tsne = SourceDataset(datasets.ImageFolder(root=cfg['target_val'],
                                     transform=transform_v))


dc = DataloaderCreator(batch_size=32, num_workers=cfg['num_workers'])
dataloaders = dc(src_val = src_val)
tsne_loader = dc(src_val = target_tsne)

device = torch.device("cuda")

model_path = os.path.join(cfg['save_path'], "subset/subset_24.pth")
model_weights = torch.load(model_path)
G = model_weights['G'].to(device)
C = model_weights['C'].to(device)

models = Models({"G": G, "C": C})


models.eval()

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

species_ordered = ['Lynx canadensis',
                   'Lynx rufus',
                   'Canis latrans',
                   'Canis lupus',
                   'Vulpes lagopus',
                   'Vulpes vulpes',
                   'Gulo gulo',
                   'Lontra canadensis',
                   'Martes americana',
                   'Mephitis mephitis',
                   'Neogale vison',
                   'Pekania pennanti',
                   'Procyon lotor',
                   'Ursus americanus',
                   'Ursus arctos',
                   'Ursus maritimus',
]
sp_ord_idx = [2, 3, 6, 7, 0, 1, 8, 11, 9, 10, 12, 13, 14, 15, 4, 5]

true_ord = []
for i in range(0, len(true_labs)):
    true_ord.append(sp_ord_idx[true_labs[i]])
    
sp_labs = np.array(species_ordered)[true_ord]

tsne_df = pd.DataFrame({'tsne1': tsne_embeddings[:,0],
                        'tsne2': tsne_embeddings[:,1],
                        'labels': sp_labs,
                        'domain': domain})

c25 = ["#1C86EE", "#E31A1C", "#008B00", "#6A3D9A", "#FF7F00", "#000000", "#FFD700", 
       "#7EC0EE", "#FB9A99", "#90EE90", "#CAB2D6", "#FDBF6F", "#B3B3B3", "#EEE685", 
       "#B03060", "#FF83FA", "#FF1493", "#0000FF", "#36648B", "#00CED1", "#00FF00", 
       "#8B8B00", "#CDCD00", "#8B4500", "#A52A2A"]

def tsne_plot(data, hue_order = None, palette = 'viridis', domains = False, 
              title = None, xlim = None, ylim = None, exp = None): 
    
    style = None
    if domains:
        style = 'domain'
    
    markers = True
    edgecolor = 'white'
    if title == "Source":
        markers = ['X']
        style = 'domain'
        edgecolor = 'black'
    
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
        
    plt.figure(dpi=300) 
    
    ax = sns.scatterplot(x='tsne1', 
                    y='tsne2', 
                    hue=title, 
                    hue_order= hue_order,
                    palette=palette,
                    style = style,
                    markers=markers,
                    s = 25,
                    data=data,
                    edgecolor = edgecolor)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(exp)
    handles, labels = ax.get_legend_handles_labels()
    print(handles)
    print(labels)
    ax.legend(title='Domains', bbox_to_anchor=(1.05, 1), ncol=4, loc='upper left', handles=handles[1:3], labels=labels[1:3], fontsize = '10', markerscale = 1.5)

    plt.xlim(xmin, xmax)  # Set the x-axis bounds
    plt.ylim(ymin, ymax)  # Set the y-axis bounds
    plt.xticks([])
    plt.yticks([])    
    plt.tight_layout()
    #plt.savefig(f'..\plots\{exp}_{title}.png', dpi=300)
    plt.show()
    

xlim = (min(tsne_df['tsne1']) - 5, max(tsne_df['tsne1']) + 5)
ylim = (min(tsne_df['tsne2']) - 5, max(tsne_df['tsne2']) + 5)

exp = 'Photograph subset'

tsne_plot(data = tsne_df, 
          hue_order = ['Source', 'Target'],
          palette = c25, 
          domains = True,
          title = "domain",
          exp = exp,
          xlim = xlim,
          ylim = ylim)

tsne_plot(data = tsne_df, 
          hue_order = species_ordered,
          palette = c25, 
          domains = True,
          title = "labels",
          exp = exp,
          xlim = xlim,
          ylim = ylim)

just_target = tsne_df[tsne_df['domain'] == 'Target']
just_source = tsne_df[tsne_df['domain'] == 'Source']

target = tsne_plot(data = just_target, 
          species = species_ordered,
          palette = c25,
          title = "Target",
          xlim = xlim,
          ylim = ylim,
          exp = exp)

source = tsne_plot(data = just_source, 
          species = species_ordered,
          palette = c25,
          title = "Source",
          xlim = xlim,
          ylim = ylim,
          exp = exp)





from sklearn.metrics import silhouette_score

print(silhouette_score(just_source[["tsne1", "tsne2"]], just_source["labels"]))
print(silhouette_score(just_target[["tsne1", "tsne2"]], just_target["labels"]))


def tsne_plot_overlay(data, data2, species = None, palette = 'viridis', domains = False, 
              title = None, xlim = None, ylim = None, exp = None): 
    
    style = None
    if domains:
        style = 'domain'
    
    markers = True
    if title == "Source":
        markers = ['X']
    
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
        
    hue_order = species
    plt.figure(dpi=300) 
    
    ax = sns.scatterplot(x='tsne1', 
                    y='tsne2', 
                    hue='labels', 
                    hue_order= hue_order,
                    palette=palette,
                    style = style,
                    s = 25,
                    data=data,
                    edgecolor = 'black',
                    legend=False)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(exp)
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(bbox_to_anchor=(1.05, 1), ncol=1, loc='upper left', handles=handles[1:16], labels=labels[1:16], fontsize = '10', markerscale = 1.5)
    bx = sns.scatterplot(x='tsne1', 
                    y='tsne2', 
                    hue='labels', 
                    hue_order= hue_order,
                    palette=palette,
                    style = style,
                    markers=markers,
                    s = 25,
                    data=data2,
                    edgecolor = 'white',
                    legend=False)
    bx.set_xlabel('')
    bx.set_ylabel('')
    bx.set_title(exp)
    
    plt.xlim(xmin, xmax)  # Set the x-axis bounds
    plt.ylim(ymin, ymax)  # Set the y-axis bounds
    plt.xticks([])
    plt.yticks([])    
    plt.tight_layout()
    #plt.savefig(f'..\plots\{exp}_{title}.png', dpi=300)
    plt.show()
    

tsne_plot_overlay(data = just_target, 
                  data2= just_source,
          species = species_ordered,
          palette = c25,
          title = "Source",
          xlim = xlim,
          ylim = ylim,
          exp = exp)





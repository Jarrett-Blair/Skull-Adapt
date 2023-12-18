# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 00:03:28 2023

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
os.chdir(r"C:\Users\blair\OneDrive - UBC\CV-eDNA-Hybrid\ct_classifier")
from util import init_seed
from eval_metrics import predict

import torch.nn as nn
from torchvision.models import vgg19

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image



class CustomResNet18(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet18, self).__init__()
        
        # get the pretrained VGG19 network
        self.vgg = vgg19(pretrained=True)
        
        # disect the network to access its last convolutional layer
        self.features_conv = self.vgg.features[:36]
        
        # get the max pool of the features stem
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        in_features = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Linear(in_features, num_classes)
        
        self.classifier = self.vgg.classifier
        
        # Capture activations from the final convolutional layer
        self.activations = None
        
        self.features_conv[-1].register_forward_hook(self.activations_hook)
        # placeholder for the gradients
        self.gradients = None
    
    def activations_hook(self, module, input, output):
        self.activations = output

    def forward(self, x):
        # Forward pass to the final convolutional layer
        x = self.features_conv(x)
        
        # Apply the remaining pooling and classifier
        x = self.max_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        if self.gradients is not None:
            return self.gradients

        raise RuntimeError("No gradients available. Run backward on the model's output to compute gradients.")

    # method for the activation extraction
    def get_activations(self, x):
        return self.features_conv(x)

    # method to compute gradients (use this during Grad-CAM calculation)
    def compute_gradients(self):
        if self.activations is not None:
            self.gradients = torch.autograd.grad(outputs=self.activations, inputs=self.features_conv.parameters())

    
    


def load_model(cfg):
    '''
        Creates a model instance and loads the latest model state weights.
    '''
    model_instance = CustomResNet18(cfg['num_classes'])         # create an object instance of our CustomResNet18 class

    # load latest model state
    model_states = glob.glob('model_states/*.pt')
    model_states = [os.path.normpath(path) for path in model_states]
    
    if len(model_states):
        # at least one save state found; get latest
        model_epochs = [int(os.path.basename(m).replace('.pt','')) for m in model_states]
        start_epoch = max(model_epochs)

        # load state dict and apply weights to model
        print(f'Resuming from epoch {start_epoch}')
        state = torch.load(open(f'model_states/{start_epoch}.pt', 'rb'), map_location='cpu')
        model_instance.load_state_dict(state['model'])

    else:
        # no save state found; start anew
        print('Starting new model')
        start_epoch = 0

    return model_instance, start_epoch


def save_cams(loader, idxs, preds, species_list, model, target_layer, filepath):
    for batch_idx, data in enumerate(itertools.islice(loader, max(idxs))): 
        if batch_idx in idxs:
            # get the image from the dataloader
            img, lab = data
            
            image = img.to(device)
            pred_lab = preds[batch_idx]
            labels = [ClassifierOutputTarget(pred_lab)]
            species = species_list[pred_lab]
            
            target_layer = [model.features_conv[-1]]
            gradcam = GradCAM(model, target_layer, use_cuda=True)
            
            
            mask = gradcam(image, labels)
            mask = mask.squeeze()
            img = img.squeeze()
            img = img.permute(1,2,0)
            
            visualisation = show_cam_on_image(np.array(img), mask, use_rgb=True)
            bgr_vis = cv2.cvtColor(visualisation, cv2.COLOR_RGB2BGR)
            
            filename = os.path.basename(valid_dataset.samples[batch_idx][0])
            cv2.imwrite(filepath+f'{species}_'+filename, bgr_vis)



parser = argparse.ArgumentParser(description='Train deep learning model.')
parser.add_argument('--config', help='Path to config file', default='../configs/skull.yaml')
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
test_dir = r'C:\Users\blair\OneDrive - UBC\Skulls\testing-named'
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust the size as needed
    transforms.ToTensor()
])

# Create the ImageFolder dataset for training data
valid_dataset = datasets.ImageFolder(root=test_dir,
                                     transform=transform)
# Create a data loader for training data
dl_test = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=cfg['num_workers'])
dl_val = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1)

model, current_epoch = load_model(cfg)
all_true, all_preds, all_probs = predict(cfg, dl_test, model)
model.to(device)

init_seed(cfg.get('seed', None))
random_numbers = [random.randint(0, 8650) for _ in range(100)]
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
target_layer = [model.features_conv[-1]]
filepath = r'C:\Users\blair\OneDrive - UBC\Skulls\CAMs-S2P/'

save_cams(dl_val, random_numbers, all_preds, species_list, model, target_layer, filepath)

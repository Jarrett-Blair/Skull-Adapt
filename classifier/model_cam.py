# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 23:07:38 2023

@author: blair
"""

'''
    Model implementation.
    We'll be using a "simple" ResNet-18 for image classification here.

    2022 Benjamin Kellenberger
'''

import torch
import torch.nn as nn
from torchvision.models import vgg19


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

    

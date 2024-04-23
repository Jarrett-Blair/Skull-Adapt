# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 13:09:54 2024

@author: blair
"""

import numpy as np
import os
import json


models = os.listdir(r"C:\Users\blair\OneDrive - UBC\Skull-Adapt\classifier\models")

stuff_keys = ['src_t_loss_curve',
              'src_v_acc_curve',
              'src_v_loss_curve',
              'target_acc_curve',
              'target_loss_curve',
              'epoch',
              'src_acc',
              'target_acc']

stuff = dict.fromkeys(stuff_keys)
model_curves = dict.fromkeys(models, stuff)

for model in model_curves.keys():
    
    os.chdir(rf"C:\Users\blair\OneDrive - UBC\Skull-Adapt\classifier\models\{model}")

    with open('src_v_loss', 'r') as f:
        model_curves['model']['src_v_loss_curve'] = json.load(f)
    
    with open('src_t_loss', 'r') as f:
        model_curves['model']['src_t_loss_curve'] = json.load(f)
        
    with open('src_v_acc', 'r') as f:
        model_curves['model']['src_v_acc_curve'] = json.load(f)
    
    with open('target_acc', 'r') as f:
        model_curves['model']['target_acc_curve'] = json.load(f)
    
    with open('target_loss', 'r') as f:
        model_curves['model']['target_loss_curve'] = json.load(f)


idx = np.argmin(src_v_loss)
print(idx)
print(src_v_acc[idx])
print(target_acc[idx])


import matplotlib.pyplot as plt

plt.plot(src_t_loss, label = "Source Train")
plt.plot(src_v_loss, label = "Source Val")
plt.legend()
plt.title('Source Loss')
plt.show()

plt.plot(src_v_acc, label = "Source Val")
plt.plot(target_acc, label = "Target Val")
plt.legend()
plt.title('Accuracy')
plt.show()

plt.plot(target_loss[:35])
plt.legend()
plt.title('Target Loss')
plt.show()


# Manual patience
patience = 0
low_loss = 100
for i in range(100):
    if src_v_loss[i] < low_loss:
        low_loss = src_v_loss[i]
        low_epoch = i
        patience = 0
    else:
        patience += 1
        if patience >= 15:
            break
        
low_epoch


# with open('src_v_loss', 'r') as f:
#     fine_tune_src_loss = json.load(f)

# with open('src_t_loss', 'r') as f:
#     src_t_loss = json.load(f)
    
# with open('src_v_acc', 'r') as f:
#     fine_tune_src_acc = json.load(f)

# with open('target_acc', 'r') as f:
#     fine_tune_target_acc = json.load(f)

# with open('target_loss', 'r') as f:
#     fine_tune_target_loss = json.load(f)

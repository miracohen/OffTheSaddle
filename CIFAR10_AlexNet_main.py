# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 09:51:53 2024

@author: mirac
"""

from numba import cuda

import time
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import os 
import datetime
from CIFAR10_AlexNet import AlexNet, CustomTensorDataset, CIFAR10_ALEXNET

torch.manual_seed(int(torch.rand(1).item()*1000))

# OPTIMIZER_TYPE = 'ADAM'
# LR = 1e-4
# LR_REDUCE_FACTOR = 0.95

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
# BATCH_SIZE = 512↕
OPTIMIZER_TYPE = 'SGD'
LR = 1e-1
LR_REDUCE_FACTOR = 0.95

def main():
    
    CIFAR10_ALEXNET.init_model(3, 10)
    CIFAR10_ALEXNET.load_data()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(CIFAR10_ALEXNET.model.parameters(), lr=LR)#, weight_decay=WEIGHT_DECAY)
    if OPTIMIZER_TYPE == 'ADAM':
        optimizer = optim.AdamW(CIFAR10_ALEXNET.model.parameters(), lr=LR)#, weight_decay=WEIGHT_DECAY)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = LR_REDUCE_FACTOR, patience=10, verbose=True)
    
    CIFAR10_ALEXNET.set_criterion_optimizer_scheduler(OPTIMIZER_TYPE, criterion, optimizer, lr_scheduler)
    model, val_acc_history, epochs_data =  CIFAR10_ALEXNET.train_model(EPOCHS, phases = ['train'])#, 'val'])
    
    # epoch_train_stats = pd.DataFrame() 
    # for i in range(len(epochs_data)):
        
    #     grads = epochs_data[i]['train']['grads_norm_2'].item()
    #     loss = epochs_data[i]['train']['loss']
    #     acc = epochs_data[i]['train']['acc'].item()
    #     new_row = {"file_path" : CIFAR10_RESNET18.path_save_model,"epoch_num" : len(epochs_data), "BATCH_SIZE" : CIFAR10_RESNET18.batch_size, 
    #                "grads_norm_2" : grads, 
    #                "loss" : loss, "acc" : acc}
    #     epoch_train_stats = pd.concat([epoch_train_stats, pd.DataFrame([new_row])], ignore_index = True)
    # # model, _ = CIFAR10_RESNET18.train_model(1, phases = ['val'])
    # epoch_train_stats.to_excel(CIFAR10_RESNET18.path_save_model + "/epochs_main_stats.xlsx")
    

        
        
if __name__ == '__main__':
    main()


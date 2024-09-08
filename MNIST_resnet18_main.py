# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 13:24:56 2024

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
from MNIST_resnet18 import Block, ResNet_18, CustomTensorDataset, MNIST_RESNET18

torch.manual_seed(17)
LR = 0.01
WEIGHT_DECAY=1e-2
PATH_SAVE_MODEL = r"./runs/MNIST_RESNET18/SGD/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"LR="+str(LR)+"WEIGHT_DECAY="+str(WEIGHT_DECAY)



DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS = 100





def main():
    
    MNIST_RESNET18.init_model(1, 10)
    MNIST_RESNET18.load_data()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(MNIST_RESNET18.model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
    
    MNIST_RESNET18.set_criterion_optimizer_scheduler(criterion, optimizer, lr_scheduler)
    model, _ =  MNIST_RESNET18.train_model(EPOCHS, phases = ['train', 'val'])
9    # model, _ = MNIST_RESNET18.train_model(1, phases = ['val'])
    
    



  
    
        
        
if __name__ == '__main__':
    main()
    
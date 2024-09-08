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

torch.manual_seed(17)
LR = 0.01
WEIGHT_DECAY=1e-2
PATH_SAVE_MODEL = r"./runs/MNIST_RESNET18/SGD/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"LR="+str(LR)+"WEIGHT_DECAY="+str(WEIGHT_DECAY)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS = 200
BATCH_SIZE = 32
class Block(nn.Module):
    
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x

class ResNet_18(nn.Module):
    
    def __init__(self, image_channels, num_classes):
        
        super(ResNet_18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        #resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def __make_layer(self, in_channels, out_channels, stride):
        
        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)
            
        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride), 
            Block(out_channels, out_channels)
        )
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x 
    
    def identity_downsample(self, in_channels, out_channels):
        
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(out_channels)
        )
    
class CustomTensorDataset(Dataset):

    def __init__(self, data, labels=None, transform=None):      
        self.data = data
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):       
        x = self.data[index]
        
        if self.transform is not None:
            x = self.transform(x)
        if self.labels is not None:
            y = self.labels[index]
            return x, y
        else:
            return x

    def __len__(self):    
        return self.data.size(0)
    
    
class MNIST_RESNET18():
    
  
    model = []
    dataloaders = []
    criterion = []    
    optimizer = [] 
    lr_scheduler = [] 
    path_save_model = "" 
    batch_size = BATCH_SIZE 

    @classmethod
    def init_model(cls,image_channels, num_classes):
        cls.model = ResNet_18(image_channels, num_classes)
        cls.model.to(DEVICE)
        cls.path_save_model = r"./runs/MNIST_RESNET18/SGD/STD_STATS_BATCH_SZ=" + str(BATCH_SIZE) + "LOG_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")#+"_LR="+str(learning_rate)+"_WEIGHT_DECAY="+str(weight_decay)
        
        
    @classmethod
    def load_data(cls):
        # tensor_transform = transforms.ToTensor()
        tensor_transform =transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize((112,112)),
        transforms.ToTensor()])
        dataset = datasets.MNIST(root = "./data", train = True, download = True, transform = None)
        X_train, X_val, X_test, y_train, y_val, y_test = cls.get_dataset_partitions_tf(torch.reshape(dataset.data,(-1,1,28,28))/255, dataset.targets, 0.8, 0.2, 0)
        trainset = ConcatDataset([
        CustomTensorDataset(X_train, y_train),
        CustomTensorDataset(X_train, y_train, transform=tensor_transform)
                    ])
        valset = CustomTensorDataset(X_val, y_val)
        data_loaders = {}
        data_loaders['train'] = DataLoader(trainset, batch_size=32, shuffle=True)
       
        data_loaders['val'] = DataLoader(valset, batch_size=32, shuffle=False)
        cls.dataloaders = data_loaders 
        # return train_loader, val_loader
    
    @classmethod
    def set_criterion_optimizer_scheduler(cls, criterion, optimizer, lr_scheduler):
        
        cls.criterion = criterion
        cls.optimizer = optimizer
        cls.lr_scheduler = lr_scheduler
    
    @classmethod
    def train_model(cls, num_epochs=1, phases = ['train', 'val']):
        
        
        val_acc_history = []
        best_model_wts = copy.deepcopy(cls.model.state_dict())
        best_acc = 0.0
    
        for epoch in range(num_epochs):
            since = time.time()
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            for phase in phases: # Each epoch has a training and validation phase
                if phase == 'train':
                    cls.model.train()  # Set model to training mode
                    training_grads = torch.zeros(cls.get_model_size()).to(DEVICE)
                    training_loss = 0.0
    
                else:
                    cls.model.eval()   # Set model to evaluate mode
    
                running_loss = 0.0
                running_corrects = 0
    
                
                for inputs, labels in cls.dataloaders[phase]: # Iterate over data
                    
                    inputs = transforms.functional.resize(inputs, (112, 112))
                    inputs = inputs.to(DEVICE)
    
                    labels = labels.to(DEVICE)
    
                    cls.optimizer.zero_grad() # Zero the parameter gradients
    
                    with torch.set_grad_enabled(phase == 'train'): # Forward. Track history if only in train
                        
                        outputs = cls.model(inputs)
                        loss = cls.criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)
    
                        if phase == 'train': # Backward + optimize only if in training phase
                            loss.backward()
                            training_grads += cls.get_model_grads_as_tensor() * inputs.size(0)
                            cls.optimizer.step()
    
                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    if phase == 'train':
                        training_loss = running_loss
    
                epoch_loss = running_loss / len(cls.dataloaders[phase].dataset)
                epoch_training_loss = training_loss / len(cls.dataloaders['train'].dataset)
                epoch_training_grad = training_grads / len(cls.dataloaders['train'].dataset)
                if phase == 'val': # Adjust learning rate based on val loss
                    cls.lr_scheduler.step(epoch_loss)
                    
                epoch_acc = running_corrects.double() / len(cls.dataloaders[phase].dataset)
    
                print('epoch:{} phase:{} Loss: {:.5f} Acc: {:.5f} Train Grad:{:.5f} Train grad inf norm: {:.5f}'.format(epoch, phase, epoch_loss, epoch_acc, torch.linalg.norm(epoch_training_grad), torch.linalg.norm(epoch_training_grad, ord = float('inf'))))
    
    
    
                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    cls.save_model(PATH_SAVE_MODEL, epoch, acc = epoch_acc, loss=epoch_training_loss, test_loss = epoch_loss, grad = torch.linalg.norm(epoch_training_grad), lr = cls.lr_scheduler.get_last_lr())
                    best_model_wts = copy.deepcopy(cls.model.state_dict())
                if phase == 'val':
                    val_acc_history.append(epoch_acc)
    
    
            time_elapsed = time.time() - since
            print('Epoch complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
    
        # load best model weights
        cls.model.load_state_dict(best_model_wts)
        return cls.model, val_acc_history
    
    @classmethod 
    def get_model_grads_as_tensor(cls):
        grads = []
        for param in cls.model.parameters():
            grads.append(param.grad.view(-1))
        grads = torch.cat(grads)
        grads= grads.clone() #detach
        return grads
    
    @classmethod 
    def get_model_size(cls):
        size = 0
        for param in cls.model.parameters():
            flat_param = torch.flatten(param)
            size += flat_param.shape[0]
        return size
    
    @classmethod 
    def count_parameters(cls):
        return sum(p.numel() for p in cls.model.parameters() if p.requires_grad)
    
    @classmethod 
    def get_dataset_partitions_tf(cls, X, y, train_split, val_split, test_split):
        assert (train_split + test_split + val_split) == 1
        assert len(X) == len(y)
        
        num_samples = X.shape[0]
        idx = torch.randperm(num_samples)
        train_samples_num = int(train_split*num_samples)
        val_samples_num = int(val_split*num_samples)
        
        X_train = X[idx[0:train_samples_num]]
        X_val = X[idx[train_samples_num:train_samples_num+val_samples_num]]
        X_test = X[idx[train_samples_num+val_samples_num:]]
        y_train =  y[idx[0:train_samples_num]]
        y_val = y[idx[train_samples_num:train_samples_num+val_samples_num]]
        y_test = y[idx[train_samples_num+val_samples_num:]]
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    @classmethod 
    def save_model(cls,path_save_model, epoch: int, acc = None, loss=None, test_loss = None, grad = None, lr = None):
        if not os.path.exists(path_save_model):
            os.makedirs(path_save_model)
        model_filename = '{}_{}__test_acc={:1.5f}_train_loss={:1.5f}_test_loss={:1.5f}_lr={:1.5f}_grad={:1.5f}.pth'.format(epoch+1, 'model', acc, loss, test_loss, lr[0], grad.item())
        
        checkpoint = {
            'epoch': epoch,
            'model': cls.model,
            'optimizer': cls.optimizer,
            'scheduler': cls.lr_scheduler,
            'criterion': cls.criterion,

            }
        torch.save(checkpoint,os.path.join(path_save_model,model_filename))
        # cls.model.to(DEVICE)

    @classmethod
    def load_model(cls, model_filepath):
        checkpoint = torch.load(model_filepath)
        cls.epoch = checkpoint['epoch']
        cls.model = checkpoint['model']
        cls.optimizer = checkpoint['optimizer']
        cls.lr_scheduler = checkpoint['scheduler']
        cls.criterion = checkpoint['criterion']
        
    @classmethod
    def set_model(cls, model):
        cls.model = model
    






def main():
    
    MNIST_RESNET18.init_model(1, 10)
    MNIST_RESNET18.load_data()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(MNIST_RESNET18.model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
    
    MNIST_RESNET18.set_criterion_optimizer_scheduler(criterion, optimizer, lr_scheduler)
    model, _ =  MNIST_RESNET18.train_model(EPOCHS, phases = ['train', 'val'])
    model, _ = MNIST_RESNET18.train_model(1, phases = ['val'])
    
    

    # model.eval()
    # labels = []
    # for inputs in val_loader:
    #     inputs = transforms.functional.resize(inputs, (112, 112))
    #     inputs = inputs.to(DEVICE)
    #     outputs = model(inputs)
    #     _, predictions = torch.max(outputs, 1)
    #     predictions = predictions.to("cpu")
    #     labels.extend(predictions.numpy())
        
        
        
if __name__ == '__main__':
    main()
    
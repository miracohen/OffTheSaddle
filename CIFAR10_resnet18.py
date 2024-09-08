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
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, ConcatDataset, TensorDataset

import os 
import datetime

import time


# torch.manual_seed(17)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256

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
    
class CIFAR10_RESNET18():
    
    model = []
    dataloaders = []
    criterion = []    
    optimizer_type = ""
    optimizer = [] 
    lr_scheduler = [] 
    path_save_model = "" 
    batch_size = BATCH_SIZE
    
    
    @classmethod
    def init_model(cls,image_channels, num_classes):
        cls.model = ResNet_18(image_channels, num_classes)
        cls.model.to(DEVICE)
        
    @classmethod
    def load_data(cls):
        # tensor_transform = transforms.ToTensor()
        # tensor_transform =transforms.Compose([
        # transforms.ToPILImage(),
        # # transforms.Resize((112,112)),
        # transforms.ToTensor()])
        # dataset = datasets.MNIST(root = "./data", train = True, download = True, transform = None)
        # X_train, X_val, X_test, y_train, y_val, y_test = cls.get_dataset_partitions_tf(torch.reshape(dataset.data,(-1,1,28,28))/255, dataset.targets, 0.8, 0.2, 0)
        
        X_train, y_train = cls.get_data_from_trainset(True)
        X_val, y_val = cls.get_data_from_trainset(False)
        
        trainset = ConcatDataset([
        CustomTensorDataset(X_train, y_train),
        # CustomTensorDataset(X_train, y_train, transform=tensor_transform)
                    ])
        valset = CustomTensorDataset(X_val, y_val)
        data_loaders = {}
        data_loaders['train'] = DataLoader(trainset, cls.batch_size, shuffle=False)
       
        data_loaders['val'] = DataLoader(valset, cls.batch_size, shuffle=False)
        cls.dataloaders = data_loaders 
        # return train_loader, val_loader
        
        
    @classmethod
    def set_criterion_optimizer_scheduler(cls, optimizer_type, criterion, optimizer, lr_scheduler):
        cls.optimizer_type = optimizer_type
        cls.criterion = criterion
        cls.optimizer = optimizer
        cls.lr_scheduler = lr_scheduler
        cls.path_save_model = r"./runs/CIFAR10_RESNET18/" + cls.optimizer_type + "/P=30_STD_B_STATS_BATCH_SZ=" + str(BATCH_SIZE) + "LOG_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    
    @classmethod
    def get_data_from_trainset(cls,is_train):
        cifar10_trainset = datasets.CIFAR10(root='./data', train=is_train, download=True, transform=None)
        y = cifar10_trainset.targets
        # y = torch.nn.functional.one_hot(torch.tensor(y), num_classes= 10).float()
        X = cifar10_trainset.data
        max_val = (X.max()).max()
        if (max_val < 255):
            max_val = 255
        X = X/max_val
        X = X.reshape(-1, 3, 32,32)
        X = torch.tensor(X).float()
        return X, y
        
# def load_data():
#     # tensor_transform = transforms.ToTensor()
#     tensor_transform =transforms.Compose([
#     transforms.ToPILImage(),
#     # transforms.Resize((112,112)),
#     transforms.ToTensor()
#     ])
#     # dataset = datasets.MNIST(root = "./data", train = True, download = True, transform = None)
#     # X_train, X_val, X_test, y_train, y_val, y_test = get_dataset_partitions_tf(torch.reshape(dataset.data,(-1,1,28,28))/255, dataset.targets, 0.8, 0.2, 0)
#     X_train, y_train = get_data_from_trainset(True)
#     X_val, y_val = get_data_from_trainset(False)
#     trainset = ConcatDataset([
#     CustomTensorDataset(X_train, y_train),
#     CustomTensorDataset(X_train, y_train, transform=tensor_transform)
#                 ])
#     valset = CustomTensorDataset(X_val, y_val)
    
#     train_loader = DataLoader(trainset, batch_size=32, shuffle=True)
#     val_loader = DataLoader(valset, batch_size=32, shuffle=False)
    
#     return train_loader, val_loader
    @classmethod
    def train_model(cls, num_epochs=1, phases = ['train', 'val']):
        init_lr = cls.optimizer.state_dict()['param_groups'][0]['lr']
        
        val_acc_history = []
        best_model_wts = copy.deepcopy(cls.model.state_dict())
        best_acc = 0.0
        epochs_data = {}
        epochs_grads_std_by_epoch = pd.DataFrame()
        for epoch in range(num_epochs):
            since = time.time()
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            epoch_data = {}
            epoch_lr = cls.optimizer.state_dict()['param_groups'][0]['lr']
            
            model_size = cls.get_model_size()
            percentile99p99 = int(0.9999*model_size)
            
            epoch_grads_norm_l0_at_99p99percntile = torch.empty(0,1).to(DEVICE)
            epoch_grads_norm_l0_at_1000end = torch.empty(0,1).to(DEVICE)
            epoch_grads_norm_l0_l2 = torch.empty(0,1).to(DEVICE)
            epoch_grads_norm_l2 = torch.empty(0,1).to(DEVICE)
            epoch_grads_norm_linf = torch.empty(0,1).to(DEVICE)
            
            init_weight_decay = cls.optimizer.state_dict()['param_groups'][0]['weight_decay']
            for phase in ['train', 'val']: # Each epoch has a training and validation phase
                epoch_grads = torch.zeros(cls.get_model_size()).to(DEVICE)
                epoch_grads_legacy = []
                epoch_loss = 0.0
                if phase == 'train':
                    cls.model.train()  # Set model to training mode
                    dataset_index = 0
                    
                else:
                    cls.model.eval()   # Set model to evaluate mode
                    dataset_index = 1
                   
                running_loss = 0.0
                data_len = 0                
                running_corrects = 0
                # for inputs, labels in phase_data_loader: # Iterate over data
                for inputs, labels in cls.dataloaders[phase]: # Iterate over data
                    # inputs = transforms.functional.resize(inputs, (112, 112))
                    inputs = inputs.to(DEVICE)
    
                    labels = labels.to(DEVICE)
                    data_len += len(labels)
                    cls.optimizer.zero_grad() # Zero the parameter gradients
    
                    with torch.set_grad_enabled(phase == 'train'): # Forward. Track history if only in train
                                    
                        outputs = cls.model(inputs)
                        loss = cls.criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)
    
                        if phase == 'train': # Backward + optimize only if in training phase
                            loss.backward()
                            grads = cls.get_model_grads_as_tensor()
                            
                            grads_no_near0 = grads.clone()
                            grads_no_near0[grads_no_near0 < 1e-3] = 0.0
                            grads_abs_sorted = grads.abs().sort()[0]
                            epoch_grads_norm_l0_at_99p99percntile = torch.cat((epoch_grads_norm_l0_at_99p99percntile, grads_abs_sorted[percentile99p99].reshape(1,1)), dim = 0)
                            epoch_grads_norm_l0_at_1000end = torch.cat((epoch_grads_norm_l0_at_1000end, grads_abs_sorted[-1000].reshape(1,1)), dim = 0)
                            epoch_grads_norm_l0_l2 = torch.cat((epoch_grads_norm_l0_l2, torch.linalg.vector_norm(grads_no_near0, 2).reshape(1,1)), dim = 0)
                            epoch_grads_norm_l2 = torch.cat((epoch_grads_norm_l2, torch.linalg.vector_norm(grads, 2).reshape(1,1)), dim = 0)
                            epoch_grads_norm_linf = torch.cat((epoch_grads_norm_linf, torch.linalg.vector_norm(grads, float('inf')).reshape(1,1)), dim = 0)
                                                              
                            epoch_grads += grads * inputs.size(0)
                            
                            cls.optimizer.step()
    
                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    
                                   
                    
                epoch_loss = running_loss / data_len #len(labels)#len(phase_labels)
                epoch_training_loss = 0.0
                if phase == 'train':
                    epoch_training_loss = epoch_loss
                    
                    
                        
                epoch_grad = epoch_grads / data_len #len(labels)#len(phase_labels)
                if phase == 'val': # Adjust learning rate based on val loss
                    cls.lr_scheduler.step(epoch_loss)
                    
                epoch_acc = running_corrects.double() / data_len#len(labels)#len(phase_labels)
                
                
                print('epoch:{} phase:{} Loss: {:.5f} Acc: {:.5f} Train Grad:{:.5f} Train grad inf norm: {:.5f}'.format(epoch, phase, epoch_loss, epoch_acc, torch.linalg.norm(epoch_grad), torch.linalg.norm(epoch_grad, ord = float('inf'))))
    
                # deep copy the model
                if phase == 'val' and (epoch_acc > best_acc or epoch%10 == 0):
                    best_acc = epoch_acc
                    cls.save_model(epoch, acc = epoch_acc, loss=epoch_training_loss, test_loss = epoch_loss, grad = torch.linalg.norm(epoch_grad), lr = cls.lr_scheduler.get_last_lr())
                    best_model_wts = copy.deepcopy(cls.model.state_dict())
                if phase == 'val':
                    val_acc_history.append(epoch_acc)
                epoch_data[phase] = {"grads_norm_2" : torch.linalg.vector_norm(epoch_grad, 2).item(), 
                                     "grads_norm_inf" : epoch_grad.abs().max().item(),
                                     "loss": epoch_loss, 
                                     "acc": epoch_acc.item(), 
                                     "epoch_lr" : epoch_lr, 
                                     "init_lr" : init_lr, 
                                     "init_weight_decay" : init_weight_decay}
                
                epochs_data[epoch] = epoch_data
                if phase == 'train':
                    epoch_data_dict = epoch_data[phase]
                    epoch_data_dict["mean_epoch_grads_norm_l0_at_99p99percntile"] = epoch_grads_norm_l0_at_99p99percntile.mean().item()
                    epoch_data_dict["std_epoch_grads_norm_l0_at_99p99percntile"] = epoch_grads_norm_l0_at_99p99percntile.std().item()
                    epoch_data_dict["epoch_grads_norm_l0_at_1000end"] = epoch_grads_norm_l0_at_1000end.std().item()
                    
                    
                    epoch_data_dict["std_epoch_grads_norm_linf"] = epoch_grads_norm_linf.std().item()
                    epoch_data_dict["std_epoch_grads_norm_l0_l2"] = epoch_grads_norm_l0_l2.std().item()
                    epoch_data_dict["std_epoch_grads_norm_l2"] = epoch_grads_norm_l2.std().item()
                if phase == 'val':
                    cls.save_epochs_data(epochs_data)
                time_elapsed = time.time() - since
                print('Epoch complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('Best val Acc: {:4f}'.format(best_acc))
    
        # load best model weights
        cls.model.load_state_dict(best_model_wts)
        return cls.model, val_acc_history, epochs_data
    
    
    @classmethod
    def save_epochs_data(cls, epochs_data):
        if not os.path.exists(cls.path_save_model):
            os.makedirs(cls.path_save_model)
        epoch_train_stats_train = pd.DataFrame() 
        epoch_train_stats_val = pd.DataFrame() 
        for i in range(len(epochs_data)):
            new_row = epochs_data[i]['train']
            new_row["file_path"] = cls.path_save_model
            new_row["epoch_num"] = i
            new_row["BATCH_SIZE"] = BATCH_SIZE
            epoch_train_stats_train = pd.concat([epoch_train_stats_train, pd.DataFrame([new_row])], ignore_index = True)
            
            new_row_val = epochs_data[i]['val']
            epoch_train_stats_val = pd.concat([epoch_train_stats_val, pd.DataFrame([new_row_val])], ignore_index = True)

        # model, _ = CIFAR10_RESNET18.train_model(1, phases = ['val'])
        epoch_train_stats_train.to_excel(cls.path_save_model + "/epoch_stats_train.xlsx")
        epoch_train_stats_val.to_excel(cls.path_save_model + "/epoch_stats_val.xlsx")

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
    def get_dataset_partitions_tf(cls,X, y, train_split, val_split, test_split):
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
    def save_model(cls, epoch: int, acc = None, loss=None, test_loss = None, grad = None, lr = None):
        if not os.path.exists(cls.path_save_model):
            os.makedirs(cls.path_save_model)
        model_filename = '{}_{}__test_acc={:1.5f}_train_loss={:1.5f}_test_loss={:1.5f}_lr={:1.5f}_grad={:1.5f}.pth'.format(epoch+1, 'model', acc, loss, test_loss, lr[0], grad.item())
         
        checkpoint = {
            'epoch': epoch,
            'model': cls.model,
            'optimizer': cls.optimizer,
            'scheduler': cls.lr_scheduler,
            'criterion': cls.criterion,
            'path_save_model' : cls.path_save_model,
            }
        torch.save(checkpoint,cls.path_save_model +'/'+ model_filename)
        # cls.model.to(DEVICE)


    @classmethod
    def load_model(cls, model_filepath):
        checkpoint = torch.load(model_filepath)
        cls.path_save_model = checkpoint['path_save_model']
        cls.epoch = checkpoint['epoch']
        cls.model = checkpoint['model']
        cls.optimizer = checkpoint['optimizer']
        cls.lr_scheduler = checkpoint['scheduler']
        cls.criterion = checkpoint['criterion']

    @classmethod
    def set_model(cls, model):
        cls.model = model
       


# def main():
    
#     CIFAR10_RESNET18.init_model(3, 10)
#     CIFAR10_RESNET18.load_data()
    
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(CIFAR10_RESNET18.model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
#     lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
    
#     CIFAR10_RESNET18.set_criterion_optimizer_scheduler(criterion, optimizer, lr_scheduler)
#     model, _ =  CIFAR10_RESNET18.train_model(EPOCHS, phases = ['train', 'val'])
#     model, _ = CIFAR10_RESNET18.train_model(1, phases = ['val'])
    
    

        
        
# if __name__ == '__main__':
#     main()
    
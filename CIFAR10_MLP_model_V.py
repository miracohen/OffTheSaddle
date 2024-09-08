# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 23:34:35 2024

@author: dafna
"""
from numba import cuda
import time
import copy
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, TensorDataset
from torch import Tensor
from torch.autograd import Variable
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.datasets as datasets

import datetime

import pickle 

import os

image_size = [32,32]

input_dim = 8
hidden_dim = 12
output_dim = 10

lr = 0.001 #initial lr
n_epochs = 1000
drop_prob = 0.3
scheduler_exp_gamma = 0.99

log_dir_prefix = "./logs/fit/"
log_dir= log_dir_prefix + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
path_save_dir = r"./runs/CIFAR10/ADAMW_full_classifier/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"_lr="+str(lr)+"_gamma="+str(scheduler_exp_gamma)


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 256



class MLP(torch.nn.Module):  
    
    
    def __init__(self,image_channels, num_classes):
        super(MLP,self).__init__()  
        self.fc1 = torch.nn.Linear(32 * 32 * image_channels,64) 
        self.fc2 = torch.nn.Linear(64,64) 
        self.fc3 = torch.nn.Linear(64,64)
        self.fc4 = torch.nn.Linear(64,num_classes) 
     
    def forward(self,x):
        x = x.view(-1,32 * 32 * 3)    
        x = F.relu(self.fc1(x))  
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=1) 
        return x

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

class CIFAR10_MLP():
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
        cls.model = MLP(image_channels, num_classes)
        cls.model.to(DEVICE)
    
    @classmethod
    def load_data(cls):
        
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
        cls.path_save_model = r"./runs/CIFAR10_MLP/" + cls.optimizer_type + "/STATS_BATCH_SZ=" + str(BATCH_SIZE) + "LOG_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    
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
    

    


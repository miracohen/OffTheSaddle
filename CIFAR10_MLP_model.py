# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 23:34:35 2024

@author: dafna
"""
from numba import cuda


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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

train_split = 0.8
test_split = 0.2
val_split = 0.0

input_dim = 8
hidden_dim = 12
output_dim = 10

lr = 0.001 #initial lr
n_epochs = 1000
batch_size = 256
drop_prob = 0.3
scheduler_exp_gamma = 0.99

log_dir_prefix = "./logs/fit/"
log_dir= log_dir_prefix + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
path_save_dir = r"./runs/CIFAR10/ADAMW_full_classifier/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"_lr="+str(lr)+"_gamma="+str(scheduler_exp_gamma)


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
class CFRAR10_Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2), # output: 64 x 16 x 16
    
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2), # output: 128 x 8 x 8
    
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2), # output: 256 x 4 x 4
    
                nn.Flatten(), 
                nn.Linear(256*4*4, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 10))
        
    def forward(self, x):
        return self.network(x)

class CFRAR10_Classifier_Small_conv(nn.Module):
    def __init__(self):
        super(CFRAR10_Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

def get_data_from_trainset(is_train):
    cifar10_trainset = datasets.CIFAR10(root='./data', train=is_train, download=True, transform=None)
    y = cifar10_trainset.targets
    y = torch.nn.functional.one_hot(torch.tensor(y), num_classes= 10).float()
    y = torch.reshape(y,[-1, output_dim])
    X = cifar10_trainset.data
    max_val = (X.max()).max()
    if (max_val < 255):
        max_val = 255
    X = X/max_val
    X = X.reshape(-1, 3, image_size[0],image_size[1])
    X = torch.tensor(X).float()
    return X, y

def get_data():
    X_train, y_train = get_data_from_trainset(True)
    X_test, y_test = get_data_from_trainset(False)
    return torch.tensor(X_train), torch.tensor(X_test), y_train, y_test

def save_data(path_save_dir, X_train, X_test, y_train, y_test):
    if not os.path.exists(path_save_dir):
      os.makedirs(path_save_dir)
    
    with open(path_save_dir + '/data.pkl', 'wb') as file: 
        pickle.dump((X_train, X_test, y_train, y_test), file) 

def get_dataset_partitions_tf(X, y, train_split, val_split, test_split):
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



def get_num_accurate_predictions(probs, target):
    prediction = torch.zeros(probs.shape).to(DEVICE).scatter(1, probs.argmax(1).unsqueeze (1), 1.0)
    correlation = torch.mul(prediction.float(),target.float())
    num_correct_preds = torch.sum(correlation)
    return num_correct_preds

def model_eval(model,X,y, batch_size, output_dim):
    dataset = TensorDataset(X.to(DEVICE),y.to(DEVICE))
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
    correct_preds = 0.0
    for i, (batch_X, batch_y) in enumerate(data_loader):
        probs = model(batch_X)
        i_correct_preds = get_num_accurate_predictions(probs, batch_y)
        correct_preds += i_correct_preds
    accuracy = correct_preds/len(y)
    return accuracy, correct_preds


def model_step(model,optimizer, criterion, X, y):
    optimizer.zero_grad() # <= initialization of the gradients

    # forward propagation
    probs = model(X)
    #y = y.type('torch.LongTensor')

    y_max_indices = torch.max(y, 1)[1]
    loss = criterion(probs, y_max_indices) # <= compute the loss function

    # Backward propagation
    loss.backward() # <= compute the gradient of the loss/cost function
    optimizer.step() # <= Update the gradients

    accuracy = get_num_accurate_predictions(probs, y)/X.size()[0]

    return accuracy, loss


def get_model_size(model_params):
    size = 0
    for param in model_params:
        flat_param = torch.flatten(param)
        size = size + flat_param.shape[0]
    return size 

def train_model(model, criterion, optimizer_supplier, scheduler_supplier,X_train, y_train, X_test, y_test, batch_size, epoch_num, path_save_dir):
    optimizer = optimizer_supplier(model)
    scheduler = scheduler_supplier(optimizer)
    train_dataset = TensorDataset(X_train,y_train)
    data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                         shuffle=False)
    train_loss = []
    train_accu = []
    test_accu = []

    total_batch = len(X_train) / batch_size

    print('Size of the training dataset is {}'.format(len(X_train)))
    print('Size of the testing dataset is {}'.format(len(X_test)))
    print('Batch size is : {}'.format(batch_size))
    print('Total number of batches is : {0:2.0f}'.format(total_batch))
    print('\nTotal number of epochs is : {0:2.0f}'.format(n_epochs))
    best_score_on_test = 0
    model_size = get_model_size(model.parameters())
    
    for epoch in range(n_epochs):
        total_loss = 0
        epoch_lr = scheduler.get_last_lr()
        
        total_grads = torch.zeros(model_size).to(DEVICE)
        
        for i, (iX, iy) in enumerate(data_loader):
            optimizer.zero_grad() # <= initialization of the gradients
            #accuracy, loss = model_step(model,optimizer, criterion, iX, iy)
            iX = iX.to(DEVICE)
            iy = iy.to(DEVICE)
            
            probs = model(iX)
            
            y_max_indices = torch.max(iy, 1)[1]
            loss = criterion(probs, y_max_indices) # <= compute the loss function
            
            loss.backward() # <= compute the gradient of the loss/cost function
            

            accuracy = get_num_accurate_predictions(probs, iy)/iX.size()[0]
            
            train_accu.append(accuracy)
            train_loss.append(loss.item())
            
            iGrads = get_model_grads_as_tensor(model)
            total_grads = total_grads + iGrads
            
            # if i % 100 == 0:
            #     print("Epoch= {},\t batch = {},\t loss = {:2.4f},\t accuracy = {:2.4f}".format(epoch+1, i, train_loss[-1], train_accu[-1]))

            total_loss += loss.data 
            optimizer.step() # <= Update the gradients

        train_loss.append(total_loss/i)
        train_accuracy, train_correct_preds = model_eval(model,X_train,y_train, batch_size, output_dim)
        train_accu.append(train_accuracy)
        test_accuracy, test_correct_preds = model_eval(model,X_test,y_test, batch_size, output_dim)
        avg_grad_norm = torch.linalg.vector_norm(total_grads/i)
        if (test_accuracy > best_score_on_test):
            save_model(path_save_dir, model, optimizer, epoch, test_accuracy, avg_grad_norm, epoch_lr[0], train_accuracy, test_accuracy)
            best_score_on_test = test_accuracy
        else:
            if (epoch % 25 == 0):
                save_model(path_save_dir, model, optimizer, epoch, test_accuracy, avg_grad_norm, epoch_lr[0], train_accuracy, test_accuracy)
        test_accu.append(test_accuracy)
        print("Epoch= {},\t avg loss = {:2.4f},\t lr = {},\t |avg grad| = {:2.4f},\t test accuracy = {:2.4f}, train accuracy = {:2.4f}".format(epoch+1, total_loss/i, epoch_lr[0], avg_grad_norm, test_accuracy, train_accuracy))
        
        scheduler.step()
    return train_loss, train_accu, test_accu


def save_model(path_save_dir, model, optimizer, epoch: int, loss=None, grad=None, lr=None, train_accuracy = None, test_accuracy = None):
    if not os.path.exists(path_save_dir):
      os.makedirs(path_save_dir)

    model_filename = '{}_{}_loss={:1.5f}_avg_grad={:1.5f}_lr={:1.5f}_train_acc={:1.5f}_test_acc={:1.5f}.pth'.format(epoch+1, 'model', loss, torch.linalg.vector_norm(grad), lr, train_accuracy, test_accuracy)
  
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'grad': grad,
            }, os.path.join(path_save_dir, model_filename))
    
    print("saved model: ", model_filename)


def get_model_grads_as_tensor(model):
    grads = []
    for param in model.parameters():
        grads.append(param.grad.view(-1))
    grads = torch.cat(grads)
    grads = grads.clone() #detach
    return grads

def main():
    
    model = CFRAR10_Classifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer_supplier = lambda model : torch.optim.Adam(params=model.parameters(), lr = lr)
    # lambda model : torch.optim.SGD(params=model.parameters(), lr = lr)
    
    
    # lambda model : torch.optim.Adam(params=model.parameters(), lr = lr)
    #
    exp_decay_scheduler_supplier = lambda optimizer: torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_exp_gamma)
    X_train, X_test, y_train, y_test = get_data()
    save_data(path_save_dir, X_train, X_test, y_train, y_test)
    train_model(model, criterion, optimizer_supplier, exp_decay_scheduler_supplier,X_train, y_train, X_test, y_test, batch_size, n_epochs, path_save_dir)

    

if __name__ == '__main__':
    main()
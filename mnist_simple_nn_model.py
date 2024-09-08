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

image_size = [28,28]

train_split = 0.8
test_split = 0.2
val_split = 0.0

input_dim = 8
hidden_dim = 12
output_dim = 10

lr = 0.0001 #initial lr
n_epochs = 10000
batch_size = 128
drop_prob = 0.3
scheduler_exp_gamma = 0.998

log_dir_prefix = "./logs/fit/"
log_dir= log_dir_prefix + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
path_save_dir = r"./runs/MNIST/ADAMW_full_classifier/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"_lr="+str(lr)+"_gamma="+str(scheduler_exp_gamma)


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class MNIST_Classifier(nn.Module):
    def __init__(self):
        super(MNIST_Classifier, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=2, stride=1, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=drop_prob)       )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=drop_prob)        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            torch.nn.Dropout(drop_prob)       )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Linear(3200, 1024, bias=True),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(p=drop_prob))
        self.layer5 = torch.nn.Linear(1024, output_dim, bias=True)


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)   # Flatten them for layer4
        out = self.layer4(out)
        out = self.layer5(out)
        #out = nn.Softmax()(out)
        return out
    


    
def get_data():
    
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    len_full_dataset = len(mnist_trainset)
    
    
    
    train_set, test_set = torch.utils.data.random_split(mnist_trainset, [train_split, 1-train_split])
    
    y = mnist_trainset.targets
    y = torch.nn.functional.one_hot(y.long(), num_classes= 10)
    y = torch.reshape(y,[-1, output_dim])

    X = mnist_trainset.data
    max_val = (X.max()).max()
    if (max_val < 255):
        max_val = 255
    X = X/max_val
    X = X.reshape(-1, 1, image_size[0],image_size[1])

    X_train, X_val, X_test, y_train, y_val, y_test = get_dataset_partitions_tf(X, y, train_split, val_split, test_split)
    return X_train, X_val, X_test, y_train, y_val, y_test

def save_data(path_save_dir, X_train, X_val, X_test, y_train, y_val, y_test):
    if not os.path.exists(path_save_dir):
      os.makedirs(path_save_dir)
    
    with open(path_save_dir + '/data.pkl', 'wb') as file: 
        pickle.dump((X_train, X_val, X_test, y_train, y_val, y_test), file) 

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

        train_loss.append(total_loss)
        train_accuracy, train_correct_preds = model_eval(model,X_train,y_train, batch_size, output_dim)
        train_accu.append(train_accuracy)
        test_accuracy, test_correct_preds = model_eval(model,X_test,y_test, batch_size, output_dim)
        grad_norm = torch.linalg.vector_norm(total_grads)
        if (test_accuracy > best_score_on_test):
            save_model(path_save_dir, model, optimizer, epoch, test_accuracy, grad_norm, epoch_lr[0], train_accuracy, test_accuracy)
            best_score_on_test = test_accuracy
        else:
            if (epoch % 25 == 0):
                save_model(path_save_dir, model, optimizer, epoch, test_accuracy, grad_norm, epoch_lr[0], train_accuracy, test_accuracy)
        test_accu.append(test_accuracy)
        print("Epoch= {},\t total loss = {:2.4f},\t lr = {},\t |sum of grad| = {:2.4f},\t test accuracy = {:2.4f}, train accuracy = {:2.4f}".format(epoch+1, total_loss, epoch_lr[0], grad_norm, test_accuracy, train_accuracy))
        
        scheduler.step()
    return train_loss, train_accu, test_accu


def save_model(path_save_dir, model, optimizer, epoch: int, loss=None, grad=None, lr=None, train_accuracy = None, test_accuracy = None):
    if not os.path.exists(path_save_dir):
      os.makedirs(path_save_dir)

    model_filename = '{}_{}_loss={:1.5f}_grad={:1.5f}_lr={:1.5f}_train_acc={:1.5f}_test_acc={:1.5f}.pth'.format(epoch+1, 'model', loss, torch.linalg.vector_norm(grad), lr, train_accuracy, test_accuracy)
  
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
    
    model = MNIST_Classifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer_supplier = lambda model : torch.optim.Adam(params=model.parameters(), lr = lr)
    # lambda model : torch.optim.SGD(params=model.parameters(), lr = lr)
    
    
    # lambda model : torch.optim.Adam(params=model.parameters(), lr = lr)
    #
    exp_decay_scheduler_supplier = lambda optimizer: torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_exp_gamma)
    X_train, X_val, X_test, y_train, y_val, y_test = get_data()
    save_data(path_save_dir, X_train, X_val, X_test, y_train, y_val, y_test)
    train_model(model, criterion, optimizer_supplier, exp_decay_scheduler_supplier,X_train, y_train, X_test, y_test, batch_size, n_epochs, path_save_dir)

    

if __name__ == '__main__':
    main()
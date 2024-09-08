# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 17:36:00 2024

@author: mirac
"""

from numba import cuda


import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.func import hessian

import torch.nn.functional as F
from torch.func import functional_call
from torch.utils.data import DataLoader, TensorDataset


import os

import datetime
import mini_nn_hessian_reighley_quotient


N_HIDDEN = 20
OUTPUT_DIM = 10
IMAGE_SIZE = [28,28]
INPUT_DIM = IMAGE_SIZE[0]*IMAGE_SIZE[1]
TRAIN_SPLIT = 0.8
LR = 0.0001
GAMMA = 0.995
EPOCH_NUM = 5000
BATCH_SIZE = 512

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PATH_SAVE_DIR = r"./runs/MNIST_LINEAR_HESSIAN_TRY/ADAMW_full_classifier/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"n_hidden="+str(N_HIDDEN)+"_lr="+str(LR)+"_gamma="+str(GAMMA)

class Simple_NN_MNIST(nn.Module):
    def __init__(self):
        super().__init__()

        # 1 Hidden Layer Network
        self.fc1 = nn.Linear(INPUT_DIM, N_HIDDEN)
        self.fc2 = nn.Linear(N_HIDDEN, OUTPUT_DIM)

    def forward(self, x):
        t = F.relu(self.fc1(x))
        t = F.relu(self.fc2(t))
        return t
    
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
    
def get_data():
    
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    len_full_dataset = len(mnist_trainset)
    
    
    
    # train_set, test_set = torch.utils.data.random_split(mnist_trainset, [TRAIN_SPLIT, 1-TRAIN_SPLIT])
    
    y = mnist_trainset.targets
    y = torch.nn.functional.one_hot(y.long(), OUTPUT_DIM)
    y = torch.reshape(y,[-1, OUTPUT_DIM])

    X = mnist_trainset.data
    max_val = (X.max()).max()
    if (max_val < 255):
        max_val = 255
    X = X/max_val
    X = X.reshape(-1, IMAGE_SIZE[0]*IMAGE_SIZE[1])

    X_train, X_val, X_test, y_train, y_val, y_test = get_dataset_partitions_tf(X, y, TRAIN_SPLIT, 0, 1-TRAIN_SPLIT)
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
    print('\nTotal number of epochs is : {0:2.0f}'.format(EPOCH_NUM))
    best_score_on_test = 0
    model_size = get_model_size(model.named_parameters())
    
    for epoch in range(EPOCH_NUM):
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
        
        
        
        
            # def model_loss_sum_params(params, X_train, y_train):
            #     #pred_y: torch.Tensor = _stateless.functional_call(model, {n: p for n, p in zip(names, params)}, X)
            #     pred_y = functional_call(model, params, X_train)
            #     loss = criterion(pred_y, y_train)
            #     return loss.sum()
            
            # H = hessian(model_loss_sum_params)(
            #     dict(model.named_parameters()), iX, y_max_indices)

            # # reshaping hessian to nXn matrix
            # HM = reshape_hessian(H, model)
            # # HM should be symmetrical matrix. If it isn't (due to nummerical instability) and in order to avoid complex eig vectors:
            # diff = HM - HM.t()
            # if (diff.max() > 0):
            #     HM = (HM + HM.t())/2
            # #print("L0 by finite differencing: ", L0)
            # #HM_condition_number = torch.linalg.cond(HM)
            # #print("condition number of hessian: ", HM_condition_number)
            # V0_HM, L0_HM, Vn_HM, Ln_HM = get_V0L0_VnLn(HM)
        
        
        train_loss.append(total_loss)
        train_accuracy, train_correct_preds = model_eval(model,X_train,y_train, batch_size, OUTPUT_DIM)
        train_accu.append(train_accuracy)
        test_accuracy, test_correct_preds = model_eval(model,X_test,y_test, BATCH_SIZE, OUTPUT_DIM)
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
    
def get_sorted_LV(symA):

    L, V = torch.linalg.eigh(symA)

    sorted_L, indices = torch.sort(L, dim=0, descending=True, out=None)

    sortedV = V[:, indices]
    sortedL = L[indices]
    return sortedL, sortedV



def get_sorted_LV_by_abs_L(symA):

    L, V = torch.linalg.eigh(symA)

    sorted_absL, indices = torch.sort(abs(L), dim=0, descending=True, out=None)

    sortedV = V[:, indices]
    sortedL = L[indices]
    return sortedL, sortedV



def get_V0L0_VnLn(A):
    L, V = get_sorted_LV_by_abs_L(A)
    V0 = V[:, 0]
    L0 = L[0]
    L, V = get_sorted_LV(A)
    Vn = V[:, -1]
    Ln = L[-1]
    return V0, L0, Vn, Ln


def get_model_size(model_named_params):
    size = 0
    for name, param in model_named_params:
        flat_param = torch.flatten(param)
        size = size + flat_param.shape[0]
    return size


def get_model_grads_as_tensor(model):
    grads = []
    for param in model.parameters():
        grads.append(param.grad.view(-1))
    grads = torch.cat(grads)
    grads = grads.clone() #detach
    return grads



def reshape_hessian(H, model):
    model_size = get_model_size(model.named_parameters())
    HM = torch.zeros([model_size, model_size])
    acc_i = 0
    acc_j = 0
    for i, (i_name, i_param) in enumerate(model.named_parameters()):
        for j, (j_name, j_param) in enumerate(model.named_parameters()):
            i_param_size = torch.flatten(i_param).shape[0]
            j_param_size = torch.flatten(j_param).shape[0]
            Hij = H[i_name][j_name]
            reshaped_Hij = torch.reshape(Hij, [i_param_size, j_param_size])
            HM[acc_i:acc_i+i_param_size, acc_j:acc_j+j_param_size] = reshaped_Hij

            acc_j = acc_j + j_param_size
        acc_j = 0
        acc_i = acc_i + i_param_size
    return HM






def main():
    model = Simple_NN_MNIST().to(DEVICE)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer_supplier = lambda model : torch.optim.Adam(params=model.parameters(), lr = LR)
    # lambda model : torch.optim.SGD(params=model.parameters(), lr = lr)
    
    
    # lambda model : torch.optim.Adam(params=model.parameters(), lr = lr)
    #
    exp_decay_scheduler_supplier = lambda optimizer: torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA)
    X_train, X_val, X_test, y_train, y_val, y_test = get_data()
    train_model(model, criterion, optimizer_supplier, exp_decay_scheduler_supplier,X_train, y_train, X_test, y_test, BATCH_SIZE, EPOCH_NUM, PATH_SAVE_DIR)
if __name__ == '__main__':
    main()



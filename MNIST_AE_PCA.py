# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 21:53:15 2024

@author: mirac
"""
from numba import cuda


import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset

import os

import datetime


EPOCHS = 1000
LR =1e-5
GAMMA = 0.99
DROP_PROB = 0.3
BATCH_SIZE = 128
TRAIN_SPLIT = 0.8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIM = 10
IMAGE_SIZE = [28,28]
N_COMPONENTS = 230 #(150 for accuracy of 0.8, 230 for accuracy of 0.9, 325 for accuracy of 0.95)


PATH_SAVE_MODEL = r"./runs/MNIST_AE_Conv/ADAMW/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"LR="+str(LR)+"gamma="+str(GAMMA)+"num_v="+str(N_COMPONENTS)



class MNIST_AE_PCA(nn.Module):
	def __init__(self):
		super().__init__()
		
		self.encoder = nn.Sequential(
			nn.Linear(N_COMPONENTS, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 36),
			nn.ReLU(),
			nn.Linear(36, 18),
		)
		
		self.decoder = nn.Sequential(
			nn.Linear(18, 36),
			nn.ReLU(),
			nn.Linear(36, 64),
			nn.ReLU(),
			nn.Linear(64, 128),
			nn.ReLU(),
			nn.Linear(128, N_COMPONENTS),
			nn.Sigmoid()
		)

	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded
    
    
    
def get_model_size(model):
    size = 0
    for param in model.parameters():
        flat_param = torch.flatten(param)
        size += flat_param.shape[0]
    return size

def get_model_grads_as_tensor(model):
    grads = []
    for param in model.parameters():
        grads.append(param.grad.view(-1))
    grads = torch.cat(grads)
    grads= grads.clone() #detach
    return grads

def save_model(path_save_model, model, optimizer, epoch: int, loss=None, test_loss = None, grad = None, lr = None):
    if not os.path.exists(path_save_model):
        os.makedirs(path_save_model)
    model_filename = '{}_{}_train_loss={:1.5f}_test_loss={:1.5f}_lr={:1.5f}_grad={:1.5f}.pth'.format(epoch+1, 'model', loss, test_loss, lr, grad)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_disct': optimizer.state_dict(),
        'loss': loss,
        'grad' : grad,
        }, os.path.join(path_save_model,model_filename))
    model.to(DEVICE)    


def model_eval(model, data_loader, pca_V):
    loss = 0.
    pca_loss = 0.0

    for i, (batch_X,_) in enumerate(data_loader):
        pca_batch_X = torch.matmul(batch_X, pca_V)
        reconstructed_x = model(pca_batch_X)
        i_pca_loss = ((reconstructed_x - pca_batch_X)**2).mean()
        pca_loss += i_pca_loss
        reconstructed_x = torch.matmul(reconstructed_x, pca_V.t())
        i_loss = ((reconstructed_x - batch_X)**2).mean()
        loss += i_loss
    return loss, pca_loss
    
def train_model(model, loss_function, optimizer_supplier, scheduler_supplier, epochs, X_train, y_train, X_test, y_test, pca_V, path_save_model):
    optimizer = optimizer_supplier(model)
    scheduler = scheduler_supplier(optimizer)
    pca_X_train = torch.matmul(X_train.to(DEVICE), pca_V)
    
    train_dataset = TensorDataset(pca_X_train.to(DEVICE), y_train.to(DEVICE))
    test_dataset = TensorDataset(X_test.to(DEVICE), y_test.to(DEVICE))
    grads = []
    losses = []
    train_dataset_loader = DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True)
    test_dataset_loader = DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, shuffle = True)
    for epoch in range(epochs):
        epoch_lr = scheduler.get_last_lr()
        total_grad = torch.zeros(get_model_size(model)).to(DEVICE)
        total_loss = 0.0
        for (images, _) in train_dataset_loader:

          optimizer.zero_grad()

          reconstructed = model(images)
          
          loss = loss_function(reconstructed, images)
          total_loss += loss.item()
          loss.backward()
          
          iGrad = get_model_grads_as_tensor(model)
          total_grad = total_grad + iGrad
          optimizer.step()
           
          # Storing the losses in a list for plotting
        test_accuracy, test_pca_accuracy  = model_eval(model, test_dataset_loader, pca_V)
        
        grad_norm = torch.linalg.vector_norm(total_grad)
 
        scheduler.step()
        save_model(path_save_model, model, optimizer, epoch, total_loss, test_accuracy, grad_norm, epoch_lr[0])
        print ("Epoch: {}\t train_loss ={:1.5f}\ttest_loss= {:1.5f}\ttest_pca_loss={:1.5f}\tgrad={:1.5f}\t lr={:1.5f}".format(epoch, total_loss, test_accuracy, test_pca_accuracy, grad_norm, epoch_lr[0]))

    
    
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
    
    # m = X_train.mean(0, keepdim=True)
    # s = X_train.std(0, unbiased=False, keepdim=True)
    
    # X_train =  (X_train - m)/s
    # X_test =  (X_test - m)/s
    
    U,S,pca_V = torch.pca_lowrank(X_train, q=N_COMPONENTS, center=True, niter=3)
    
    pca_V = pca_V.to(DEVICE)
    return X_train, X_val, X_test, y_train, y_val, y_test, pca_V 

def main():
    # Transforms images to a PyTorch Tensor
    tensor_transform = transforms.ToTensor()
 
    # Download the MNIST Dataset
    dataset = datasets.MNIST(root = "./data", train = True, download = True, transform = tensor_transform)
 
    # DataLoader is used to load the dataset 
    # for training
    
    # model = MNIST_AE_Small().to(DEVICE)
    model = MNIST_AE_PCA().to(DEVICE)
    loss_function = torch.nn.MSELoss().to(DEVICE)
    #optimizer = torch.optim.Adam(model.parameters(), lr = LR, weight_decay = WEIGHT_DECAY)
    optimizer_supplier = lambda optimizer: torch.optim.AdamW(model.parameters(), lr = LR)
    exp_decay_scheduler_supplier = lambda optimizer: torch.optim.lr_scheduler.ExponentialLR(optimizer, GAMMA)
    X_train, X_val, X_test, y_train, y_val, y_test, pca_V = get_data()
    
    
    
    train_model(model, loss_function, optimizer_supplier, exp_decay_scheduler_supplier, EPOCHS, X_train, y_train, X_test, y_test, pca_V, PATH_SAVE_MODEL)
    
    
  
    
if __name__ == '__main__':
    main()
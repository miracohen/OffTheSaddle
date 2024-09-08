# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 10:11:37 2024

@author: mirac
"""

# import the necessary packages
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import datetime
import numpy as np


# set device to 'cpu' or 'cuda' (GPU) based on availability
# for model training and testing
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# define model hyperparameters
LR = 0.001
PATIENCE = 2
IMAGE_SIZE = 28
CHANNELS = 1
BATCH_SIZE = 64
EMBEDDING_DIM = 2
EPOCHS = 100


PATH_SAVE_MODEL = r"./runs/MNIST_AE_ED/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"LR="+str(LR)#+"gamma="+str(GAMMA)

class Encoder(nn.Module):
    def __init__(self, image_size, channels, embedding_dim):
        super(Encoder, self).__init__()
        # define convolutional layers
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        # variable to store the shape of the output tensor before flattening
        # the features, it will be used in decoders input while reconstructing
        self.shape_before_flattening = None
        # compute the flattened size after convolutions
        flattened_size = (image_size // 8) * (image_size // 8) * 128
        # define fully connected layer to create embeddings
        self.fc = nn.Linear(flattened_size, embedding_dim)
    def forward(self, x):
        # apply ReLU activations after each convolutional layer
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # store the shape before flattening
        self.shape_before_flattening = x.shape[1:]
        # flatten the tensor
        x = x.view(x.size(0), -1)
        # apply fully connected layer to generate embeddings
        x = self.fc(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, embedding_dim, shape_before_flattening, channels):
        super(Decoder, self).__init__()
        # define fully connected layer to unflatten the embeddings
        self.fc = nn.Linear(embedding_dim, np.prod(shape_before_flattening))
        # store the shape before flattening
        self.reshape_dim = shape_before_flattening
        # define transpose convolutional layers
        self.deconv1 = nn.ConvTranspose2d(
            128, 128, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.deconv2 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.deconv3 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        # define final convolutional layer to generate output image
        self.conv1 = nn.Conv2d(32, channels, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        # apply fully connected layer to unflatten the embeddings
        x = self.fc(x)
        # reshape the tensor to match shape before flattening
        x = x.view(x.size(0), *self.reshape_dim)
        # apply ReLU activations after each transpose convolutional layer
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        # apply sigmoid activation to the final convolutional layer to generate output image
        x = torch.sigmoid(self.conv1(x))
        return x


def train_model(model,optimizer, criterion, train_loader):
    # initialize the best validation loss as infinity
    best_val_loss = float("inf")
    # start training by looping over the number of epochs
    for epoch in range(EPOCHS):
        print(f"Epoch: {epoch + 1}/{EPOCHS}")
        # set the encoder and decoder models to training mode
        model.encoder.train()
        model.decoder.train()
        # initialize running loss as 0
        running_loss = 0.0
        # loop over the batches of the training dataset
        for batch_idx, (data, _) in enumerate(train_loader):
            # move the data to the device (GPU or CPU)
            # data = data.to(config.DEVICE)
            # reset the gradients of the optimizer
            optimizer.zero_grad()
            # forward pass: encode the data and decode the encoded representation
            encoded = model.encoder(data)
            decoded = model.decoder(encoded)
            # compute the reconstruction loss between the decoded output and
            # the original data
            loss = criterion(decoded, data)
            # backward pass: compute the gradients
            loss.backward()
            # update the model weights
            optimizer.step()
            # accumulate the loss for the current batch
            running_loss += loss.item()



def main():
    if not os.exist(PATH_SAVE_MODEL):
        os.makedirs(PATH_SAVE_MODEL)
    
    dataset = torchvision.datasets.MNIST(root = "./data", train = True, download = True, transform = torchvision.tensor_transform)
    trainset = torchvision.datasets.MNIST("data", train=True, download=True, transform=torchvision.transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
# Load the FashionMNIST test data and create a dataloader
    testset = torchvision.datasets.MNIST("data", train=False, download=True, transform=torchvision.transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)
    

if __name__ == '__main__':
    main()
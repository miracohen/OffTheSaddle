# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 09:08:37 2023

@author: dafna
"""
from numba import cuda


import torch
#import functorch
import torch.nn as nn
# from torch.autograd import jacobian
# from torch.autograd.functional import hessian
# from torch.autograd import grad
import torch.nn.functional as F
from torch.func import functional_call
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torchvision.transforms as transforms

#from torch.nn.utils import stateless

import pickle

import pandas as pd
import os

import datetime
import time

# from MNIST_resnet18 import Block, ResNet_18, CustomTensorDataset, MNIST_RESNET18
from CIFAR10_resnet18 import Block, ResNet_18, CustomTensorDataset, CIFAR10_RESNET18

import copy


from torch.func import hessian, grad


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LR = 0.0001
# GLOBALS
MAX_ITER_NUM = 20
COSIN_DIST = 2*1e-02
MAX_DIST_CONVERGENCE = 1e-07
EM = 1e-07  # machine precision
LANCZOS_DIM = 1e+01

BATCH_SIZE = 256



N_HALT_MULTIPLIER = 2

OUT_PATH_SUMMARY = r'C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD'

MODEL_PATHS =[ 
        # r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_MLP\SGD\STATS_BATCH_SZ=256LOG_20240729-215031\98_model__test_acc=0.45960_train_loss=0.00000_test_loss=1.99703_lr=0.09500_grad=0.00000.pth",
        # r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_MLP\SGD\STATS_BATCH_SZ=256LOG_20240729-213712\99_model__test_acc=0.46510_train_loss=0.00000_test_loss=1.99154_lr=0.10000_grad=0.00000.pth",
        # r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_MLP\SGD\STATS_BATCH_SZ=256LOG_20240729-212823\97_model__test_acc=0.47610_train_loss=0.00000_test_loss=1.97963_lr=0.09500_grad=0.00000.pth",
        # r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_MLP\SGD\STATS_BATCH_SZ=256LOG_20240729-212357\98_model__test_acc=0.47160_train_loss=0.00000_test_loss=1.98346_lr=0.09025_grad=0.00000.pth",
        # r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_MLP\SGD\STATS_BATCH_SZ=256LOG_20240729-212128\91_model__test_acc=0.46850_train_loss=0.00000_test_loss=1.98809_lr=0.09025_grad=0.00000.pth",
        # r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_MLP\SGD\STATS_BATCH_SZ=256LOG_20240729-211648\95_model__test_acc=0.45610_train_loss=0.00000_test_loss=1.99987_lr=0.09025_grad=0.00000.pth",
        # r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_MLP\SGD\STATS_BATCH_SZ=256LOG_20240729-211446\100_model__test_acc=0.46130_train_loss=0.00000_test_loss=1.99549_lr=0.10000_grad=0.00000.pth",
        # r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_MLP\SGD\STATS_BATCH_SZ=256LOG_20240729-205558\95_model__test_acc=0.46300_train_loss=0.00000_test_loss=1.99379_lr=0.10000_grad=0.00000.pth",
    
        # r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\2_lin_layers_P=30_STD_B_STATS_BATCH_SZ=256LOG_20240724-094109\92_model__test_acc=0.58540_train_loss=0.00000_test_loss=2.79390_lr=0.06983_grad=0.00000.pth",
        # r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\2_lin_layers_P=30_STD_B_STATS_BATCH_SZ=256LOG_20240724-094655\93_model__test_acc=0.58020_train_loss=0.00000_test_loss=2.82908_lr=0.06983_grad=0.00000.pth",
        # r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\2_lin_layers_P=30_STD_B_STATS_BATCH_SZ=256LOG_20240724-095115\93_model__test_acc=0.57600_train_loss=0.00000_test_loss=2.86358_lr=0.06983_grad=0.00000.pth",
        # r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\2_lin_layers_P=30_STD_B_STATS_BATCH_SZ=256LOG_20240725-220237\91_model__test_acc=0.59300_train_loss=0.00000_test_loss=2.81837_lr=0.06983_grad=0.00000.pth",
        # r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\2_lin_layers_P=30_STD_B_STATS_BATCH_SZ=256LOG_20240725-212256\100_model__test_acc=0.57220_train_loss=0.00000_test_loss=2.85969_lr=0.06634_grad=0.00000.pth"
        # r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\2_lin_layers_P=30_STD_B_STATS_BATCH_SZ=256LOG_20240725-210959\99_model__test_acc=0.58240_train_loss=0.00000_test_loss=2.79760_lr=0.06634_grad=0.00000.pth"
    
    
    
        r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\P=30_STD_B_STATS_BATCH_SZ=256LOG_20240714-233855\96_model__test_acc=0.60030_train_loss=0.00000_test_loss=2.80336_lr=0.06634_grad=0.00000.pth",    
        r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\P=30_STD_B_STATS_BATCH_SZ=256LOG_20240715-214055\97_model__test_acc=0.56760_train_loss=0.00000_test_loss=4.28845_lr=0.06634_grad=0.00000.pth",
        r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\P=30_STD_B_STATS_BATCH_SZ=256LOG_20240714-234010\92_model__test_acc=0.60390_train_loss=0.00000_test_loss=2.63514_lr=0.06983_grad=0.00000.pth",
        r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\P=30_STD_B_STATS_BATCH_SZ=256LOG_20240712-231631\95_model__test_acc=0.60240_train_loss=0.00000_test_loss=2.69332_lr=0.06634_grad=0.00000.pth",
        r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\P=30_STD_B_STATS_BATCH_SZ=256LOG_20240714-225438\51_model__test_acc=0.61800_train_loss=0.00000_test_loss=2.45781_lr=0.08145_grad=0.00000.pth",
        r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\P=30_STD_B_STATS_BATCH_SZ=256LOG_20240713-002435\32_model__test_acc=0.61360_train_loss=0.00000_test_loss=2.51258_lr=0.09025_grad=0.00000.pth",
        r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\P=30_STD_B_STATS_BATCH_SZ=256LOG_20240713-180601\99_model__test_acc=0.60830_train_loss=0.00000_test_loss=2.64720_lr=0.06634_grad=0.00000.pth",
        r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\P=30_STD_B_STATS_BATCH_SZ=256LOG_20240723-181932\91_model__test_acc=0.56370_train_loss=0.00000_test_loss=4.66027_lr=0.06983_grad=0.00000.pth",
    
    
    
    
    
        # r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240328-125845LR=0.01WEIGHT_DECAY=0.01\91_model__test_acc=0.65560_train_loss=0.02753_test_loss=1.12555_lr=0.00000_grad=0.06716.pth",
        # r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240328-125858LR=0.01WEIGHT_DECAY=0.01\81_model__test_acc=0.66470_train_loss=0.02722_test_loss=1.10858_lr=0.00000_grad=0.06759.pth",
        # r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240328-125833LR=0.01WEIGHT_DECAY=0.01\94_model__test_acc=0.64970_train_loss=0.02692_test_loss=1.14297_lr=0.00000_grad=0.06718.pth",
        # r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240328-100039LR=0.01WEIGHT_DECAY=0.01\84_model__test_acc=0.65860_train_loss=0.02666_test_loss=1.11371_lr=0.00000_grad=0.06807.pth",
        # r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240326-173955LR=0.01WEIGHT_DECAY=0.01\81_model__test_acc=0.65940_train_loss=0.02735_test_loss=1.10770_lr=0.00000_grad=0.06689.pth",
        # r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240326-142704LR=0.01WEIGHT_DECAY=0.01\71_model__test_acc=0.65680_train_loss=0.02699_test_loss=1.12244_lr=0.00000_grad=0.06644.pth",
        # r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240326-142603LR=0.01WEIGHT_DECAY=0.01\72_model__test_acc=0.65770_train_loss=0.02721_test_loss=1.09544_lr=0.00000_grad=0.06745.pth",
        # r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240326-142539LR=0.01WEIGHT_DECAY=0.01\88_model__test_acc=0.65630_train_loss=0.02693_test_loss=1.11485_lr=0.00000_grad=0.06718.pth",
        # r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240326-142534LR=0.01WEIGHT_DECAY=0.01\76_model__test_acc=0.66260_train_loss=0.02675_test_loss=1.09290_lr=0.00000_grad=0.06762.pth",
        # r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240325-221555LR=0.01WEIGHT_DECAY=0.01\86_model__test_acc=0.65420_train_loss=0.02705_test_loss=1.13709_lr=0.00000_grad=0.06622.pth",
        # r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240325-221631LR=0.01WEIGHT_DECAY=0.01\81_model__test_acc=0.65740_train_loss=0.02742_test_loss=1.11738_lr=0.00000_grad=0.06801.pth",
        # r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240325-221603LR=0.01WEIGHT_DECAY=0.01\71_model__test_acc=0.65320_train_loss=0.02686_test_loss=1.12823_lr=0.00000_grad=0.06737.pth",
        # r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240325-221655LR=0.01WEIGHT_DECAY=0.01\72_model__test_acc=0.65480_train_loss=0.02689_test_loss=1.12950_lr=0.00000_grad=0.06649.pth",
        # r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240325-221715LR=0.01WEIGHT_DECAY=0.01\91_model__test_acc=0.65110_train_loss=0.02748_test_loss=1.13140_lr=0.00000_grad=0.06763.pth",
        # r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240325-075155LR=0.01WEIGHT_DECAY=0.01\13_model__test_acc=0.65960_train_loss=0.05885_test_loss=1.08773_lr=0.00100_grad=0.12887.pth",
        # r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240325-075217LR=0.01WEIGHT_DECAY=0.01\11_model__test_acc=0.66030_train_loss=0.15025_test_loss=1.06402_lr=0.00100_grad=0.27435.pth",
        # r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240322-181049LR=0.01WEIGHT_DECAY=0.01\86_model__test_acc=0.66260_train_loss=0.02692_test_loss=1.10135_lr=0.00000_grad=0.06617.pth",
        # r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240322-181243LR=0.01WEIGHT_DECAY=0.01\13_model__test_acc=0.65790_train_loss=0.05919_test_loss=1.09230_lr=0.00100_grad=0.12977.pth",
        # r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240322-180247LR=0.01WEIGHT_DECAY=0.01\14_model__test_acc=0.65630_train_loss=0.04158_test_loss=1.11635_lr=0.00100_grad=0.09695.pth",
        # r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240322-180701LR=0.01WEIGHT_DECAY=0.01\11_model__test_acc=0.65220_train_loss=0.15224_test_loss=1.08844_lr=0.00100_grad=0.27484.pth",
        # r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240322-180342LR=0.01WEIGHT_DECAY=0.01\11_model__test_acc=0.66000_train_loss=0.15108_test_loss=1.05759_lr=0.00100_grad=0.27612.pth",
        

    ]

class Simple_NN(nn.Module):
    def __init__(self, N_HIDDEN):
        super().__init__()

        # 1 Hidden Layer Network
        self.fc1 = nn.Linear(1, N_HIDDEN)
        #self.fc2 = nn.Linear(N_HIDDEN, N_HIDDEN)
        self.fc3 = nn.Linear(N_HIDDEN, 1)

    def forward(self, x):
        t = F.relu(self.fc1(x))
        #t = F.relu(self.fc2(t))
        t = F.sigmoid(self.fc3(t))
        return t

def set_model_parameters(model, new_parameters):
    with torch.no_grad():
        data_accumulative_len = 0
        for param in model.parameters():
            flat_param = torch.flatten(param)
            len_i = flat_param.shape[0]
            new_param_values = new_parameters[data_accumulative_len:
                                              data_accumulative_len + len_i].reshape(param.data.shape)
            param.copy_(new_param_values)
            data_accumulative_len = data_accumulative_len + len_i
    return 

def get_dist(v1, v2):
    cosi = torch.nn.CosineSimilarity(dim=0)
    dist = cosi(v1, v2)
    if (dist.item() > 1):  # trimming known bug >1 result
        return 1.0
    if (dist.item() < -1):
        return -1.0
    return dist.item()


def get_sorted_LV_by_abs_L(symA):

    L, V = torch.linalg.eigh(symA)

    sorted_absL, indices = torch.sort(abs(L), dim=0, descending=True, out=None)

    sortedV = V[:, indices]
    sortedL = L[indices]
    return sortedL, sortedV


def get_sorted_LV(symA):

    L, V = torch.linalg.eigh(symA)

    sorted_L, indices = torch.sort(L, dim=0, descending=True, out=None)

    sortedV = V[:, indices]
    sortedL = L[indices]
    return sortedL, sortedV


def get_model_size(model_params):
    size = 0
    for param in model_params:
        flat_param = torch.flatten(param)
        size = size + flat_param.shape[0]
    return size


def reshape_hessian(H, model_params):
    model_size = get_model_size(model_params)
    HM = torch.zeros([model_size, model_size])
    acc_i = 0
    acc_j = 0
    for i, i_key in enumerate(H.keys(), start=0):
        for j, j_key in enumerate(H.keys()):
            i_param_size = torch.flatten(model_params[i]).shape[0]
            j_param_size = torch.flatten(model_params[j]).shape[0]
            Hij = H[i_key][j_key]
            reshaped_Hij = torch.reshape(Hij, [i_param_size, j_param_size])
            HM[acc_i:acc_i+i_param_size, acc_j:acc_j+j_param_size] = reshaped_Hij

            acc_j = acc_j + j_param_size
        acc_j = 0
        acc_i = acc_i + i_param_size
    return HM




def get_grads(model, X, y, loss_function, optimizer):
    total_loss = 0.0
    train_dataset = TensorDataset(X, y)
    data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # initialization of the gradients
    total_grads = torch.zeros([get_model_size(model.parameters()),1]).to(DEVICE)
    for i, (iX, iy) in enumerate(data_loader):
        optimizer.zero_grad() #URI- if not practicing  optimizer.step() do I need to nullify grads here?


        # forward + backward + optimize
        probs = model(iX)
        # y_max_indices = torch.max(iy, 1)[1]
        # loss = loss_function(probs, y_max_indices)
        loss = loss_function(probs, iy)
        loss.backward()
        iGrads = get_model_grads_as_tensor(model)
        total_grads = total_grads + iGrads
        #optimizer.step() #no parameter update
    # pred_y = model(X)
    # loss = loss_function(pred_y, y) # <= compute the loss function
    # loss.backward()
    grads = total_grads/i
    return grads


def reset_model(model, parameters, optimizer):
    set_model_parameters(model, parameters)
    optimizer.zero_grad()


def get_model_grads_at_set_parameters(model, parameters, X, y, loss_function, optimizer):
    optimizer.zero_grad()
    set_model_parameters(model, parameters)
    grads = get_grads(model, X, y, loss_function, optimizer)
    return grads


def get_model_grads_as_tensor(model):
    grads = []
    for param in model.parameters():
        grads.append(param.grad.view(-1))
    grads = torch.cat(grads)
    grads = grads.clone().reshape([grads.shape[0], 1])  # detach
    return grads


def model_parametres_to_tensor(model):
    model_params_tensor = []
    for param in model.parameters():
        model_params_tensor.append(param.data.view(-1).clone())
    model_params_tensor = torch.cat(model_params_tensor)
    model_params_tensor = model_params_tensor.reshape([model_params_tensor.shape[0],1])
    return model_params_tensor


def get_epsilon(model, v):
    # N. Andrei, “Accelerated conjugate gradient algorithm with finite difference hessian/vector product approximation for unconstrained optimization,”
    # epsilon = 2*sqrt(machine precision)*(model's norm + 1)/(v's norm)
    model_params = model_parametres_to_tensor(model)
    model_size = torch.matmul(model_params.t(), model_params)**0.5
    v_size = torch.linalg.vector_norm(v,2)
    epsilon = (2*(EM**0.5)*(1+model_size)/v_size).item()
    # epsilon *= 100
    return epsilon




def get_V0L0_VnLn_Lanczos(model, loss_function, optimizer, X, y, lambda_to_sub=0, preconditioner1 = None, preconditioner2 = None):
    V0, L0, dist_converged_0, converge_stats0, i_max0, i0 = get_L0V0_Lanczos(
        model, loss_function, optimizer, X, y, lambda_to_sub=None, preconditioner = preconditioner1)
    # converge_stats0["type"] = "0"
    Vn, second_iter_Ln, dist_converged_n, converge_statsn,  i_maxN, iN= get_L0V0_Lanczos(
        model, loss_function, optimizer, X, y, lambda_to_sub=L0, preconditioner = preconditioner1)
    # converge_statsn["type"] = "n"
    Ln = L0.abs() - second_iter_Ln
    return V0, L0, dist_converged_0, converge_stats0, Vn, Ln, dist_converged_n, converge_statsn, i0, iN, i_max0, i_maxN, second_iter_Ln


def get_L0V0_Lanczos(model, loss_function, optimizer, X, y, lambda_to_sub=0, preconditioner = None):
    converge_stats = {}
    model_init_params = model_parametres_to_tensor(model).clone()
    i_model_params = model_init_params.clone()
    i_v = torch.ones([get_model_size(tuple(model.parameters())),1]).to(DEVICE)
    i_v = i_v/torch.linalg.vector_norm(i_v, ord=2)
    i_v_prev = i_v.clone()
    i_epsilon = get_epsilon(model, i_v)
    i_w_tag = get_model_v_fd(model, loss_function,
                             optimizer, X, y, i_v, lambda_to_sub, preconditioner, i_epsilon)
    i_a = i_w_tag*i_v
    i_w = i_w_tag - i_a*i_v
    v_max = i_v.clone()
    # l_min = (i_w_tag/i_v).nanmean() # changed to r(x)
    l_max = torch.matmul(i_w_tag, i_v.t()) / torch.matmul(i_v.t(), i_v)
    cos_dist_at_convergence = 1.0
    i = 2
    i_max = i
    while (i < LANCZOS_DIM) :
        print("LANCZOS loop")
        # and (abs(cos_dist_at_convergence) > COSIN_DIST):
        beta_i = torch.matmul(i_w.t(), i_w)**0.5
        if beta_i > 0:
            i_v = i_w/beta_i
            i_epsilon = get_epsilon(model, i_v)
            i_w_t = get_model_v_fd(model, loss_function,
                                   optimizer, X, y, i_v, lambda_to_sub, preconditioner, i_epsilon)
            i_a = i_w_t*i_v
            i_w = i_w_t - i_a*i_v - beta_i*i_v_prev
            i_v_prev = i_v.clone()
            # i_l = (i_w_t/i_v).nanmean() # changed to r(x)
            i_l = torch.matmul(i_w_t, i_v.t()) / torch.matmul(i_v.t(), i_v)
            if (i_l > l_max):
                cos_dist_at_convergence = get_dist(i_v, v_max)
                # print("lanczos improve at iteration: {} dist: {} new lambda: {}".format(
                    # i, cos_dist_at_convergence, l_max))
                v_max = i_v.clone()
                l_max = i_l
                i_max = i

        else:
            print("ERROR beta_i")
        i += 1

    return v_max, l_max, cos_dist_at_convergence, converge_stats, i_max, i

def get_min_halt_condition(Av, l, v, eps):
    term1 = torch.linalg.norm(Av-l*v)
    term2 = eps**2
    return term1+term2

#get_model_loss(model, i_v, step_size, X, y, loss_function, optimizer)
def get_loss_point_ratio(model, v, X, y, loss_function, optimizer, a, b, delta_tolerance = 0.0, n = 5):
    tau = (3.0-5.0**0.5)/2.0
    delta = b-a
    p = a + tau*delta
    q = a + (1-tau)*delta
    loss_a = get_model_loss(model, v, a, X, y, loss_function, optimizer, False)
    loss_p = get_model_loss(model, v, p, X, y, loss_function, optimizer, False)
    loss_q = get_model_loss(model, v, q, X, y, loss_function, optimizer, False)
    loss_b = get_model_loss(model, v, b, X, y, loss_function, optimizer, False)
    # print("losses a = {}, p = {}, q = {} , b = {}".format(loss_a, loss_p, loss_q, loss_b))
    for i in range(n):
        #check convexity (needed at all)
        if (delta < delta_tolerance):
            break
        if (loss_p < loss_q):
            b = q
            loss_b = loss_q
            q = p
            loss_q = loss_p
            delta = b - a
            p = a + tau*delta 
            loss_p = get_model_loss(model, v, p, X, y, loss_function, optimizer, False)
            # print("losses a = {}, p = {}, q = {} , b = {} at iteration i={}".format(loss_a, loss_p, loss_q, loss_b, i))
        else:
            a = p
            loss_a = loss_p
            p = q
            loss_p = loss_q
            delta = b - a
            q = a + (1-tau)*delta
            loss_q = get_model_loss(model, v, q, X, y, loss_function, optimizer, False)
            # print("losses a = {}, p = {}, q = {} , b = {} at iteration i={}".format(loss_a, loss_p, loss_q, loss_b, i))

    opt_point = (b+a)/2.0
    loss_opt_point = get_model_loss(model, v, opt_point, X, y, loss_function, optimizer, False)
    # print("losses at opt point = {} is = {}".format(opt_point, loss_opt_point))

    return opt_point, loss_opt_point

def get_lambda(v, Av):
    return ((v.t() @ Av) / (v.t() @ v)).item()

def get_L0V0_LnVn_LOBPCG(model, loss_function, optimizer, X, y, step_size, adjust_step_size = False, use_preconditioner = False,ref_v=torch.zeros(1), ref_l = 0.0, ref_H = torch.zeros(1), filepath = ""):
    model_base = ""
    if filepath != "":
        CIFAR10_RESNET18.load_data()
        CIFAR10_RESNET18.load_model(filepath)
    
        model_base = CIFAR10_RESNET18.model
    run_stats = pd.DataFrame()#columns=['iteration', 'cosin vec', 'lambda', 'lambda diff', 'loss', 'loss_diff', 'halt condition'])
    
    t0 = time.time()
    model_init_params = model_parametres_to_tensor(model).clone()
    i_model_params = model_init_params.clone()
    model_size = get_model_size(tuple(model.parameters()))
    
    i_grad_at_start_point = get_model_grads_at_set_parameters(model, i_model_params , X, y, loss_function, optimizer)
    
    i_v = torch.rand([model_size,1]).to(DEVICE)
    i_v = i_v/torch.linalg.vector_norm(i_v, ord=2)
    i_v_prev = i_v.clone()
    i_v_prev_min = i_v.clone()
    i_v_prev_max = i_v.clone()
    
    t1 = time.time()
    time_elapsed = t1-t0
    i_epsilon = get_epsilon(model, i_v)
    # smooth_df = get_model_v_fd_smoothness_stats(model, loss_function, optimizer, X, y, i_v, i_epsilon, 0.001, i_epsilon+1.0, 0.01)
    # smooth_df.to_excel(r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\shift_3_smooth_df_iter="+str(0)+"_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+".xlsx")

    i_Av = get_model_v_fd(model, loss_function, optimizer, X, y, i_v, None, None, i_epsilon)
    i_Av_min = i_Av.clone()
    i_Av_max = i_Av.clone()
    i_l = get_lambda(i_v, i_Av)
    print("i_l b4 RR: ", i_l)
    t2 = time.time()
    time_elapsed = t2-t1
    i_w = i_Av - (i_v.t() @ i_Av) * i_v  # A*v -v'Av*v
    # i_w = i_w/torch.linalg.vector_norm(i_w, ord=2)
    i_w_min = i_w.clone()
    i_w_max = i_w.clone()
    V = torch.cat((i_v, i_w), dim = 1)
    t3 = time.time()
    i_v_min, i_l_min, i_r_min, approx_rr_i_l_min = get_LV_LOBPCG_Rayleigh_Ritz(model, loss_function, optimizer, X, y, V, preconditioner = None, run_type="MIN", ref_v = ref_v, ref_l = ref_l, ref_H = ref_H)
    i_v_max, i_l_max, i_r_max, approx_rr_i_l_max = get_LV_LOBPCG_Rayleigh_Ritz(model, loss_function, optimizer, X, y, V, preconditioner = None, run_type="MAX", ref_v = ref_v, ref_l = ref_l, ref_H = ref_H)
    t4 = time.time()
    print("i_l b4 AFTER: ", i_l)
    time_elapsed = t4-t3
    print('get_LV_LOBPCG_Rayleigh_Ritz two ends in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    i_v_min = i_v_min/torch.linalg.vector_norm(i_v_min, ord=2)
    i_v_max = i_v_min/torch.linalg.vector_norm(i_v_max, ord=2)
    i = 0
    cosin_dist_min = get_dist(i_v_min, i_v_prev_min)
    cosin_dist_max = get_dist(i_v_max, i_v_prev_max)
    cosin_dist_min_max = get_dist(i_v_min, i_v_max)
    epsilon = get_epsilon(model, i_v)
    i_halt_condition = get_min_halt_condition(i_Av, i_l, i_v, epsilon)
    i_delta = torch.linalg.norm(i_Av - i_v_min*i_l_min)
    i_l_prev = 0.0
    # LOBPCG_loss_prev_min= = 0.0

    # Halting when (i_l)**2 > delta**2
    while (i < MAX_ITER_NUM):# and ((i_l_min)**2 <= N_HALT_MULTIPLIER*(i_delta+0.5*epsilon*i_l_max)**2): #and ((i_l)**2 < (N_HALT_MULTIPLIER*i_halt_condition)**2) :
    #and (i_l > 0)
    #(i < MAX_ITER_NUM) and (abs(abs(cosin_dist)-1) > COSIN_DIST):
    #(abs(abs(cosin_dist) - abs(cosin_dist_prev)) > COSIN_DIST) and 
        print("LOBPCG loop at iteration: ", i)
        t4 = time.time()
        Vmin = torch.cat((i_v_min, i_w_min, i_v_prev_min), dim = 1)
        Vmax = torch.cat((i_v_max, i_w_max, i_v_prev_max), dim = 1)
        i_v_prev_min = i_v_min.clone()
        i_v_prev_max = i_v_max.clone()
        
        i_l_prev_min = i_l_min
        i_l_prev_max = i_l_max
        t5 = time.time()
        # time_elapsed = t5-t4
        # print('LOBLOOP part I {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print("i_l b4 RR: {} {}".format(i_l_min, i_l_max))
        
        i_v_min, i_l_min, i_r_min, approx_rr_i_l_min = get_LV_LOBPCG_Rayleigh_Ritz(model, loss_function, optimizer, X, y, Vmin, preconditioner = None, run_type="MIN", ref_v = ref_v, ref_l = ref_l, ref_H = ref_H)
        i_v_max, i_l_max, i_r_max, approx_rr_i_l_max = get_LV_LOBPCG_Rayleigh_Ritz(model, loss_function, optimizer, X, y, Vmax, preconditioner = None, run_type="MAX", ref_v = ref_v, ref_l = ref_l, ref_H = ref_H)
        # smooth_df = get_model_v_fd_smoothness_stats(model, loss_function, optimizer, X, y, i_v, i_epsilon, 0.001, i_epsilon+1.0, 0.01)
        # smooth_df.to_excel(r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\shift_3_smooth_df_iter="+str(i+1)+"_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+".xlsx")
        print("i_l after RR: {} {}".format(i_l_min, i_l_max))

        t6 = time.time()
        # time_elapsed = t6-t5
        # print('LOBLOOP part II {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        i_v_min = i_v_min/torch.linalg.vector_norm(i_v_min, ord=2)
        i_v_max = i_v_max/torch.linalg.vector_norm(i_v_max, ord=2)
        
        i_epsilon = get_epsilon(model, i_v_min)
        i_Av_min = get_model_v_fd(model, loss_function, optimizer, X, y, i_v_min, None, None, i_epsilon)
        i_Av_max = get_model_v_fd(model, loss_function, optimizer, X, y, i_v_max, None, None, i_epsilon)
        i_r_min = i_Av_min - torch.matmul(i_v_min.t(), i_Av_min) * i_v_min  # A*v -v'Av*v
        i_r_max = i_Av_max - torch.matmul(i_v_max.t(), i_Av_max) * i_v_max  # A*v -v'Av*v
        i_delta = torch.linalg.norm(i_Av_min - i_v_min*i_l_min)
        
        if (use_preconditioner is False):
            i_w_min = i_r_min
            i_w_max = i_r_max
        else:
            precond_min = precond_grad_sqr(model, loss_function, optimizer, X, y)*i_r_min
            # precond= precond_bi_diagonal(i_r, 0.95)
            i_w_min = precond_min#precond_bi_diagonal*i_r
            
            precond_max = precond_grad_sqr(model, loss_function, optimizer, X, y)*i_r_max
            # precond= precond_bi_diagonal(i_r, 0.95)
            i_w_max = precond_max#precond_bi_diagonal*i_r
        # i_w_min = i_w_min/torch.linalg.vector_norm(i_w_min, ord=2)
        # i_w_max = i_w_max/torch.linalg.vector_norm(i_w_max, ord=2)
        t7 = time.time()
        # time_elapsed = t7-t6
        # print('LOBLOOP part III {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
       
        cosin_dist_prev_min = cosin_dist_min
        cosin_dist_min = get_dist(i_v_min, i_v_prev_min)
        
        cosin_dist_prev_max = cosin_dist_max
        cosin_dist_max = get_dist(i_v_max, i_v_prev_max)
        
        cosin_dist_min_max = get_dist(i_v_min, i_v_max)
        
        i_l_min = get_lambda(i_v_min, i_Av_min)
        i_l_max = get_lambda(i_v_max, i_Av_max)
                
        i_epsilon = get_epsilon(model, i_v_min)
        print("epsilon:", epsilon)
        
        i_halt_condition = get_min_halt_condition(i_Av_min, i_l_min, i_v_min, i_epsilon)
        print("tmp_halt_condition: {} is greater than lambda : {} in abs  {}".format( i_halt_condition, i_l_min, (i_l_min**2 < i_halt_condition**2) ))
        LOBPCG_loss_plus_min = get_model_loss(model, i_v_min, step_size, X, y, loss_function, optimizer, False)
        LOBPCG_loss_no_step = get_model_loss(model, (0.0)*i_v_min, step_size, X, y, loss_function, optimizer, False)
        
        if (adjust_step_size):
            opt_point_LOBPCG_plus, opt_point_LOBPCG_loss_plus = get_loss_point_ratio(model, i_v_min, X, y, loss_function, optimizer, 1.0*step_size, 2.0*step_size, 0.0, 5)
            # opt_point_LOBPCG_minus, opt_point_LOBPCG_loss_minus = get_loss_point_ratio(model, (-1)*i_v_min, X, y, loss_function, optimizer, 0.0, step_size*10, 0.0, 5)
        else:
            opt_point_LOBPCG_plus = step_size
            # opt_point_LOBPCG_minus = (-1)*step_size
            opt_point_LOBPCG_loss_plus = LOBPCG_loss_plus_min
            # opt_point_LOBPCG_loss_minus = LOBPCG_loss_minus_min
            
        i_new_halt_condition = (i_delta.item()+0.5*i_epsilon*i_l_max)
        print("i_new_halt_condition: ", i_new_halt_condition)
        print("i_l_min:", i_l_min)
        t8 = time.time()
        # time_elapsed = t8-t7
        # print('LOBLOOP part IV {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        time_elapsed = t8-t4
        print('LOBLOOP iteration running time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        # print("i_r: ",  i_r)
        i_grad_at_opt_point = 0.0
        # if opt_point_LOBPCG_loss_plus < opt_point_LOBPCG_loss_minus:
        i_grad_at_opt_point = get_model_grads_at_set_parameters(model, i_model_params + i_v_min*opt_point_LOBPCG_plus, X, y, loss_function, optimizer)
        # else:
        #     i_grad_at_opt_point = get_model_grads_at_set_parameters(model, i_model_params + (-1)*i_v_min*opt_point_LOBPCG_minus, X, y, loss_function, optimizer)
        
        new_row = {'iteration' : i, 
                   'cosin vec min': cosin_dist_min, 
                   'cosin vec max': cosin_dist_max, 
                   'cosin min_max' : cosin_dist_min_max,
                   'lambda min': i_l_min, 'lambda max': i_l_max, 
                    'loss+': LOBPCG_loss_plus_min.item(), 
                    'loss 0.0 step': LOBPCG_loss_no_step.item(),
                    # 'loss-': LOBPCG_loss_minus_min.item(), 
                   'delta': i_delta.item(), 
                   'epsion': i_epsilon,
                   'halt condition (delta - eps^2)': i_halt_condition.item(), 
                   'new halt condition (delta + 0.5epsLmax)': i_new_halt_condition,
                    'grad at init' : torch.linalg.vector_norm(i_grad_at_start_point, ord=2).item(),
                    'grad at step' : torch.linalg.vector_norm(i_grad_at_opt_point, ord=2).item(),
                   'step size' : step_size,
                    '+ opt step size': opt_point_LOBPCG_plus, 
                    # '- opt step size': opt_point_LOBPCG_minus,
                    'loss+ at opt step size': opt_point_LOBPCG_loss_plus.item(),
                   # 'loss- at opt step size': opt_point_LOBPCG_loss_minus.item(), 
                   # 'approx_rr_i_l_min': approx_rr_i_l_min
                   }
        run_stats = pd.concat([run_stats, pd.DataFrame([new_row])], ignore_index = True)
        
        with pd.ExcelWriter(OUT_PATH_SUMMARY + "\\iter=" + str(i) + "two_ways_with_step_grad_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+ "_summary_res.xlsx") as writer:
            run_stats.to_excel(writer, sheet_name='run_stats')
        if filepath != "":
            compare_models(model, model_base)
        i += 1
        # i_v_min_step_size = opt_point_LOBPCG_plus
        # if opt_point_LOBPCG_loss_minus < opt_point_LOBPCG_loss_plus:
        #     i_v_min = (-1)*i_v_min
        #     i_v_min_step_size = opt_point_LOBPCG_minus
    return i_v_min, i_l_min, i, run_stats#, i_v_min_step_size


def get_LV_LOBPCG_Rayleigh_Ritz(model, loss_function, optimizer, X, y, V, preconditioner = None, run_type = "MIN", ref_v = torch.zeros(1), ref_l = 0.0,  ref_H = torch.zeros(1)):
    t0 = time.time()
    Q, R = torch.linalg.qr(V)
    t1 = time.time()
    # time_elapsed = t1-t0
    # print('torch.linalg.qr in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    model_size = get_model_size(tuple(model.parameters()))
    AV = torch.tensor([]).to(DEVICE)
    Vorthnorm = torch.tensor([]).to(DEVICE)
    i = 0
    for vi in Q.t():
        if abs(R[i][i]) > EM:
            vi = vi.reshape([vi.shape[0],1])
            t2 = time.time()
            vi = vi / torch.linalg.vector_norm(vi, 2)
            i_epsilon = get_epsilon(model, vi)
            Avi = get_model_v_fd(model, loss_function, optimizer, X, y, vi, None, preconditioner, i_epsilon)
            t3 = time.time()
            time_elapsed = t3-t2
            # print('\t- inner loop get_model_v_fd in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            AV = torch.cat((AV, Avi), dim = 1)
            Vorthnorm = torch.cat((Vorthnorm, vi), dim = 1)
            i = + 1
            t4 = time.time()
            time_elapsed = t4-t2
            # print('\t- inner loop in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    M = Vorthnorm.t() @ AV
    Ls, Vs = torch.linalg.eig(M)

    if Vs.sum().sum().imag.item() > 0:  # complex eig vecs
        raise("complex eig vecs in get_L0V0_LOBPCG_Rayleigh_Ritz")
    Ls = Ls.real
    Vs = Vs.real
    # if run_type == "MIN":
    #     (i_Ls, i_Ls_index) = torch.min(Ls,0)
    # else:
    #     (i_Ls, i_Ls_index) = torch.max(Ls,0)
    
    NV = Vorthnorm @ Vs
    t_Ls_index = 0
    tV = (NV.t()[t_Ls_index,:]).reshape([model_size,1])
    tV = tV/torch.linalg.vector_norm(tV,2)
    epsilon = get_epsilon(model, tV)
    A_tV = get_model_v_fd(model, loss_function, optimizer, X, y, tV, None, preconditioner, epsilon)
    t_lambda_n = get_lambda(tV, A_tV)
    
    
    for i_Ls_index in range(1,NV.shape[1]):
        
        i_Vn = (NV.t()[i_Ls_index,:]).reshape([model_size,1])
        i_Vn = i_Vn/torch.linalg.vector_norm(i_Vn,2)
        i_epsilon = get_epsilon(model, i_Vn)
        Ai_Vn = get_model_v_fd(model, loss_function, optimizer, X, y, i_Vn, None, preconditioner, i_epsilon)
        i_lambda_n = get_lambda(i_Vn, Ai_Vn)
        if run_type == "MIN":
            if i_lambda_n < t_lambda_n:
                t_lambda_n = i_lambda_n
                tV = i_Vn
                t_Ls_index = i_Ls_index
        else:
            if i_lambda_n > t_lambda_n:
                t_lambda_n = i_lambda_n
                tV = i_Vn
                t_Ls_index = i_Ls_index


    tn = time.time()
    time_elapsed = tn-t0
    # vector_choosing = min_i_by_M != min_i
    # print('\t- total time of get_LnVn_LOBPCG_Rayleigh_Ritz in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return tV, t_lambda_n, R[-1][-1], t_Ls_index





def get_LnVn_LOBPCG(model, loss_function, optimizer, X, y, step_size, use_preconditioner = False):
    run_stats = pd.DataFrame()#columns=['iteration', 'cosin vec', 'lambda', 'lambda diff', 'loss', 'loss_diff', 'halt condition'])
    
    t0 = time.time()
    model_init_params = model_parametres_to_tensor(model).clone()
    i_model_params = model_init_params.clone()
    model_size = get_model_size(tuple(model.parameters()))
    i_v = torch.rand([model_size,1]).to(DEVICE)
    i_v = i_v/torch.linalg.vector_norm(i_v, ord=2)
    i_v_prev = i_v.clone()
    
    t1 = time.time()
    time_elapsed = t1-t0
    i_epsilon = get_epsilon(model, i_v)
    i_Av = get_model_v_fd(model, loss_function, optimizer, X, y, i_v, None, None, i_epsilon)
    i_l = get_lambda(i_v, i_Av)
    print("i_l b4 RR: ", i_l)
    t2 = time.time()
    time_elapsed = t2-t1
    i_w = i_Av - (i_v.t() @ i_Av) * i_v  # A*v -v'Av*v
    i_w = i_w/torch.linalg.vector_norm(i_w, ord=2)
    V = torch.cat((i_v, i_w), dim = 1)
    t3 = time.time()
    i_v, i_l, i_r, approx_rr_i_l = get_LnVn_LOBPCG_Rayleigh_Ritz(model, loss_function, optimizer, X, y, V, preconditioner = None)
    t4 = time.time()
    print("i_l b4 AFTER: ", i_l)
    time_elapsed = t4-t3
    print('get_LnVn_LOBPCG_Rayleigh_Ritz in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    i_v = i_v/torch.linalg.vector_norm(i_v, ord=2)
    i = 0
    cosin_dist = get_dist(i_v, i_v_prev)
    cosin_dist_prev = 0

    epsilon = get_epsilon(model, i_v)
    i_halt_condition = get_min_halt_condition(i_Av, i_l, i_v, epsilon)
    i_delta = torch.linalg.norm(i_Av - i_v*i_l)
    i_l_prev = 0.0
    LOBPCG_loss_prev = 0.0

    # Halting when (i_l)**2 > delta**2
    while (i < MAX_ITER_NUM) : #and ((i_l)**2 < (N_HALT_MULTIPLIER*i_halt_condition)**2) :
    #and (i_l > 0)
    #(i < MAX_ITER_NUM) and (abs(abs(cosin_dist)-1) > COSIN_DIST):
    #(abs(abs(cosin_dist) - abs(cosin_dist_prev)) > COSIN_DIST) and 
        print("LOBPCG loop at iteration: ", i)
        t4 = time.time()
        V = torch.cat((i_v, i_w, i_v_prev), dim = 1)
        i_v_prev = i_v.clone()
        i_l_prev = i_l
        t5 = time.time()
        # time_elapsed = t5-t4
        # print('LOBLOOP part I {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print("i_l b4 RR: ", i_l)

        i_v, i_l, i_r, approx_rr_i_l = get_LnVn_LOBPCG_Rayleigh_Ritz(model, loss_function, optimizer, X, y, V, preconditioner = None)
        print("i_l AFTER RR: ", i_l)

        t6 = time.time()
        # time_elapsed = t6-t5
        # print('LOBLOOP part II {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        i_v = i_v/torch.linalg.vector_norm(i_v, ord=2)
        i_Av = get_model_v_fd(model, loss_function, optimizer, X, y, i_v, None, None)
        i_r = i_Av - torch.matmul(i_v.t(), i_Av) * i_v  # A*v -v'Av*v
        i_delta = torch.linalg.norm(i_Av - i_v*i_l)
        
        if (use_preconditioner is False):
            i_w = i_r
        else:
            precond = precond_grad_sqr(model, loss_function, optimizer, X, y)*i_r
            # precond= precond_bi_diagonal(i_r, 0.95)
            i_w = precond#precond_bi_diagonal*i_r
            
        
        i_w = i_w/torch.linalg.vector_norm(i_w, ord=2)
        t7 = time.time()
        # time_elapsed = t7-t6
        # print('LOBLOOP part III {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
       
        cosin_dist_prev = cosin_dist
        cosin_dist = get_dist(i_v, i_v_prev)
        i_l = get_lambda(i_v, i_Av)
        epsilon = get_epsilon(model, i_v)
        i_halt_condition = get_min_halt_condition(i_Av, i_l, i_v, epsilon)
        print("tmp_halt_condition: {} is greater than lambda : {} in abs  {}".format( i_halt_condition, i_l, (i_l**2 < i_halt_condition**2) ))
        LOBPCG_loss_plus = get_model_loss(model, i_v, step_size, X, y, loss_function, optimizer, False)
        LOBPCG_loss_minus = get_model_loss(model, (-1)*i_v, step_size, X, y, loss_function, optimizer, False)
        opt_point_LOBPCG_plus, opt_point_LOBPCG_loss_plus = get_loss_point_ratio(model, i_v, X, y, loss_function, optimizer, 0.0, 1e-2, 0.0, 3)
        opt_point_LOBPCG_minus, opt_point_LOBPCG_loss_minus = get_loss_point_ratio(model, (-1)*i_v, X, y, loss_function, optimizer, 0.0, 1e-2, 0.0, 3)

        
  
        t8 = time.time()
        # time_elapsed = t8-t7
        # print('LOBLOOP part IV {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        time_elapsed = t8-t4
        print('LOBLOOP iteration running time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        # print("i_r: ",  i_r)
        
        new_row = {'iteration' : i, 'cosin vec': cosin_dist, 'lambda': i_l, 'loss+': LOBPCG_loss_plus.item(), 'loss-': LOBPCG_loss_minus.item(), 'delta': i_delta.item(), 
                   'halt condition': i_halt_condition.item(), 
                   '+ opt step size': opt_point_LOBPCG_plus, '- opt step size': opt_point_LOBPCG_minus,
                   'loss+ at opt step size': opt_point_LOBPCG_loss_plus.item(),
                   'loss- at opt step size': opt_point_LOBPCG_loss_minus.item(), 
                   'approx_rr_i_l': approx_rr_i_l}
        run_stats = pd.concat([run_stats, pd.DataFrame([new_row])], ignore_index = True)
        
        with pd.ExcelWriter(OUT_PATH_SUMMARY + "\\iter=" + str(i) + "sign_rand_1_start_offs_points_RR_100_iterartions_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+ "_summary_res.xlsx") as writer:
            run_stats.to_excel(writer, sheet_name='run_stats')
            
        
        i += 1
    return i_v, i_l, i, run_stats


def get_LnVn_LOBPCG_Rayleigh_Ritz(model, loss_function, optimizer, X, y, V, preconditioner = None):
    t0 = time.time()
    Q, R = torch.linalg.qr(V)
    t1 = time.time()
    # time_elapsed = t1-t0
    # print('torch.linalg.qr in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    model_size = get_model_size(tuple(model.parameters()))
    AV = torch.tensor([]).to(DEVICE)
    Vorthnorm = torch.tensor([]).to(DEVICE)
    i = 0
    for vi in Q.t():
        if abs(R[i][i]) > EM:
            vi = vi.reshape([vi.shape[0],1])
            t2 = time.time()
            vi = vi / (vi.t() @ vi)
            i_epsilon = get_epsilon(model, vi)
            Avi = get_model_v_fd(model, loss_function, optimizer, X, y, vi, None, preconditioner, i_epsilon)
            t3 = time.time()
            time_elapsed = t3-t2
            # print('\t- inner loop get_model_v_fd in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            AV = torch.cat((AV, Avi), dim = 1)
            Vorthnorm = torch.cat((Vorthnorm, vi), dim = 1)
            i = + 1
            t4 = time.time()
            time_elapsed = t4-t2
            # print('\t- inner loop in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    M = Vorthnorm.t() @ AV
    Ls, Vs = torch.linalg.eig(M)

    if Vs.sum().sum().imag.item() > 0:  # complex eig vecs
        raise("complex eig vecs in get_L0V0_LOBPCG_Rayleigh_Ritz")
    Ls = Ls.real
    Vs = Vs.real
    
    (min_Ls, mim_Ls_index) = torch.min(Ls,0)
    ############# Ls = tensor([2.2171e-02, 2.4142e+02], device='cuda:0')
    ########### V 
    #######tensor(0.1468, device='cuda:0') - Random v
    #######tensor(241.2913, device='cuda:0') - Residual of v
    
    # i_l_in RR iterations are:  tensor(0.4978, device='cuda:0')
    # i_l_in RR iterations are:  tensor(240.8912, device='cuda:0')
    
    NV = Vorthnorm @ Vs
    ########################ADD NORM HERE!!!!!!!!!!!!!!!!
    min_Vn = (NV.t()[mim_Ls_index,:]).reshape([model_size,1])
    i_epsilon = get_epsilon(model, min_Vn)
    Amin_Vn = get_model_v_fd(model, loss_function, optimizer, X, y, min_Vn, None, preconditioner, i_epsilon)
    min_lambda_n = get_lambda(min_Vn, Amin_Vn)
    # for i in range(1, len(NV[0,:])):
    #     v_i = NV.t()[i,:]
    #     v_i = v_i.reshape([v_i.shape[0],1])
    #     Avi = get_model_v_fd(model, loss_function, optimizer, X, y, v_i, None, preconditioner)
    #     lambda_i = get_lambda(v_i, Avi)
    #     print("i_l_in RR iterations are: ", lambda_i)
    #     print("i_l approx by RR M[i][i]:", Ls[i])
    #     if (lambda_i < min_lambda_n):
    #         min_i = i
    #         min_lambda_n = lambda_i
    #         min_Vn = v_i
    # min_i_by_M = 0
    # min_Ls = Ls[0]
    # for i in range(1, len(Ls)):
    #     if Ls[i] < min_Ls:
    #         min_i_by_M = i
    #         min_Ls = Ls[i]
    # if (min_i_by_M != min_i):
    #     print("!!!!!!!!!!!!!!!!!!!!!!!Not Same vector choosing in RR")
    #sorted_L, indices = torch.sort(Ls, dim=0, descending=True, out=None)

    # return argmin [-1]

    #Vn = NV[indices[-1]]
    #Ln = Ls[indices[-1]]
    tn = time.time()
    time_elapsed = tn-t0
    # vector_choosing = min_i_by_M != min_i
    # print('\t- total time of get_LnVn_LOBPCG_Rayleigh_Ritz in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return min_Vn, min_lambda_n, R[-1][-1], min_Ls 


def get_model_v_fd(model, loss_function, optimizer, X, y, v, lambda_to_substract=None, preconditioner = None, epsilon = 0.0):
    model_init_params = model_parametres_to_tensor(model).clone()

    delta = v*epsilon
    delta_size = torch.linalg.vector_norm(delta, ord = 2)
    plus_eps_i_v = model_init_params + delta
    minus_eps_i_v = model_init_params - delta

    grad_plus_eps_niv = get_model_grads_at_set_parameters(
        model, plus_eps_i_v, X, y, loss_function, optimizer)
    reset_model(model, model_init_params, optimizer)  # back to init
    grad_minus_eps_niv = get_model_grads_at_set_parameters(
        model, minus_eps_i_v, X, y, loss_function, optimizer)
    reset_model(model, model_init_params, optimizer)  # back to init
    nv = (grad_plus_eps_niv - grad_minus_eps_niv)/(2*delta_size)
    
    if lambda_to_substract is not None:
        nv = lambda_to_substract*v - nv

    if preconditioner is not None:
        nv = nv * preconditioner
    return nv





def get_model_v_fd_smoothness_stats(model, loss_function, optimizer, X, y, v, epsilon, epsilon_start, epsilon_end, epsilon_step):
    
    smoothness_stats_df = pd.DataFrame()
    
    model_init_params = model_parametres_to_tensor(model).clone()
    
    model_v_epsilon_ref =  get_model_v_fd(model, loss_function, optimizer, X, y, v, lambda_to_substract=None, preconditioner = None, epsilon = epsilon)
    
    model_v_epsilon_prev =  get_model_v_fd(model, loss_function, optimizer, X, y, v, lambda_to_substract=None, preconditioner = None, epsilon = epsilon_start)
    i_epsilon_prev = epsilon_start
    i_epsilon = epsilon_start + epsilon_step
    while (i_epsilon<=epsilon_end):
        model_v_epsilon_cur =  get_model_v_fd(model, loss_function, optimizer, X, y, v, lambda_to_substract=None, preconditioner = None, epsilon = i_epsilon)
        row = {
            
            "epsilon" : epsilon,
            "prev_epsilon": i_epsilon_prev,
            "i_epsilon" : i_epsilon,
            "cosin i with prev": get_dist(model_v_epsilon_prev, model_v_epsilon_cur),
            "cosin i with epsilon": get_dist(model_v_epsilon_ref, model_v_epsilon_cur),

        }
        model_v_epsilon_prev = model_v_epsilon_cur.clone()
        smoothness_stats_df = pd.concat([smoothness_stats_df, pd.DataFrame([row])])
    
        i_epsilon += epsilon_step
        i_epsilon_prev = i_epsilon
    
    return smoothness_stats_df









'''
returns the biggest in abs value lambda with it's eig vector
'''
def get_L0V0_finite_differencing(model, loss_function, optimizer, X, y, max_iter_num, is_first=True, lambda_to_sub=0, preconditioner = None):

    df_stats = pd.DataFrame(columns=['iteration', 'cosin vec', 'lambda diff'])

    i_v = torch.ones([get_model_size(tuple(model.parameters())),1]).to(DEVICE)
    i_v_size = torch.matmul(i_v.t(), i_v)**0.5
    i_v = i_v / i_v_size

    prev_cosin_dist = -1
    cosin_dist_diff = 1/MAX_DIST_CONVERGENCE
    cosin_dist = 1/MAX_DIST_CONVERGENCE
    i_l = 1/MAX_DIST_CONVERGENCE

   
    i_v_norm = 1
    i_l_prev = -1
    i = 0
    # halting at the sooner of (max_iteration,no change between two consecutive iterations)
    while ((i < MAX_ITER_NUM) and (abs(cosin_dist_diff) > COSIN_DIST)):
        print("FD loop")
        prev_cosin_dist = cosin_dist
        i_v_prev = i_v.clone()
        i_l_prev = i_l
        i_epsilon = get_epsilon(model, i_v_prev)
        i_v = get_model_v_fd(model, loss_function, optimizer, X, y, i_v_prev, lambda_to_sub, preconditioner, i_epsilon)


        i_v = i_v.type(torch.float)
        # i_l -> (i_v_prev.t()*i_v)/((i_v_prev.t()*i_v_prev)) since i_v_prev is normalized denominator is always 1
        i_l = torch.matmul(i_v_prev.t(), i_v)
        i_v_norm = torch.matmul(i_v.t(), i_v)**0.5
        i_v = i_v / i_v_norm


        i = i + 1
        cosin_dist = get_dist(i_v, i_v_prev)

        cosin_dist_diff = abs(abs(cosin_dist) - abs(prev_cosin_dist))
        lambda_dist = i_l - i_l_prev
       
    i_v = torch.reshape(i_v, (-1,))

    return i_l, i_v, cosin_dist, df_stats, i




def get_V0L0_VnLn_finite_differencing(model, loss_function, optimizer, X, y, max_iter_num, preconditioner1 = None, preconditioner2 = None):
    L0, V0, dist_converged_0, converge_stats0, i0 = get_L0V0_finite_differencing(model, loss_function, optimizer, X, y, max_iter_num, is_first=True, lambda_to_sub=0, preconditioner = preconditioner1)
    converge_stats0["type"] = "0"
    second_iter_Ln, Vn, dist_converged_n, converge_statsn, iN = get_L0V0_finite_differencing(model, loss_function, optimizer, X, y, max_iter_num, is_first=False, lambda_to_sub=L0, preconditioner = preconditioner2)
    converge_statsn["type"] = "n"
    Ln = abs(L0) - second_iter_Ln
    return V0, L0, dist_converged_0, converge_stats0, Vn, Ln, dist_converged_n, converge_statsn, i0, iN, second_iter_Ln


def get_V0L0_VnLn(A):
    # L, V = get_sorted_LV_by_abs_L(A)
    # V0 = V[:, 0]
    # L0 = L[0]
    L, V = get_sorted_LV(A)
    Vn = V[:, -1]
    Ln = L[-1]
    V0 = V[:, 0]
    L0 = L[0]
    return V0, L0, Vn, Ln

def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models in perfect match')
        
        
# def get_closest_l_to_givenL(L, l):
#     ref_tensor = (L.abs()-l.abs()).abs()
#     min_index = torch.argmin(ref_tensor)
#     if (ref_tensor[min_index].item() != 0.0):
#         return L[min_index].item()
#     else:  # found l itself deleting it and researching
#         ref_tensor = torch.cat(
#             [ref_tensor[0:min_index], ref_tensor[min_index+1:]])
#         ref_L = torch.cat([L[0:min_index], L[min_index+1:]])
#         min_index = torch.argmin(ref_tensor)
#         return ref_L[min_index].item()

def get_model_loss(model, v, step_size, X, y, loss_function, optimizer, enable_grad):
    total_loss = 0.0
    model_init_state_dict = copy.deepcopy(model.state_dict)

    model_init_params = model_parametres_to_tensor(model).clone()
    set_model_parameters(model, model_init_params + v*step_size)
    train_dataset = TensorDataset(X, y)
    data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    if enable_grad:
        model.train()
    else:
        model.eval()
    with torch.set_grad_enabled(enable_grad):
        for i, (iX, iy) in enumerate(data_loader):
            if enable_grad:
                optimizer.zero_grad()   
            probs = model(iX)
            i_loss = loss_function(probs, iy)
            total_loss += i_loss*len(iy)
            if enable_grad:
                i_loss.backward()

    set_model_parameters(model, model_init_params)
    model.state_dict = model_init_state_dict
    optimizer.zero_grad()
    total_loss /= len(y)
    return total_loss

def precond_tri_diagonal(model, loss_function, optimizer, X, y, v):
    delta = torch.tensor.ones([len(v),1])*(2/len(v))**0.5.to(DEVICE)
    ## consider delta_i to be  δi = max(|xi|, 1)
    v1 = delta.clone().to(DEVICE)
    v1[range(1,len(v),2)] = 0.0 #v1 = [d1, 0, d2, 0, ...]
    v2 = delta - v1 #v1 = [0, d1, 0, d3, ...]
    epsilon = get_epsilon(model, v)
    alpha1 = get_model_v_fd(model, loss_function, optimizer, X, y, v, None, None, epsilon)
    
def precond_bi_diagonal(v, ratio):
    shift_v = torch.zeros(v.shape).to(DEVICE)
    shift_v[:-1] = v[1:]
    precond = (shift_v + v*ratio).clone().reshape([v.shape[0],1]) #detach
    return precond

def precond_grad_sqr(model, loss_function, optimizer, X, y):
    grads = get_grads(model, X, y, loss_function, optimizer)
    shifted_grads = grads
    shifted_grads[torch.logical_and(shifted_grads<=0, shifted_grads>=-1e-6)] = -1e-6
    shifted_grads[torch.logical_and(shifted_grads>=0, shifted_grads<=1e-6)] = 1e-6
    shifted_grads = shifted_grads.abs()**(-0.5)
    shifted_grads /= torch.linalg.vector_norm(shifted_grads, 2)
    return shifted_grads

def evaluate_smallest_eig_model(filepath):
    
    CIFAR10_RESNET18.load_data()
    CIFAR10_RESNET18.load_model(filepath)
    
    model, val_acc_history, epochs_data_pre_lobpcg =  CIFAR10_RESNET18.train_model(3, phases = ['train', 'val']) #single epoch
    
    
    print("number of trainable parameters of model:", CIFAR10_RESNET18.get_model_size())
    print("is cude:", next(CIFAR10_RESNET18.model.parameters()).is_cuda)
    # epoch_data[phase] = {"grads" : epoch_grad, "loss": epoch_loss, "acc": epoch_acc}
    mean_grads9999_prelobpcg = epochs_data_pre_lobpcg[0]['train']['mean_epoch_grads_norm_l0_at_99p99percntile']
    std_grads9999_prelobpcg = epochs_data_pre_lobpcg[0]['train']['std_epoch_grads_norm_l0_at_99p99percntile']
    print("train mean grad before lob: ", mean_grads9999_prelobpcg)
    print("with std of: ", std_grads9999_prelobpcg)
    print("train loss before lob: ", epochs_data_pre_lobpcg[0]['train']['loss'])
    print("train loss before lob: ", epochs_data_pre_lobpcg[1]['train']['loss'])
    print("train loss before lob: ", epochs_data_pre_lobpcg[2]['train']['loss'])
    print("val loss before lob: ", epochs_data_pre_lobpcg[0]['val']['loss'])
    print("val loss before lob: ", epochs_data_pre_lobpcg[1]['val']['loss'])
    print("val loss before lob: ", epochs_data_pre_lobpcg[2]['val']['loss'])
    
    
    loss_function = CIFAR10_RESNET18.criterion
    optimizer = CIFAR10_RESNET18.optimizer
    lr_scheduler = CIFAR10_RESNET18.lr_scheduler    
    data_x, data_y = ((CIFAR10_RESNET18.dataloaders['train']).dataset).datasets[0][:]
    
    data_x = data_x.to(DEVICE)
    data_y = torch.tensor(data_y).to(DEVICE)
    model_size = get_model_size(model.parameters())

    pre_lob_loss = get_model_loss(model, torch.zeros([model_size, 1]).to(DEVICE), 0.0, data_x, data_y, loss_function, optimizer, True)
    print("outer func loss before lob: ", pre_lob_loss)

    
    CIFAR10_RESNET18.load_data()
    CIFAR10_RESNET18.load_model(filepath)
    
    model = CIFAR10_RESNET18.model
    loss_function = CIFAR10_RESNET18.criterion
    optimizer = CIFAR10_RESNET18.optimizer
    lr_scheduler = CIFAR10_RESNET18.lr_scheduler
    
    data_x, data_y = ((CIFAR10_RESNET18.dataloaders['train']).dataset).datasets[0][:]
        
    # data_val_x, data_val_y = (CIFAR10_RESNET18.dataloaders['val']).dataset[:]
    
    data_x = data_x.to(DEVICE)
    data_y = torch.tensor(data_y).to(DEVICE)
    
    # data_val_x = data_val_x.to(DEVICE)
    # data_val_y = torch.tensor(data_val_y).to(DEVICE)
    
    
    model.eval()
    
    preconditioner = None
    model_size = get_model_size(model.parameters())
    results_dict = {}
    run_log_PRECOND = pd.DataFrame()
    run_log = pd.DataFrame()
    try:
        v = (torch.ones(get_model_size(model.parameters())) /torch.linalg.vector_norm(torch.ones(get_model_size(model.parameters())),2)).to(DEVICE)
        step_size = get_epsilon(model, v)
        # cur_lr = lr_scheduler.get_last_lr()[0]
        # prev_step_size = epochs_data_pre_lobpcg[0]['train']['grads_norm_2']*epochs_data_pre_lobpcg[0]['train']['init_lr']
        print("step_size:", step_size)
        # init_val_loss = get_model_loss(model, torch.zeros([model_size, 1]).to(DEVICE), 0.0, data_val_x, data_val_y, loss_function, optimizer)
        
        init_loss = get_model_loss(model, torch.zeros([model_size, 1]).to(DEVICE), 0.0, data_x, data_y, loss_function, optimizer, True)
        print("init_loss:", init_loss)
        run_log_PRECOND = pd.DataFrame()

        
        Vn_LOBPCG, Ln_LOBPCG, iN_LOBPCG, run_log = get_L0V0_LnVn_LOBPCG(model, loss_function, optimizer, data_x, data_y, step_size, adjust_step_size=True, use_preconditioner=False, ref_v = torch.zeros(1), ref_l =0.0, filepath=filepath)
        # Vn_LOBPCG, Ln_LOBPCG, iN_LOBPCG, run_log = get_LnVn_LOBPCG(model, loss_function, optimizer, data_x, data_y, step_size, adjust_step_size=False, use_preconditioner=False)
        print("finished LOB")
        LOBPCG_loss = get_model_loss(model, Vn_LOBPCG, step_size, data_x, data_y, loss_function, optimizer, False)
        print("LOBPCG_loss:", LOBPCG_loss)
           
        new_model_parameters = model_parametres_to_tensor(model).clone() + Vn_LOBPCG*step_size#Vn_stepsize

        set_model_parameters(model, new_model_parameters)
        
        
        CIFAR10_RESNET18.load_data()
        CIFAR10_RESNET18.load_model(filepath)
        print("=== model vs disk")
        compare_models(model, CIFAR10_RESNET18.model)

        
        
        CIFAR10_RESNET18.set_model(model)
        print("=== model with new params vs disk")
        compare_models(model, CIFAR10_RESNET18.model)

        model, val_acc_history, epochs_data_after_lobpcg =  CIFAR10_RESNET18.train_model(3, phases = ['train', 'val']) #single epoch
        print("=== model with new params after 3 trains vs disk")
        compare_models(model, CIFAR10_RESNET18.model)
    
        # epoch_data[phase] = {"grads" : epoch_grad, "loss": epoch_loss, "acc": epoch_acc}
        mean_grads9999_afterlobpcg = epochs_data_after_lobpcg[0]['train']['mean_epoch_grads_norm_l0_at_99p99percntile']
        std_grads9999_afterlobpcg = epochs_data_after_lobpcg[0]['train']['std_epoch_grads_norm_l0_at_99p99percntile']
        print("train mean grad after lob: ", mean_grads9999_afterlobpcg)
        print("with std of: ", std_grads9999_afterlobpcg)
        
        print("train loss after lob: ", epochs_data_after_lobpcg[0]['train']['loss'])
        print("train loss after lob: ", epochs_data_after_lobpcg[1]['train']['loss'])
        print("train loss after lob: ", epochs_data_after_lobpcg[2]['train']['loss'])
        results_dict = {"Model size" : model_size, 
                        "Model loss" : init_loss.item(),
                        "LOBPCG #iteration" :  iN_LOBPCG,
                        "LOBPCG lambda" : Ln_LOBPCG,
                        "loss pre LOBPCG 0" : epochs_data_pre_lobpcg[0]['train']['loss'],
                        "loss pre LOBPCG 1" : epochs_data_pre_lobpcg[1]['train']['loss'],
                        "loss pre LOBPCG 2" : epochs_data_pre_lobpcg[2]['train']['loss'],
                        "pre_lob_loss" : pre_lob_loss,
                        "LOBPCG loss" : LOBPCG_loss.item(),
                        "loss after LOBPCG" : epochs_data_after_lobpcg[0]['train']['loss'],
                        "loss after LOBPCG1" : epochs_data_after_lobpcg[1]['train']['loss'],
                        "loss after LOBPCG2" : epochs_data_after_lobpcg[2]['train']['loss'],
                        "mean grads pre LOBPCG" : mean_grads9999_prelobpcg,
                        "std grads pre LOBPCG" : std_grads9999_prelobpcg,
                        "mean grads after LOBPCG" : mean_grads9999_afterlobpcg,
                        "std grads after LOBPCG" : std_grads9999_afterlobpcg,
                        }
        
        

    except Exception as e:
        print(e)
        print("An exception occurred in eigs calculation")
       
    finally:

        return results_dict, run_log_PRECOND, run_log



def evaluate_run_over_simple_nn_model():
    # model = Tiny_Linear_NN()
    # new_params = torch.tensor([1.0, 1.0, 0.0,0.0, 2.0, 4.0, 1.0])
    # set_model_parameters(model,new_params)
    # model.zero_grad()

    model = Simple_NN(100).to(DEVICE)

    #data_x = torch.tensor([[1.0], [2.0], [3.0]])
    data_x = torch.normal(0.0, 1, size=(10000, 1)).to(DEVICE)
    with torch.no_grad():
        data_y = model(data_x)+torch.normal(0.0, 10, size=(10000, 1)).to(DEVICE)

    loss_function = nn.MSELoss(reduction='mean')
    # CustomLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    # func, func_params = functorch.make_functional(model)
    def model_loss_sum_params(params, X, data_y):
        # pred_y: torch.Tensor = _stateless.functional_call(model, {n: p for n, p in zip(names, params)}, data_x)
        pred_y = functional_call(model, params, X)
        loss = loss_function(pred_y, data_y)
        return loss.sum()

    names = list(n for n, _ in model.named_parameters())
    model_parameters = tuple(model.parameters())
    is_comparable = True
    correct_cosin = True
    convereged = True

    dist_converged_stats = pd.DataFrame()
    overview_run_stats = pd.DataFrame()
    try:
        H = hessian(model_loss_sum_params)(
            dict(model.named_parameters()), data_x, data_y)

        # reshaping hessian to nXn matrix
        HM = reshape_hessian(H, model_parameters)
        # HM should be symmetrical matrix. If it isn't (due to nummerical instability) and in order to avoid complex eig vectors:
        diff = HM - HM.t()
        if (diff.max() > 0):
            HM = ((HM + HM.t())/2).to(DEVICE)
        #print("L0 by finite differencing: ", L0)
        #HM_condition_number = torch.linalg.cond(HM)
        #print("condition number of hessian: ", HM_condition_number)
        V0_HM, L0_HM, Vn_HM, Ln_HM = get_V0L0_VnLn(HM)
        L, V = get_sorted_LV_by_abs_L(HM)

        v = (torch.ones(get_model_size(model.parameters())) /torch.linalg.vector_norm(torch.ones(get_model_size(model.parameters())),2)).to(DEVICE)
        step_size = get_epsilon(model, v)

        i_v_min, i_l_min, i, run_stats, i_v_min_step_size = get_L0V0_LnVn_LOBPCG(model, loss_function, optimizer, data_x, data_y, step_size, adjust_step_size=True, use_preconditioner=False, ref_v= Vn_HM, ref_l = Ln_HM, ref_H= HM, filepath="")

        print(i_l_min)
        print(Ln_HM)
        
        print(get_dist(i_l_min, Vn_HM))
        print("a")
    except Exception as e:
        print(e)
        print("An exception occurred in eigs calculation")
       
def evaluate_epsilon():
    epsilon_stats_df = pd.DataFrame()
    iteration_num = 30
    model_sizes = [150, 200, 250]
    # [10, 25, 50, 75, 100, 150, 200, 250]
    epsilon_multiplier = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
    vector_number = 100
    for model_size in model_sizes:
        print("model size:", model_size)
        for i in range(iteration_num):
            print("model num:", i)
            model = Simple_NN(model_size).to(DEVICE)
    
            #data_x = torch.tensor([[1.0], [2.0], [3.0]])
            data_x = torch.normal(0.0, 1, size=(10000, 1)).to(DEVICE)
            with torch.no_grad():
                data_y = model(data_x)+torch.normal(0.0, 10, size=(10000, 1)).to(DEVICE)
    
            loss_function = nn.MSELoss(reduction='mean')
            # CustomLoss()
    
            optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    
            # func, func_params = functorch.make_functional(model)
            def model_loss_sum_params(params, X, data_y):
                # pred_y: torch.Tensor = _stateless.functional_call(model, {n: p for n, p in zip(names, params)}, data_x)
                pred_y = functional_call(model, params, X)
                loss = loss_function(pred_y, data_y)
                return loss.sum()
    
            names = list(n for n, _ in model.named_parameters())
            model_parameters = tuple(model.parameters())
            is_comparable = True
            correct_cosin = True
            convereged = True
    
            dist_converged_stats = pd.DataFrame()
            overview_run_stats = pd.DataFrame()
            try:
                H = hessian(model_loss_sum_params)(
                    dict(model.named_parameters()), data_x, data_y)
    
                # reshaping hessian to nXn matrix
                HM = reshape_hessian(H, model_parameters)
                # HM should be symmetrical matrix. If it isn't (due to nummerical instability) and in order to avoid complex eig vectors:
                diff = HM - HM.t()
                if (diff.max() > 0):
                    HM = ((HM + HM.t())/2).to(DEVICE)
                
                model_init_params = model_parametres_to_tensor(model).clone()
                for j in range(vector_number):
                    v = torch.rand([len(model_init_params), 1]).to(DEVICE) 
                    v = v / torch.linalg.vector_norm(v,2)
                    epsilon = get_epsilon(model, v)
                    for mult in epsilon_multiplier:
                        Av = get_model_v_fd(model, loss_function, optimizer, data_x, data_y, v, lambda_to_substract=None, preconditioner = None, epsilon = epsilon*mult)
                        Hv = HM@v
                        dist = get_dist(Av,Hv)
                        row_dict = {
                            "cos_dist" : dist,
                            "orig_epsilon" : epsilon,
                            "mult": mult,
                            "mult_epsilon" : epsilon*mult,
                            "vec_number" : j,
                            "model_number" : i,
                            "number_hidden_units": model_size,
                            "model_size" : len(model_init_params),
                            "model_norm" : torch.linalg.vector_norm(model_init_params,2).item(),
                            }
    
                        epsilon_stats_df = pd.concat([epsilon_stats_df, pd.DataFrame([row_dict])])
            except Exception as e:
                print(e)
                print("An exception occurred in eigs calculation")
        with pd.ExcelWriter(OUT_PATH_SUMMARY + "\\big_m_epsilon_stats_"+str(model_size)+".xlsx") as writer:
            epsilon_stats_df.to_excel(writer, sheet_name='runs_precond_stats')
    return epsilon_stats_df

def main():
    # epsilon_stats_df = evaluate_epsilon()
    # with pd.ExcelWriter(OUT_PATH_SUMMARY + "\\epsilon_stats_n.xlsx") as writer:
    #     epsilon_stats_df.to_excel(writer, sheet_name='runs_precond_stats')
   
    overview_run_stats = pd.DataFrame()
    runs_precond_stats = pd.DataFrame()
    runs_stats = pd.DataFrame()
    
    for model_path in MODEL_PATHS:
        print(model_path)
        res, run_log_PRECOND, run_log = evaluate_smallest_eig_model(model_path)
            
        print(res)
        overview_run_stats_i = pd.DataFrame([res])
        overview_run_stats_i["file"] = model_path
        overview_run_stats = pd.concat([overview_run_stats,overview_run_stats_i])
        run_log_PRECOND["file"] = model_path
        runs_precond_stats = pd.concat([runs_precond_stats, run_log_PRECOND])
        run_log["file"] = model_path
        runs_stats = pd.concat([runs_stats, run_log])
        now_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        with pd.ExcelWriter(OUT_PATH_SUMMARY + "\\MLP_two-ways_with_step_grad_" + now_str + "_overview_run_stats.xlsx") as writer:
            overview_run_stats.to_excel(writer, sheet_name='data_summary')
        with pd.ExcelWriter(OUT_PATH_SUMMARY + "\MLP_two-ways_with_step_grad_" + now_str + "_runs_precond_stats.xlsx") as writer:
            runs_precond_stats.to_excel(writer, sheet_name='runs_precond_stats')
        with pd.ExcelWriter(OUT_PATH_SUMMARY + "\\MLP_two-ways_with_step_grad_" + now_str + "_runs_stats.xlsx") as writer:
            runs_stats.to_excel(writer, sheet_name='runs_precond_stats')
            
if __name__ == '__main__':
    main()

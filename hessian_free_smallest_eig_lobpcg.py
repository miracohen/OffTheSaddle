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

#from torch.nn.utils import stateless

import pickle

import pandas as pd
import os

import datetime
import time

# from MNIST_resnet18 import Block, ResNet_18, CustomTensorDataset, MNIST_RESNET18
from CIFAR10_resnet18 import Block, ResNet_18, CustomTensorDataset, CIFAR10_RESNET18

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LR = 0.0001
WEIGHT_DECAY=1e-4
STEP_SIZE = 1e-2
# GLOBALS
MAX_ITER_NUM = 30
COSIN_DIST = 2*1e-02
MAX_DIST_CONVERGENCE = 1e-07
EM = 1e-07  # machine precision
LANCZOS_DIM = 1e+01
N_HIDDEN = 100
MAX_SIZE_CALCULATED_H = 100

DROP_PROB = 0.3
OUTPUT_DIM = 10

BATCH_SIZE = 32

N_HALT_MULTIPLIER = 3

OUT_PATH_SUMMARY = r'C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD'

MODEL_PATHS =[ 
        r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240328-125845LR=0.01WEIGHT_DECAY=0.01\91_model__test_acc=0.65560_train_loss=0.02753_test_loss=1.12555_lr=0.00000_grad=0.06716.pth",
        r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240328-125858LR=0.01WEIGHT_DECAY=0.01\81_model__test_acc=0.66470_train_loss=0.02722_test_loss=1.10858_lr=0.00000_grad=0.06759.pth",
        r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240328-125833LR=0.01WEIGHT_DECAY=0.01\94_model__test_acc=0.64970_train_loss=0.02692_test_loss=1.14297_lr=0.00000_grad=0.06718.pth",
        r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240328-100039LR=0.01WEIGHT_DECAY=0.01\84_model__test_acc=0.65860_train_loss=0.02666_test_loss=1.11371_lr=0.00000_grad=0.06807.pth",
        r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240326-173955LR=0.01WEIGHT_DECAY=0.01\81_model__test_acc=0.65940_train_loss=0.02735_test_loss=1.10770_lr=0.00000_grad=0.06689.pth",
        r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240326-142704LR=0.01WEIGHT_DECAY=0.01\71_model__test_acc=0.65680_train_loss=0.02699_test_loss=1.12244_lr=0.00000_grad=0.06644.pth",
        r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240326-142603LR=0.01WEIGHT_DECAY=0.01\72_model__test_acc=0.65770_train_loss=0.02721_test_loss=1.09544_lr=0.00000_grad=0.06745.pth",
        r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240326-142539LR=0.01WEIGHT_DECAY=0.01\88_model__test_acc=0.65630_train_loss=0.02693_test_loss=1.11485_lr=0.00000_grad=0.06718.pth",
        r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240326-142534LR=0.01WEIGHT_DECAY=0.01\76_model__test_acc=0.66260_train_loss=0.02675_test_loss=1.09290_lr=0.00000_grad=0.06762.pth",
        r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240325-221555LR=0.01WEIGHT_DECAY=0.01\86_model__test_acc=0.65420_train_loss=0.02705_test_loss=1.13709_lr=0.00000_grad=0.06622.pth",
        r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240325-221631LR=0.01WEIGHT_DECAY=0.01\81_model__test_acc=0.65740_train_loss=0.02742_test_loss=1.11738_lr=0.00000_grad=0.06801.pth",
        r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240325-221603LR=0.01WEIGHT_DECAY=0.01\71_model__test_acc=0.65320_train_loss=0.02686_test_loss=1.12823_lr=0.00000_grad=0.06737.pth",
        r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240325-221655LR=0.01WEIGHT_DECAY=0.01\72_model__test_acc=0.65480_train_loss=0.02689_test_loss=1.12950_lr=0.00000_grad=0.06649.pth",
        r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240325-221715LR=0.01WEIGHT_DECAY=0.01\91_model__test_acc=0.65110_train_loss=0.02748_test_loss=1.13140_lr=0.00000_grad=0.06763.pth",
        r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240325-075155LR=0.01WEIGHT_DECAY=0.01\13_model__test_acc=0.65960_train_loss=0.05885_test_loss=1.08773_lr=0.00100_grad=0.12887.pth",
        r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240325-075217LR=0.01WEIGHT_DECAY=0.01\11_model__test_acc=0.66030_train_loss=0.15025_test_loss=1.06402_lr=0.00100_grad=0.27435.pth",
        r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240322-181049LR=0.01WEIGHT_DECAY=0.01\86_model__test_acc=0.66260_train_loss=0.02692_test_loss=1.10135_lr=0.00000_grad=0.06617.pth",
        r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240322-181243LR=0.01WEIGHT_DECAY=0.01\13_model__test_acc=0.65790_train_loss=0.05919_test_loss=1.09230_lr=0.00100_grad=0.12977.pth",
        r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240322-180247LR=0.01WEIGHT_DECAY=0.01\14_model__test_acc=0.65630_train_loss=0.04158_test_loss=1.11635_lr=0.00100_grad=0.09695.pth",
        r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240322-180701LR=0.01WEIGHT_DECAY=0.01\11_model__test_acc=0.65220_train_loss=0.15224_test_loss=1.08844_lr=0.00100_grad=0.27484.pth",
        r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240322-180342LR=0.01WEIGHT_DECAY=0.01\11_model__test_acc=0.66000_train_loss=0.15108_test_loss=1.05759_lr=0.00100_grad=0.27612.pth",
        

        
    
        
    
        
    
        
    
        # r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240325-075155LR=0.01WEIGHT_DECAY=0.01\3_model__test_acc=0.57250_train_loss=0.96784_test_loss=1.19940_lr=0.01000_grad=0.16128.pth",
        # r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240325-075155LR=0.01WEIGHT_DECAY=0.01\13_model__test_acc=0.65960_train_loss=0.05885_test_loss=1.08773_lr=0.00100_grad=0.12887.pth",
        
        # r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240325-075217LR=0.01WEIGHT_DECAY=0.01\5_model__test_acc=0.61400_train_loss=0.73332_test_loss=1.12663_lr=0.01000_grad=0.18002.pth",
        # r"C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240325-075217LR=0.01WEIGHT_DECAY=0.01\11_model__test_acc=0.66030_train_loss=0.15025_test_loss=1.06402_lr=0.00100_grad=0.27435.pth",
    
        # r'C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240322-181243LR=0.01WEIGHT_DECAY=0.01\13_model__test_acc=0.65790_train_loss=0.05919_test_loss=1.09230_lr=0.00100_grad=0.12977.pth',
        # r'C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240322-181243LR=0.01WEIGHT_DECAY=0.01\5_model__test_acc=0.61380_train_loss=0.74479_test_loss=1.12920_lr=0.01000_grad=0.18092.pth',
        # r'C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240322-181243LR=0.01WEIGHT_DECAY=0.01\2_model__test_acc=0.56770_train_loss=1.16557_test_loss=1.20509_lr=0.01000_grad=0.15457.pth',
        
        # r'C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240322-180342LR=0.01WEIGHT_DECAY=0.01\1_model__test_acc=0.50930_train_loss=1.50689_test_loss=1.37861_lr=0.01000_grad=0.16921.pth',
        # r'C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240322-180342LR=0.01WEIGHT_DECAY=0.01\4_model__test_acc=0.60210_train_loss=0.84650_test_loss=1.12993_lr=0.01000_grad=0.16902.pth',
        # r'C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240322-180342LR=0.01WEIGHT_DECAY=0.01\11_model__test_acc=0.66000_train_loss=0.15108_test_loss=1.05759_lr=0.00100_grad=0.27612.pth',
        
        # r'C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240322-180701LR=0.01WEIGHT_DECAY=0.01\1_model__test_acc=0.47220_train_loss=1.50812_test_loss=1.52900_lr=0.01000_grad=0.16927.pth',
        # r'C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240322-180701LR=0.01WEIGHT_DECAY=0.01\7_model__test_acc=0.61620_train_loss=0.57100_test_loss=1.14745_lr=0.01000_grad=0.21868.pth',
        # r'C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240322-180701LR=0.01WEIGHT_DECAY=0.01\11_model__test_acc=0.65220_train_loss=0.15224_test_loss=1.08844_lr=0.00100_grad=0.27484.pth',
        
        # r'C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240322-181049LR=0.01WEIGHT_DECAY=0.01\2_model__test_acc=0.58130_train_loss=1.15540_test_loss=1.18013_lr=0.01000_grad=0.15665.pth',
        # r'C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240322-181049LR=0.01WEIGHT_DECAY=0.01\28_model__test_acc=0.66060_train_loss=0.02724_test_loss=1.09623_lr=0.00001_grad=0.06761.pth',
        # r'C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240322-181049LR=0.01WEIGHT_DECAY=0.01\86_model__test_acc=0.66260_train_loss=0.02692_test_loss=1.10135_lr=0.00000_grad=0.06617.pth',
        
        # r'C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240322-181243LR=0.01WEIGHT_DECAY=0.01\1_model__test_acc=0.48750_train_loss=1.50683_test_loss=1.42025_lr=0.01000_grad=0.17014.pth',
        # r'C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240322-181243LR=0.01WEIGHT_DECAY=0.01\4_model__test_acc=0.59050_train_loss=0.85188_test_loss=1.16334_lr=0.01000_grad=0.16919.pth',
        # r'C:\Users\mirac\Documents\OffSaddle\runs\CIFAR10_RESNET18\SGD\20240322-181243LR=0.01WEIGHT_DECAY=0.01\13_model__test_acc=0.65790_train_loss=0.05919_test_loss=1.09230_lr=0.00100_grad=0.12977.pth',
        
        ###############
        # r'C:\Users\mirac\Documents\OffSaddle\runs\MNIST_RESNET18\ADAMW\20240227-213302LR=0.0001WEIGHT_DECAY=0.0001\2_model__test_acc=0.99283_train_loss=0.03032_test_loss=0.02500_lr=0.00010_grad=0.01397.pth',
        # r'C:\Users\mirac\Documents\OffSaddle\runs\MNIST_RESNET18\ADAMW\20240227-213302LR=0.0001WEIGHT_DECAY=0.0001\24_model__test_acc=0.99575_train_loss=0.00038_test_loss=0.01654_lr=0.00001_grad=0.00164.pth',
        # r'C:\Users\mirac\Documents\OffSaddle\runs\MNIST_RESNET18\ADAMW\20240227-213302LR=0.0001WEIGHT_DECAY=0.0001\25_model__test_acc=0.99592_train_loss=0.00033_test_loss=0.01638_lr=0.00001_grad=0.00139.pth',
              
              
        # r'C:\Users\mirac\Documents\OffSaddle\runs\MNIST_RESNET18\ADAMW\20240227-195553LR=0.0001WEIGHT_DECAY=0.0001\3_model__test_acc=0.99333_train_loss=0.02208_test_loss=0.02249_lr=0.00010_grad=0.01018.pth',
        # r'C:\Users\mirac\Documents\OffSaddle\runs\MNIST_RESNET18\ADAMW\20240227-195553LR=0.0001WEIGHT_DECAY=0.0001\20_model__test_acc=0.99558_train_loss=0.00032_test_loss=0.01955_lr=0.00001_grad=0.00158.pth',
        # r'C:\Users\mirac\Documents\OffSaddle\runs\MNIST_RESNET18\ADAMW\20240227-195553LR=0.0001WEIGHT_DECAY=0.0001\28_model__test_acc=0.99575_train_loss=0.00042_test_loss=0.02097_lr=0.00000_grad=0.00703.pth',
              ]



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
    total_grads = torch.zeros(get_model_size(model.parameters())).to(DEVICE)
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
    grads = grads.clone()  # detach
    return grads


def model_parametres_to_tensor(model):
    model_params_tensor = []
    for param in model.parameters():
        model_params_tensor.append(param.data.view(-1).clone())
    model_params_tensor = torch.cat(model_params_tensor)
    return model_params_tensor


def get_epsilon(model, v):
    # N. Andrei, “Accelerated conjugate gradient algorithm with finite difference hessian/vector product approximation for unconstrained optimization,”
    # epsilon = 2*sqrt(machine precision)*(model's norm + 1)/(v's norm)
    model_params = model_parametres_to_tensor(model)
    model_size = torch.matmul(model_params.t(), model_params)**0.5
    v_size = torch.matmul(v.t(), v)**0.5
    epsilon = 2*(EM**0.5)*(1+model_size)/v_size
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
    i_v = torch.ones(get_model_size(tuple(model.parameters()))).to(DEVICE)
    i_v = i_v/torch.linalg.vector_norm(i_v, ord=2)
    i_v_prev = i_v.clone()
    i_w_tag = get_model_v_fd(model, loss_function,
                             optimizer, X, y, i_v, lambda_to_sub, preconditioner)
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
            i_w_t = get_model_v_fd(model, loss_function,
                                   optimizer, X, y, i_v, lambda_to_sub, preconditioner)
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
    print("term1:",term1)
    print("term2:",term2)
    return term1+term2

#get_model_loss(model, i_v, step_size, X, y, loss_function, optimizer)
def get_loss_point_ratio(model, v, X, y, loss_function, optimizer, a, b, delta_tolerance = 0.0, n = 5):
    tau = (3.0-5.0**0.5)/2.0
    delta = b-a
    p = a + tau*delta
    q = a + (1-tau)*delta
    loss_a = get_model_loss(model, v, a, X, y, loss_function, optimizer)
    loss_p = get_model_loss(model, v, p, X, y, loss_function, optimizer)
    loss_q = get_model_loss(model, v, q, X, y, loss_function, optimizer)
    loss_b = get_model_loss(model, v, b, X, y, loss_function, optimizer)
    print("losses a = {}, p = {}, q = {} , b = {}".format(loss_a, loss_p, loss_q, loss_b))
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
            loss_p = get_model_loss(model, v, p, X, y, loss_function, optimizer)
            print("losses a = {}, p = {}, q = {} , b = {} at iteration i={}".format(loss_a, loss_p, loss_q, loss_b, i))
        else:
            a = p
            loss_a = loss_p
            p = q
            loss_p = loss_q
            delta = b - a
            q = a + (1-tau)*delta
            loss_q = get_model_loss(model, v, q, X, y, loss_function, optimizer)
            print("losses a = {}, p = {}, q = {} , b = {} at iteration i={}".format(loss_a, loss_p, loss_q, loss_b, i))

    opt_point = (b+a)/2.0
    loss_opt_point = get_model_loss(model, v, opt_point, X, y, loss_function, optimizer)
    print("losses at opt point = {} is = {}".format(opt_point, loss_opt_point))

    return opt_point, loss_opt_point



def get_LnVn_LOBPCG(model, loss_function, optimizer, X, y, step_size, use_preconditioner = False):
    run_stats = pd.DataFrame()#columns=['iteration', 'cosin vec', 'lambda', 'lambda diff', 'loss', 'loss_diff', 'halt condition'])
    
    t0 = time.time()
    model_init_params = model_parametres_to_tensor(model).clone()
    i_model_params = model_init_params.clone()
    model_size = get_model_size(tuple(model.parameters()))
    i_v = torch.ones(model_size).to(DEVICE)
    i_v = i_v/torch.linalg.vector_norm(i_v, ord=2)
    i_v_prev = i_v.clone()
    t1 = time.time()
    time_elapsed = t1-t0
    print('build up complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    i_Av = get_model_v_fd(model, loss_function, optimizer, X, y, i_v, None, None)
    t2 = time.time()
    time_elapsed = t2-t1
    print('get_model_v_fd complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    i_w = i_Av - torch.matmul(i_v.t(), i_Av) * i_v  # A*v -v'Av*v
    i_w = i_w/torch.linalg.vector_norm(i_w, ord=2)
    V = torch.cat((i_v.reshape(1, model_size), i_w.reshape(1, model_size)))
    t3 = time.time()
    i_v, i_l, i_r = get_LnVn_LOBPCG_Rayleigh_Ritz(model, loss_function, optimizer, X, y, V, preconditioner = None)
    t4 = time.time()
    time_elapsed = t4-t3
    print('get_LnVn_LOBPCG_Rayleigh_Ritz in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    i_v = i_v/torch.linalg.vector_norm(i_v, ord=2)
    i = 0
    cosin_dist = get_dist(i_v, i_v_prev)
    cosin_dist_prev = 0
    v_min = i_v.clone()
    l_min = torch.matmul(i_Av, i_v.t()) / torch.matmul(i_v.t(), i_v)
    l_min_dist_abs = abs(cosin_dist)
    l_min_prev_dist_abs = 0
    epsilon = get_epsilon(model, v_min)
    i_halt_condition = get_min_halt_condition(i_Av, i_l, i_v, epsilon)
    i_delta = torch.linalg.norm(i_Av - i_v*i_l)
    i_l_prev = 0.0
    LOBPCG_loss_prev = 0.0
    print("=============b4 first iteration==============")
    print("i_l: ", i_l)
    print("i_halt_condition: ", i_halt_condition)
    print("condition: ", ((i_l)**2 < (N_HALT_MULTIPLIER*i_halt_condition)**2))
    print("=============================================")
    # Halting when (i_l)**2 > delta**2
    while (i < MAX_ITER_NUM) and ((i_l)**2 < (N_HALT_MULTIPLIER*i_halt_condition)**2) :
    #and (i_l > 0)
    #(i < MAX_ITER_NUM) and (abs(abs(cosin_dist)-1) > COSIN_DIST):
    #(abs(abs(cosin_dist) - abs(cosin_dist_prev)) > COSIN_DIST) and 
        print("LOBPCG loop at iteration: ", i)
        t4 = time.time()
        V = torch.cat((i_v.reshape(1, model_size), i_w.reshape(1, model_size), i_v_prev.reshape(1, model_size)))
        i_v_prev = i_v.clone()
        i_l_prev = i_l
        t5 = time.time()
        # time_elapsed = t5-t4
        # print('LOBLOOP part I {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        i_v, i_l, i_r = get_LnVn_LOBPCG_Rayleigh_Ritz(model, loss_function, optimizer, X, y, V, preconditioner = None)
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
            precond= precond_bi_diagonal(i_r, 0.95)
            i_w = precond#precond_bi_diagonal*i_r
            
        
        i_w = i_w/torch.linalg.vector_norm(i_w, ord=2)
        t7 = time.time()
        # time_elapsed = t7-t6
        # print('LOBLOOP part III {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
       
        cosin_dist_prev = cosin_dist
        cosin_dist = get_dist(i_v, i_v_prev)
        i_l = torch.matmul(i_Av, i_v.t()) / torch.matmul(i_v.t(), i_v)
        tmp_halt_condition = get_min_halt_condition(i_Av, i_l, i_v, epsilon)
        print("tmp_halt_condition: {} is greater than lambda : {} in abs  {}".format( tmp_halt_condition, i_l, (l_min**2 < tmp_halt_condition**2) ))
        LOBPCG_loss = get_model_loss(model, i_v, step_size, X, y, loss_function, optimizer)
        opt_point_LOBPCG, opt_point_LOBPCG_loss = get_loss_point_ratio(model, i_v, X, y, loss_function, optimizer, 0.0, 1e-2, 0.0, 5)
        
        
        
        
        # if (i_l < l_min):
        #     print("lower i_l: ", i_l)
        #     l_min_prev_dist_abs = l_min_dist_abs
        #     l_min_dist_abs = get_dist(i_v, v_min)
        #     v_min = i_v.clone()
        #     l_min = i_l
        #     epsilon = get_epsilon(model, v_min)
        #     i_halt_condition = get_min_halt_condition(i_Av, l_min, v_min, epsilon)
        #     print("i_halt_condition: ", i_halt_condition)
  
        t8 = time.time()
        # time_elapsed = t8-t7
        # print('LOBLOOP part IV {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        time_elapsed = t8-t4
        print('LOBLOOP iteration running time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        # print("i_r: ",  i_r)
        
        new_row = {'iteration' : i, 'cosin vec': cosin_dist, 'lambda': i_l.item(), 'loss': LOBPCG_loss.item(), 'delta': i_delta.item(), 
                   'halt condition': tmp_halt_condition.item(), 'opt step size': opt_point_LOBPCG, 'loss at opt step size': opt_point_LOBPCG_loss}
        run_stats = pd.concat([run_stats, pd.DataFrame([new_row])], ignore_index = True)
        i += 1
    return v_min, l_min, i, run_stats


def get_LnVn_LOBPCG_Rayleigh_Ritz(model, loss_function, optimizer, X, y, V, preconditioner = None):
    t0 = time.time()
    Q, R = torch.linalg.qr(V.t())
    t1 = time.time()
    # time_elapsed = t1-t0
    # print('torch.linalg.qr in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    model_size = get_model_size(tuple(model.parameters()))
    AV = torch.tensor([]).to(DEVICE)
    Vorth = torch.tensor([]).to(DEVICE)
    i = 0
    for vi in Q.t():
        if abs(R[i][i]) > EM:
            t2 = time.time()

            Avi = get_model_v_fd(model, loss_function, optimizer, X, y, vi, None, preconditioner)
            t3 = time.time()
            time_elapsed = t3-t2
            # print('\t- inner loop get_model_v_fd in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            AV = torch.cat((AV, Avi.reshape(1, model_size)))
            Vorth = torch.cat((Vorth, vi.reshape(1, model_size)))
            i = + 1
            t4 = time.time()
            time_elapsed = t4-t2
            # print('\t- inner loop in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    M = torch.matmul(Vorth, AV.t())
    Ls, Vs = torch.linalg.eig(M)

    if Vs.sum().sum().imag.item() > 0:  # complex eig vecs
        raise("complex eig vecs in get_L0V0_LOBPCG_Rayleigh_Ritz")
    Ls = Ls.real
    Vs = Vs.real
    NV = torch.matmul(Vs, Vorth)

    sorted_L, indices = torch.sort(Ls, dim=0, descending=True, out=None)

    # return argmin [-1]

    Vn = NV[indices[-1]]
    Ln = Ls[indices[-1]]
    tn = time.time()
    time_elapsed = tn-t0
    # print('\t- total time of get_LnVn_LOBPCG_Rayleigh_Ritz in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return Vn, Ln, R[-1][-1]


def get_model_v_fd(model, loss_function, optimizer, X, y, v, lambda_to_substract=None, preconditioner = None):
    model_init_params = model_parametres_to_tensor(model).clone()
    epsilon = get_epsilon(model, v)
    #print("epsilon", epsilon)
    delta = v*epsilon
    delta_size = torch.matmul(delta.t(), delta)**0.5
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








'''
returns the biggest in abs value lambda with it's eig vector
'''
def get_L0V0_finite_differencing(model, loss_function, optimizer, X, y, max_iter_num, is_first=True, lambda_to_sub=0, preconditioner = None):

    df_stats = pd.DataFrame(columns=['iteration', 'cosin vec', 'lambda diff'])

    i_v = torch.ones(get_model_size(tuple(model.parameters()))).to(DEVICE)
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

        i_v = get_model_v_fd(model, loss_function, optimizer, X, y, i_v_prev, lambda_to_sub, preconditioner)


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
    L, V = get_sorted_LV_by_abs_L(A)
    V0 = V[:, 0]
    L0 = L[0]
    L, V = get_sorted_LV(A)
    Vn = V[:, -1]
    Ln = L[-1]
    return V0, L0, Vn, Ln


def get_closest_l_to_givenL(L, l):
    ref_tensor = (L.abs()-l.abs()).abs()
    min_index = torch.argmin(ref_tensor)
    if (ref_tensor[min_index].item() != 0.0):
        return L[min_index].item()
    else:  # found l itself deleting it and researching
        ref_tensor = torch.cat(
            [ref_tensor[0:min_index], ref_tensor[min_index+1:]])
        ref_L = torch.cat([L[0:min_index], L[min_index+1:]])
        min_index = torch.argmin(ref_tensor)
        return ref_L[min_index].item()

def get_model_loss(model, v, step_size, X, y, loss_function, optimizer):
    total_loss = 0.0
    
    model_init_params = model_parametres_to_tensor(model).clone()
    set_model_parameters(model, model_init_params + v*step_size)
    
    train_dataset = TensorDataset(X, y)
    data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)

    for i, (iX, iy) in enumerate(data_loader):
        optimizer.zero_grad() #URI- if not practicing  optimizer.step() do I need to nullify grads here?


        probs = model(iX)
        i_loss = loss_function(probs, iy)
        total_loss += i_loss
        i_loss.backward()

        #optimizer.step() #no parameter update $$$$$$$$$$$$$$$$$$$$$$$
    set_model_parameters(model, model_init_params)
    optimizer.zero_grad()
    return total_loss

def precond_bi_diagonal(v, ratio):
    
    shift_v = torch.zeros(v.shape).to(DEVICE)
    shift_v[:-1] = v[1:]
    precond = (shift_v + v*ratio).clone() #detach
    return precond

def evaluate_smallest_eig_model(filepath):
    
    # MNIST_RESNET18.load_data()
    # MNIST_RESNET18.load_model(filepath)
    
    CIFAR10_RESNET18.load_data()
    CIFAR10_RESNET18.load_model(filepath)
    
    # model.to(DEVICE)
    # run_status_dict = torch.load(filepath)
    
    # model.load_state_dict(run_status_dict['model_state_dict'])
    
    print("number of trainable parameters of model:", CIFAR10_RESNET18.get_model_size())
    print("is cude:", next(CIFAR10_RESNET18.model.parameters()).is_cuda)
    
    model = CIFAR10_RESNET18.model
    loss_function = CIFAR10_RESNET18.criterion
    optimizer = CIFAR10_RESNET18.optimizer
    lr_scheduler = CIFAR10_RESNET18.lr_scheduler
    
    
    data_x, data_y = ((CIFAR10_RESNET18.dataloaders['train']).dataset).datasets[0][:]
    data_x = data_x.to(DEVICE)
    data_y = torch.tensor(data_y).to(DEVICE)
    preconditioner = None
    model_size = get_model_size(model.parameters())
    results_dict = {}
    try:
        step_size = lr_scheduler.get_last_lr()[0]
        print("step_size:", step_size)
        init_loss = get_model_loss(model, torch.zeros(model_size).to(DEVICE), 0.0, data_x, data_y, loss_function, optimizer)
        print("init_loss:", init_loss)
        run_log_PRECOND = pd.DataFrame()
        # Vn_LOBPCG_PRECOND, Ln_LOBPCG_PRECOND, iN_LOBPCG_PRECOND, run_log_PRECOND = get_LnVn_LOBPCG(model, loss_function, optimizer, data_x, data_y, step_size, preconditioner)
        # print("finished LOB PRE COND")
        # LOBPCG_PRECOND_loss = get_model_loss(model, Vn_LOBPCG_PRECOND, step_size, data_x, data_y, loss_function, optimizer)
        # print("LOBPCG_PRECOND loss:", LOBPCG_PRECOND_loss)
        # opt_point_LOBPCG_PRECOND, opt_point_LOBPCG_PRECOND_loss = get_loss_point_ratio(model, Vn_LOBPCG_PRECOND, data_x, data_y, loss_function, optimizer, 0.0, STEP_SIZE, 0.0, 5)
        # print("LOBPCG_PRECOND loss OPT point:", opt_point_LOBPCG_PRECOND_loss)
        
        # Vn_LOBPCG, Ln_LOBPCG, iN_LOBPCG = get_LnVn_LOBPCG(model, loss_function, optimizer, data_x, data_y, step_size, True)
        # print("finished LOB with precond")
        # LOBPCG_loss = get_model_loss(model, Vn_LOBPCG, step_size, data_x, data_y, loss_function, optimizer)
        # print("LOBPCG_loss with precond: ", LOBPCG_loss)
        
        
        Vn_LOBPCG, Ln_LOBPCG, iN_LOBPCG, run_log = get_LnVn_LOBPCG(model, loss_function, optimizer, data_x, data_y, step_size, False)
        print("finished LOB")
        LOBPCG_loss = get_model_loss(model, Vn_LOBPCG, step_size, data_x, data_y, loss_function, optimizer)
        print("LOBPCG_loss:", LOBPCG_loss)
        opt_point_LOBPCG, opt_point_LOBPCG_loss = get_loss_point_ratio(model, Vn_LOBPCG, data_x, data_y, loss_function, optimizer, 0.0, STEP_SIZE, 0.0, 5)
        print("LOBPCG loss OPT point:", opt_point_LOBPCG_loss)
        
        # V0_PRECOND, L0_PRECOND, dist_converged_0_PRECOND, dist_converged_stats0_PRECOND, Vn_PRECOND, Ln_PRECOND, dist_converged_n_PRECOND, dist_converged_statsn_PRECOND, i0_PRECOND, iN_PRECOND, second_iter_Ln_PRECOND = get_V0L0_VnLn_finite_differencing(
        #     model, loss_function, optimizer, data_x, data_y, MAX_ITER_NUM, preconditioner, preconditioner)
        # V0, L0, dist_converged_0, dist_converged_stats0, Vn, Ln, dist_converged_n, dist_converged_statsn, i0, iN, second_iter_Ln = get_V0L0_VnLn_finite_differencing(
        #      model, loss_function, optimizer, data_x, data_y, MAX_ITER_NUM, None, None)
        # print("finished PM ")
        # PM_loss = get_model_loss(model, Vn, step_size, data_x, data_y, loss_function, optimizer)
        # print("PM_loss:", PM_loss)
        # # V0_Lanczos_PRECOND, L0_Lanczos_PRECOND, dist_converged_0_Lanczos_PRECOND, converge_stats0_Lanczos_PRECOND, Vn_Lanczos_PRECOND, Ln_Lanczos_PRECOND, dist_converged_n_Lanczos_PRECOND, converge_statsn_Lanczos_PRECOND, i0_Lanczos_PRECOND, iN_Lanczos_PRECOND, i_max0_PRECOND, i_maxN_PRECOND, second_iter_Ln_Lanczos_PRECOND = get_V0L0_VnLn_Lanczos(
        # #     model, loss_function, optimizer, data_x, data_y, 0, preconditioner,preconditioner)
        # V0_Lanczos, L0_Lanczos, dist_converged_0_Lanczos, converge_stats0_Lanczos, Vn_Lanczos, Ln_Lanczos, dist_converged_n_Lanczos, converge_statsn_Lanczos, i0_Lanczos, iN_Lanczos, i_max0, i_maxN, second_iter_Ln_Lanczos = get_V0L0_VnLn_Lanczos(
        #     model, loss_function, optimizer, data_x, data_y, 0, None, None)
        # print("finished Lanczos")
        # Lanczos_loss = get_model_loss(model, Vn_Lanczos, step_size, data_x, data_y, loss_function, optimizer)
        # print("Lanczos_loss:", Lanczos_loss)
        # V0_Lanczos, L0_Lanczos, dist_converged_0_Lanczos, dist_converged_stats0_Lanczos, Vn_Lanczos, Ln_Lanczos, dist_converged_n_Lanczos, dist_converged_statsn_Lanczos, i0_Lanczos, iN_Lanczos, second_iter_Ln_Lanczos  = get_V0L0_VnLn_Lanczos(model, loss_function, optimizer, data_x, data_y, MAX_ITER_NUM)

        
        
        results_dict = {"Model size" : model_size, 
                        "Model loss" : init_loss.item(),
                        "LOBPCG #iteration" :  iN_LOBPCG,
                        "LOBPCG lambda" : Ln_LOBPCG.item(),
                        "LOBPCG loss" : LOBPCG_loss.item(),
                        "LOBPCG loss opt point" : opt_point_LOBPCG_loss.item(),
                        "LOBPCG opt point" : opt_point_LOBPCG,
                        # "LOBPCG #iteration precond" : iN_LOBPCG_PRECOND,
                        # "LOBPCG lambda precond" : Ln_LOBPCG_PRECOND.item(),
                        # "LOBPCG loss precond" : LOBPCG_PRECOND_loss.item(),
                        # "LOBPCG loss opt point precond" : opt_point_LOBPCG_PRECOND_loss.item(),
                        # "LOBPCG opt point precond" : opt_point_LOBPCG_PRECOND,
                        }
        
        

    except:
        print("An exception occurred in eigs calculation")
       
    finally:

        # if (to_save):
        #     if (is_comparable):
        #         if (correct_cosin):
        #             save_dir = save_dir + "\\coverged_to_right_vec"
        #         else:
        #             if (convereged):
        #                 save_dir = save_dir + "\\coverged_to_wrong_vec"
        #             else:
        #                 save_dir = save_dir + "\\did_not_coverge"
        #     else:
        #         save_dir = save_dir + "\\incomperable"
        # save_dir = save_dir + "\\" + str(iteration_number)
        # if (not os.path.exists(save_dir)):
        #     os.makedirs(save_dir)
        # save_dir = save_dir + "\\model.pth"
        # torch.save(model, save_dir)

        return results_dict, run_log_PRECOND, run_log





def main():
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
    
        with pd.ExcelWriter(OUT_PATH_SUMMARY + "\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+ "_summary_res.xlsx") as writer:
            overview_run_stats.to_excel(writer, sheet_name='data_summary')
            runs_precond_stats.to_excel(writer, sheet_name='runs_precond_stats')
            runs_stats.to_excel(writer, sheet_name='runs_stats')
if __name__ == '__main__':
    main()

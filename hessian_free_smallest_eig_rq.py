# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 16:18:02 2024

@author: dafna
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 09:08:37 2023

@author: dafna
"""


import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.func import hessian, grad
import torch.nn.functional as F

from torch.func import functional_call
from torch.utils.data import DataLoader, TensorDataset

from torchvision import datasets
from torchvision import transforms

import pickle

import pandas as pd
import os

import copy


# GLOBALS
LR = 0.01  # 0.0001
MAX_ITER_NUM = 1e+02
COSIN_DIST = 1e-03
MAX_DIST_CONVERGENCE = 1e-07
EM = 1e-07  # machine precision
LANCZOS_DIM = 1e+01
N_HIDDEN = 3
MAX_SIZE_CALCULATED_H = 100

DROP_PROB = 0.3

N_HIDDEN = 20
OUTPUT_DIM = 10
IMAGE_SIZE = [28,28]
INPUT_DIM = IMAGE_SIZE[0]*IMAGE_SIZE[1]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 512

MODEL_PATH = r'C:\Users\mirac\Documents\OffSaddle\runs\MNIST_LINEAR_HESSIAN_TRY\ADAMW_full_classifier\20240211-180807n_hidden=20_lr=0.0001_gamma=0.995'
# r'C:\Users\mirac\Documents\OffSaddle\runs\MNIST_LINEAR_HESSIAN_TRY\ADAMW_full_classifier\20240210-222453n_hidden=20_lr=0.0001_gamma=0.998'
# r'C:/Users/dafna/Documents/Hassian-Free-Proj/Hassian-Free/runs/MNIST/full_classifier/20240121-203002/'
MODEL_FILENAME = '4126_model_loss=0.94425_grad=1.18859_lr=0.00000_train_acc=0.94933_test_acc=0.94425.pth'
# '826_model_loss=0.95158_grad=0.70121_lr=0.00002_train_acc=0.97181_test_acc=0.95158.pth'
# '54_model_l=0.99025_g=0.15720.pth'

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

class MNIST_Classifier(nn.Module):
    def __init__(self):
        super(MNIST_Classifier, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=DROP_PROB))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=DROP_PROB))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(DROP_PROB))
        self.layer4 = nn.Sequential(
            nn.Linear(3200, 1024, bias=True),
            nn.LeakyReLU(),
            nn.Dropout(p=DROP_PROB))
        self.layer5 = nn.Linear(1024, OUTPUT_DIM, bias=True)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)   # Flatten them for layer4
        out = self.layer4(out)
        out = self.layer5(out)
        #out = nn.Softmax()(out)
        return out

class MNIST_AE(torch.nn.Module):
	def __init__(self):
		super().__init__()
		
		# Building an linear encoder with Linear
		# layer followed by Relu activation function
		# 784 ==> 9
		self.encoder = torch.nn.Sequential(
			torch.nn.Linear(28 * 28, 128),
			torch.nn.ReLU(),
			torch.nn.Linear(128, 64),
			torch.nn.ReLU(),
			torch.nn.Linear(64, 36),
			torch.nn.ReLU(),
			torch.nn.Linear(36, 18),
			torch.nn.ReLU(),
			torch.nn.Linear(18, 9)
		)
		
		# Building an linear decoder with Linear
		# layer followed by Relu activation function
		# The Sigmoid activation function
		# outputs the value between 0 and 1
		# 9 ==> 784
		self.decoder = torch.nn.Sequential(
			torch.nn.Linear(9, 18),
			torch.nn.ReLU(),
			torch.nn.Linear(18, 36),
			torch.nn.ReLU(),
			torch.nn.Linear(36, 64),
			torch.nn.ReLU(),
			torch.nn.Linear(64, 128),
			torch.nn.ReLU(),
			torch.nn.Linear(128, 28 * 28),
			torch.nn.Sigmoid()
		)

	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded


class Simple_NN(nn.Module):
    def __init__(self):
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
    X = X.reshape(-1, IMAGE_SIZE[0]*IMAGE_SIZE[1])

    
    train_dataset = TensorDataset(X, y)
    data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # initialization of the gradients    
    
    total_grads = torch.zeros(get_model_size(model.parameters())).to(DEVICE)
    
    # #MNIST_AE
    # for i,(image,y) in enumerate(data_loader):
       
    #   image = image.reshape(-1, 28*28).float()
       
    #   # Output of Autoencoder
    #   reconstructed = model(image)
       
       
    #   optimizer.zero_grad()
      
    #   loss = loss_function(reconstructed, image)

    #   loss.backward()
      
    #   iGrad = get_model_grads_as_tensor(model)
    #   total_grads = total_grads + iGrad
    
    
    
    
    
    # MNIST
    for i, (iX, iy) in enumerate(data_loader):
        optimizer.zero_grad() 


        # forward + backward + optimize

        probs = model(iX).float()
        y_max_indices = torch.max(iy, 1)[1]
        loss = loss_function(probs, y_max_indices)
        loss.backward()
        iGrads = get_model_grads_as_tensor(model)
        total_grads = total_grads + iGrads

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


def get_model_v_fd(model, loss_function, optimizer, X, y, v, lambda_to_substract=None):
    model_init_params = model_parametres_to_tensor(model).clone()
    epsilon = get_epsilon(model, v)
    #print("epsilon", epsilon)
    delta = v.to(DEVICE)*epsilon
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
    if lambda_to_substract is None:
        return nv
    else:
        return lambda_to_substract*v - nv


def get_V0L0_VnLn_Lanczos(model, loss_function, optimizer, X, y, lambda_to_sub=0):
    print("Lanczos first iteration")
    V0, L0, dist_converged_0, converge_stats0, i0 = get_L0V0_Lanczos(
        model, loss_function, optimizer, X, y, lambda_to_sub=None)
    # converge_stats0["type"] = "0"
    print("Lanczos second iteration")
    Vn, second_iter_Ln, dist_converged_n, converge_statsn, iN = get_L0V0_Lanczos(
        model, loss_function, optimizer, X, y, lambda_to_sub=L0)
    # converge_statsn["type"] = "n"
    Ln = L0.abs() - second_iter_Ln
    return V0, L0, dist_converged_0, converge_stats0, Vn, Ln, dist_converged_n, converge_statsn, i0, iN, second_iter_Ln


def get_L0V0_Lanczos(model, loss_function, optimizer, X, y, lambda_to_sub=0):
    converge_stats = {}
    model_init_params = model_parametres_to_tensor(model).clone()
    i_model_params = model_init_params.clone()
    i_v = torch.ones(get_model_size(tuple(model.parameters()))).to(DEVICE)
    i_v = i_v/torch.linalg.vector_norm(i_v, ord=2)
    i_v_prev = i_v.clone()
    i_w_tag = get_model_v_fd(model, loss_function,
                             optimizer, X, y, i_v, lambda_to_sub)
    i_a = i_w_tag*i_v
    i_w = i_w_tag - i_a*i_v
    v_max = i_v.clone()
    # l_min = (i_w_tag/i_v).nanmean() # changed to r(x)
    l_max = torch.matmul(i_w_tag, i_v.t()) / torch.matmul(i_v.t(), i_v)
    cos_dist_at_convergence = 1.0
    i = 2
    while (i < LANCZOS_DIM) and (abs(cos_dist_at_convergence) > COSIN_DIST):
        beta_i = torch.matmul(i_w.t(), i_w)**0.5
        if beta_i > 0:
            i_v = i_w/beta_i
            i_w_t = get_model_v_fd(model, loss_function,
                                   optimizer, X, y, i_v, lambda_to_sub)
            i_a = i_w_t*i_v
            i_w = i_w_t - i_a*i_v - beta_i*i_v_prev
            i_v_prev = i_v.clone()
            # i_l = (i_w_t/i_v).nanmean() # changed to r(x)
            i_l = torch.matmul(i_w_t, i_v.t()) / torch.matmul(i_v.t(), i_v)
            if (i_l > l_max):
                cos_dist_at_convergence = get_dist(i_v, v_max)
                print("lanczos improve at iteration: {} dist: {} new lambda: {}".format(
                    i, cos_dist_at_convergence, l_max))
                v_max = i_v.clone()
                l_max = i_l

        else:
            print("ERROR beta_i")
        i += 1

    return v_max, l_max, cos_dist_at_convergence, converge_stats, i


def get_LnVn_LOBPCG(model, loss_function, optimizer, X, y, use_preconditionaer_lambda = False):
    model_init_params = model_parametres_to_tensor(model).clone()
    i_model_params = model_init_params.clone()
    model_size = get_model_size(tuple(model.parameters()))
    i_v = torch.ones(model_size).to(DEVICE)
    i_v = i_v/torch.linalg.vector_norm(i_v, ord=2)
    i_v_prev = i_v.clone()
    i_Av = get_model_v_fd(model, loss_function, optimizer, X, y, i_v, None)
    i_w = i_Av - torch.matmul(i_v.t(), i_Av) * i_v  # A*v -v'Av*v
    i_w = i_w/torch.linalg.vector_norm(i_w, ord=2)
    V = torch.cat((i_v.reshape(1, model_size), i_w.reshape(1, model_size)))
    i_v, i_l, i_r = get_LnVn_LOBPCG_Rayleigh_Ritz(
        model, loss_function, optimizer, X, y, V)
    i_v = i_v/torch.linalg.vector_norm(i_v, ord=2)
    i = 0
    cosin_dist = get_dist(i_v, i_v_prev)
    cosin_dist_prev = 0
    v_min = i_v.clone()
    l_min = torch.matmul(i_Av, i_v.t()) / torch.matmul(i_v.t(), i_v)
    l_min_dist_abs = abs(cosin_dist)
    l_min_prev_dist_abs = 0
    while (i < MAX_ITER_NUM) and (abs(abs(cosin_dist) - abs(cosin_dist_prev)) > COSIN_DIST) and (abs(abs(cosin_dist)-1) > COSIN_DIST):
        V = torch.cat((i_v.reshape(1, model_size), i_w.reshape(
            1, model_size), i_v_prev.reshape(1, model_size)))
        i_v_prev = i_v.clone()
        i_v, i_l, i_r = get_LnVn_LOBPCG_Rayleigh_Ritz(
            model, loss_function, optimizer, X, y, V)
        i_v = i_v/torch.linalg.vector_norm(i_v, ord=2)
        i_Av = get_model_v_fd(model, loss_function, optimizer, X, y, i_v, None)
        i_r = i_Av - torch.matmul(i_v.t(), i_Av) * i_v  # A*v -v'Av*v
        i_r = i_r/torch.linalg.vector_norm(i_r, ord=2)
        i_w = i_r.clone()
        if (use_preconditionaer_lambda):
            i_w = get_model_v_fd(model, loss_function, optimizer, X, y, i_r, None) - l_min*i_r
            
        cosin_dist_prev = cosin_dist
        cosin_dist = get_dist(i_v, i_v_prev)
        i_l = torch.matmul(i_Av, i_v.t()) / torch.matmul(i_v.t(), i_v)
        if (i_l < l_min):
            l_min_prev_dist_abs = l_min_dist_abs
            l_min_dist_abs = get_dist(i_v, v_min)
            print("lobpcg improve at iteration: {} dist: {} new lambda: {}".format(
                i,  get_dist(i_v, v_min), l_min))

            v_min = i_v.clone()
            l_min = i_l
        print("cosin_dist: ",  cosin_dist)
        # print("i_r: ",  i_r)
        i += 1
    return v_min, l_min, i


def get_LnVn_LOBPCG_Rayleigh_Ritz(model, loss_function, optimizer, X, y, V):
    Q, R = torch.linalg.qr(V.t())
    model_size = get_model_size(tuple(model.parameters()))
    AV = torch.tensor([]).to(DEVICE)
    Vorth = torch.tensor([]).to(DEVICE)
    i = 0
    for vi in Q.t():
        if abs(R[i][i]) > EM:
            Avi = get_model_v_fd(model, loss_function,
                                 optimizer, X, y, vi, None)
            AV = torch.cat((AV, Avi.reshape(1, model_size)))
            Vorth = torch.cat((Vorth, vi.reshape(1, model_size)))
            i = + 1
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

    return Vn, Ln, R[-1][-1]


'''
returns the biggest in abs value lambda with it's eig vector
'''


def get_L0V0_finite_differencing(model, loss_function, optimizer, X, y, max_iter_num, is_first=True, lambda_to_sub=0):

    df_stats = pd.DataFrame(columns=['iteration', 'cosin vec', 'lambda diff'])

    i_v = torch.ones(get_model_size(tuple(model.parameters()))).to(DEVICE)
    i_v_size = torch.matmul(i_v.t(), i_v)**0.5
    i_v = i_v / i_v_size

    # func, func_params = functorch.make_functional(model)
    def model_loss_sum_params(params, X, data_y):
        #pred_y: torch.Tensor = _stateless.functional_call(model, {n: p for n, p in zip(names, params)}, X)
        pred_y = functional_call(model, params, X)
        loss = loss_function(pred_y, y)
        return loss.sum()

    prev_cosin_dist = -1
    cosin_dist_diff = 1/MAX_DIST_CONVERGENCE
    cosin_dist = 1/MAX_DIST_CONVERGENCE
    i_l = 1/MAX_DIST_CONVERGENCE

    model_init_params = model_parametres_to_tensor(model).clone()
    i_model_params = model_init_params.clone()
    i_v_norm = 1
    i_l_prev = -1
    i = 0
    # halting at the sooner of (max_iteration,no change between two consecutive iterations)
    while ((i < MAX_ITER_NUM) and (abs(cosin_dist_diff) > COSIN_DIST)):
        # (abs(i_l_prev - i_l) >= MAX_DIST_CONVERGENCE)):
        # (prev_dist != dist) and (abs(abs(dist)-1) >= MAX_DIST_CONVERGENCE) ):
        prev_cosin_dist = cosin_dist
        i_v_prev = i_v.clone()
        i_l_prev = i_l

        # df2(x)v =(df(x + espilon*d) − ∇f(x − espilon*d))/2*espilon
        epsilon = get_epsilon(model, i_v_prev)
        #print("epsilon", epsilon)
        delta = i_v_prev*epsilon
        delta_size = torch.matmul(delta.t(), delta)**0.5
        plus_eps_i_v = model_init_params + delta
        minus_eps_i_v = model_init_params - delta

        grad_plus_eps_niv = get_model_grads_at_set_parameters(
            model, plus_eps_i_v, X, y, loss_function, optimizer)
        reset_model(model, model_init_params, optimizer)  # back to init
        grad_minus_eps_niv = get_model_grads_at_set_parameters(
            model, minus_eps_i_v, X, y, loss_function, optimizer)
        reset_model(model, model_init_params, optimizer)  # back to init

        if (is_first):
            i_v = (grad_plus_eps_niv - grad_minus_eps_niv)/(2*delta_size)
        else:
            a = abs(lambda_to_sub)*i_v_prev
            b = (grad_plus_eps_niv - grad_minus_eps_niv)/(2*delta_size)
            i_v = a - b

        i_v = i_v.type(torch.float)
        # i_l -> (i_v_prev.t()*i_v)/((i_v_prev.t()*i_v_prev)) since i_v_prev is normalized denominator is always 1
        i_l = torch.matmul(i_v_prev.t(), i_v)
        i_v_norm = torch.matmul(i_v.t(), i_v)**0.5
        i_v = i_v / i_v_norm

        
        i = i + 1
        cosin_dist = get_dist(i_v, i_v_prev)

        cosin_dist_diff = abs(abs(cosin_dist) - abs(prev_cosin_dist))
        # new_row = {'iteration': i, 'cosin vec': cosin_dist,
        #            'lambda i': i_l.item()}
        # df_stats = df_stats.append(new_row, ignore_index=True)


    i_v = torch.reshape(i_v, (-1,))

    reset_model(model, model_init_params, optimizer)
    return i_l, i_v, cosin_dist, df_stats, i


def get_V0L0_VnLn_finite_differencing(model, loss_function, optimizer, X, y, max_iter_num):
    L0, V0, dist_converged_0, converge_stats0, i0 = get_L0V0_finite_differencing(
        model, loss_function, optimizer, X, y, max_iter_num, is_first=True, lambda_to_sub=0)
    converge_stats0["type"] = "0"
    second_iter_Ln, Vn, dist_converged_n, converge_statsn, iN = get_L0V0_finite_differencing(
        model, loss_function, optimizer, X, y, max_iter_num, is_first=False, lambda_to_sub=L0)
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







def evaluate_run_over_simple_nn_model(to_save, save_dir, iteration_number):

    model = Simple_NN()

    #data_x = torch.tensor([[1.0], [2.0], [3.0]])
    data_x = torch.normal(0.0, 1, size=(10000, 1))
    with torch.no_grad():
        data_y = model(data_x)+torch.normal(0.0, 10, size=(10000, 1))

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
            HM = (HM + HM.t())/2
        #print("L0 by finite differencing: ", L0)
        #HM_condition_number = torch.linalg.cond(HM)
        #print("condition number of hessian: ", HM_condition_number)
        V0_HM, L0_HM, Vn_HM, Ln_HM = get_V0L0_VnLn(HM)
        L, V = get_sorted_LV_by_abs_L(HM)

        Vn_LOBPCG, Ln_LOBPCG, iN_LOBPCG = get_LnVn_LOBPCG(
            model, loss_function, optimizer, data_x, data_y)

        V0, L0, dist_converged_0, dist_converged_stats0, Vn, Ln, dist_converged_n, dist_converged_statsn, i0, iN, second_iter_Ln = get_V0L0_VnLn_finite_differencing(
            model, loss_function, optimizer, data_x, data_y, MAX_ITER_NUM)

        V0_Lanczos, L0_Lanczos, dist_converged_0_Lanczos, converge_stats0_Lanczos, Vn_Lanczos, Ln_Lanczos, dist_converged_n_Lanczos, converge_statsn_Lanczos, i0_Lanczos, iN_Lanczos, second_iter_Ln_Lanczos = get_V0L0_VnLn_Lanczos(
            model, loss_function, optimizer, data_x, data_y, 0)

        #V0_Lanczos, L0_Lanczos, dist_converged_0_Lanczos, dist_converged_stats0_Lanczos, Vn_Lanczos, Ln_Lanczos, dist_converged_n_Lanczos, dist_converged_statsn_Lanczos, i0_Lanczos, iN_Lanczos, second_iter_Ln_Lanczos  = get_V0L0_VnLn_Lanczos(model, loss_function, optimizer, data_x, data_y, MAX_ITER_NUM)

        #err0_L = torch.matmul(HM, V0_Lanczos) - V0_Lanczos*L0_Lanczos
        #print("Lanczos 0 err: {} after {} iterations".format( torch.linalg.vector_norm(err0_L, ord =2), i0_Lanczos))
        err0_PM = torch.matmul(HM, V0) - V0*L0
        print("Power method 0 err: {} after {} iterations".format(
            torch.linalg.vector_norm(err0_PM, ord=2), i0))
        err0_L = torch.matmul(HM, V0_Lanczos) - V0_Lanczos*L0_Lanczos
        print("Lanczos 0 err: {} after {} iterations".format(
            torch.linalg.vector_norm(err0_L, ord=2), i0_Lanczos))

        print("Target Lambda: {}".format(L.min()))
        errN_LOB = torch.matmul(HM, Vn_LOBPCG) - Vn_LOBPCG*Ln_LOBPCG
        print("LOBPCG N err: {} after {} iterations. Lambda: {}".format(
            torch.linalg.vector_norm(errN_LOB, ord=2), iN_LOBPCG, Ln_LOBPCG))

        errN_L = torch.matmul(HM, Vn_Lanczos) - Vn_Lanczos*Ln_Lanczos
        print("Lanczos N err: {} after {} iterations (i0={} + iN={}). Lambda: {}".format(
            torch.linalg.vector_norm(errN_L, ord=2), iN_Lanczos + i0_Lanczos, i0_Lanczos, iN_Lanczos, Ln_Lanczos))

        # err0_L = torch.matmul(HM, V0_Lanczos) - V0_Lanczos*L0_Lanczos
        # print("Lanczos N err: {} after {} iterations. Lambda: {}".format(torch.linalg.vector_norm(err0_L, ord =2), i0_Lanczos*LANCZOS_DIM, L0_Lanczos))

        errN_PM = torch.matmul(HM, Vn) - Vn*Ln
        print("Power method N err: {} after {} iterations (i0={} + iN={}). Lambda: {}".format(
            torch.linalg.vector_norm(errN_PM, ord=2), iN+i0, i0, iN, Ln))

        HM2_ver_FD = abs(L0)*torch.eye(HM.shape[0], HM.shape[0]) - HM
        HM2_ver_H = abs(L0_HM)*torch.eye(HM.shape[0], HM.shape[0]) - HM

        LL2_ver_FD, VV2_ver_FD = get_sorted_LV(HM2_ver_FD)

        HM2_verFD_LL0 = L0.abs() - LL2_ver_FD[0]

        LL2_ver_H, VV2_ver_H = get_sorted_LV(HM2_ver_H)
        HM2_ver_H_LL0 = L0.abs() - LL2_ver_H[0]

        cosin_dist = get_dist(V0, V0_HM)

        if (abs(cosin_dist) < 0.9):
            correct_cosin = False
        if ((i0 >= MAX_ITER_NUM - 1) or (iN >= MAX_ITER_NUM - 1)):
            convereged = False

        L, V = get_sorted_LV_by_abs_L(HM)
        closest_l_to_l0 = get_closest_l_to_givenL(L, L0_HM)
        closest_l_to_ln = get_closest_l_to_givenL(L, Ln_HM)

        data = {"n hidden": [N_HIDDEN],
                "model size": [HM.shape[0]],
                "FD L0": [L0.item()],
                "True L[0]": [L0_HM.item()],
                "dist_converged_0": [dist_converged_0],
                "number of iterations 0": [i0],
                "Calculated Ln (twice FD)": [Ln.item()],
                "True L[n] of H ": [Ln_HM.item()],
                "Ln = (FD_L[0]*I - HM)": [LL2_ver_FD[0].item()],
                "Ln = (HM_L[0]*I - HM)": [LL2_ver_H[0].item()],
                "Ln = twice FD": [second_iter_Ln.item()],
                "number of iterations n": [iN],
                }
        overview_run_stats = pd.DataFrame(data)

        dist_converged_stats = pd.concat(
            [dist_converged_stats0, dist_converged_statsn])
        dist_converged_stats["n hidden"] = N_HIDDEN
        dist_converged_stats["H size"] = HM.shape[0]
        dist_converged_stats["L0"] = L0_HM.item()
        dist_converged_stats["L0-ref"] = closest_l_to_l0
        dist_converged_stats["Ln"] = Ln_HM.item()
        dist_converged_stats["Ln-ref"] = closest_l_to_ln

    except:
        print("An exception occurred in pytorch's H calculation")
        is_comparable = False

        data = {"n hidden": [N_HIDDEN],
                "model size": [get_model_size(model_parameters)],
                "Calculated L0": [L0.item()],
                "True L[0]": ["Err"],
                "True closest L to l[0]": ["Err"],
                "dist_converged_0": ["Err"],
                "number of iterations 0": [i0],
                "cosin 0": ["Err"],
                "Calculated Ln": [Ln.item()],
                "True L[n]": ["Err"],
                "True closest L to l[n-1]": ["Err"],
                "dist_converged_n": ["Err"],
                "number of iterations n": [iN],
                "cosin n": ["Err"],
                }
        overview_run_stats = pd.DataFrame(data)

        dist_converged_stats = pd.concat(
            [dist_converged_stats0, dist_converged_statsn])
        dist_converged_stats["n hidden"] = N_HIDDEN
        dist_converged_stats["H size"] = get_model_size(model_parameters)
        dist_converged_stats["L0"] = "Err in calculating explicit H"
        dist_converged_stats["L0-ref"] = "Err in calculating explicit H"
        dist_converged_stats["Ln"] = "Err in calculating explicit H"
        dist_converged_stats["Ln-ref"] = "Err in calculating explicit H"
    finally:

        if (to_save):
            if (is_comparable):
                if (correct_cosin):
                    save_dir = save_dir + "\\coverged_to_right_vec"
                else:
                    if (convereged):
                        save_dir = save_dir + "\\coverged_to_wrong_vec"
                    else:
                        save_dir = save_dir + "\\did_not_coverge"
            else:
                save_dir = save_dir + "\\incomperable"
        save_dir = save_dir + "\\" + str(iteration_number)
        if (not os.path.exists(save_dir)):
            os.makedirs(save_dir)
        save_dir = save_dir + "\\model.pth"
        torch.save(model, save_dir)

        return overview_run_stats, dist_converged_stats




def get_L0V0_power_method(A, max_iter_num):
    niv = torch.ones(A.shape[0], 1).to(DEVICE)
    niv_size = torch.matmul(niv.t(), niv)**0.5
    niv = niv / niv_size
    i = 0

    converged = False
    dist = 1/MAX_DIST_CONVERGENCE
    liv = 1/MAX_DIST_CONVERGENCE
    df_stats = pd.DataFrame(columns=['iteration', 'cosin vec', 'lambda diff'])
    convereged_by_eps = False
    while ((not convereged_by_eps) and (i < max_iter_num) and (abs(abs(dist)-1) >= MAX_DIST_CONVERGENCE)):
        niv_prev = niv.clone()  # dettach
        liv_prev = liv
        # TO REPLACE WITH df2(x)v =(df(x + espilon*d) − ∇f(x − espilon*d))/2*espilon
        niv = torch.matmul(A, niv_prev)
        niv = niv.type(torch.float)

        nominator = torch.matmul(niv_prev.t(), niv)
        #torch.matmul(torch.transpose(niv_prev, -1,0),niv)[0]
        denominator = torch.matmul(niv_prev.t(), niv_prev)
        #torch.matmul(torch.transpose(niv_prev,  -1, 0 ),niv_prev)[0]
        if (denominator == 0):
            raise Exception("Eig val cannot be inf")

        liv = nominator/denominator
        niv = niv / torch.matmul(niv.t(), niv)**0.5

        i = i + 1
        dist = get_dist(niv, niv_prev)
        lambda_dist = liv - liv_prev

        d = torch.matmul(A, niv) - niv*liv
        lambda_distance = torch.linalg.vector_norm(d)
        if (lambda_distance) < ((liv**2)**0.5):
            # convereged_by_eps = True #for lambda distance
            # for ignoring lambda, only by cosin distance
            convereged_by_eps = convereged_by_eps
        new_row = {'iteration': i, 'cosin vec': dist, 'lambda i': liv.item(), "||A*v-l*v||": lambda_distance.item(),
                   'lambda diff': lambda_dist.item()}
        df_stats = df_stats.append(new_row, ignore_index=True)
    print("finished converging at: ", i)
    print("with dist calculated as abs(abs(dist)-1): " + str(abs(abs(dist)-1)))

    niv = torch.reshape(niv, (-1,))
    return liv, niv, dist, df_stats



def get_boundry_criteria_of_true_eig_val_distance(model, v, l, loss_function, optimizer, X, y):
    # calculating (||(HV)approx-(Lambda)approx*v ||**2 + epsilon**2)**0.5
    epsilon = get_epsilon(model,v)
    Av = get_model_v_fd(model, loss_function, optimizer, X, y, v, 0.0)
    delta = torch.linalg.norm(Av-l*v)
    bound = (delta**2 + epsilon**2 )**0.5
    return bound

def find_smallest_eig_mnist_model(filepath, filename):
    print(filename)
    with open(filepath + '\\data.pkl', 'rb') as f:
        d = pickle.load(f)
    (X_train, X_val, X_test, y_train, y_val, y_test) = d
    
    
    X_train = X_train.reshape(-1, IMAGE_SIZE[0]*IMAGE_SIZE[1]).to(DEVICE)
    y_train = y_train.to(DEVICE)
    
    model = Simple_NN_MNIST().to(DEVICE)
    run_status_dict = torch.load(filepath + "\\" + filename)
    model.load_state_dict(run_status_dict['model_state_dict'])

    loss_function = nn.CrossEntropyLoss().to(DEVICE)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.000587)
    
    def model_loss_sum_params(params, x, y):
        # pred_y: torch.Tensor = _stateless.functional_call(model, {n: p for n, p in zip(names, params)}, data_x)
        pred_y = functional_call(model, params, x)
        loss = loss_function(pred_y, y.float())
        return loss.sum()

    model_parameters = tuple(model.parameters())

    data_x = X_train[0:BATCH_SIZE, :]
    data_y = y_train[0:BATCH_SIZE, :]
    print("Calculating H")
    H = hessian(model_loss_sum_params)(
        dict(model.named_parameters()), data_x, data_y)

    # reshaping hessian to nXn matrix
    HM = reshape_hessian(H, model_parameters)
    # HM should be symmetrical matrix. If it isn't (due to nummerical instability) and in order to avoid complex eig vectors:
    diff = HM - HM.t()
    if (diff.max() > 0):
        HM = (HM + HM.t())/2
    #print("L0 by finite differencing: ", L0)
    #HM_condition_number = torch.linalg.cond(HM)
    #print("condition number of hessian: ", HM_condition_number)
    print("getting v,l")
    V0_HM, L0_HM, Vn_HM, Ln_HM = get_V0L0_VnLn(HM)
    L, V = get_sorted_LV_by_abs_L(HM)
    print("Target Lambda: {} ".format(Ln_HM))
    #optimizer.load_state_dict(run_status_dict['optimizer_state_dict'])
    Vn_LOBPCG_PRECOND, Ln_LOBPCG_PRECOND, iN_LOBPCG_PRECOND = get_LnVn_LOBPCG(
        model, loss_function, optimizer, X_train, y_train, True)

    Vn_LOBPCG, Ln_LOBPCG, iN_LOBPCG = get_LnVn_LOBPCG(
        model, loss_function, optimizer, X_train, y_train, False)
    
    
    print("PRECOND LOBPCG N Lambda: {} after {} iterations.".format(Ln_LOBPCG_PRECOND, iN_LOBPCG_PRECOND))
    print("LOBPCG N Lambda: {} after {} iterations.".format(Ln_LOBPCG, iN_LOBPCG))

    V0, L0, dist_converged_0, dist_converged_stats0, Vn, Ln, dist_converged_n, dist_converged_statsn, i0, iN, second_iter_Ln = get_V0L0_VnLn_finite_differencing(
        model, loss_function, optimizer, X_train, y_train, MAX_ITER_NUM)

    V0_Lanczos, L0_Lanczos, dist_converged_0_Lanczos, converge_stats0_Lanczos, Vn_Lanczos, Ln_Lanczos, dist_converged_n_Lanczos, converge_statsn_Lanczos, i0_Lanczos, iN_Lanczos, second_iter_Ln_Lanczos = get_V0L0_VnLn_Lanczos(
        model, loss_function, optimizer, X_train, y_train, 0)

    print("Power method 0 lambda: {} after {} iterations.".format(L0, i0))
    print("Lanczos 0 lambda: {} after {} iterations.".format(L0_Lanczos, i0_Lanczos))

    print("LOBPCG N Lambda: {} after {} iterations.".format(Ln_LOBPCG, iN_LOBPCG))

    print("Lanczos N Lambda: {} after {} iterations (i0={} + iN={}).".format(
        Ln_Lanczos, iN_Lanczos + i0_Lanczos, i0_Lanczos, iN_Lanczos))

    print("Power method N Lambda: {} after {} iterations (i0={} + iN={}).".format(Ln, iN+i0, i0, iN))


def find_smallest_eig_mnist_ae_model(filepath, filename):
    tensor_transform = transforms.ToTensor()
    dataset = datasets.MNIST(root = "./data", train = True, download = True, transform = tensor_transform)
 

    model = MNIST_AE()
    run_status_dict = torch.load(filepath + filename)
    model.load_state_dict(run_status_dict['model_state_dict'])

    loss_function = torch.nn.MSELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
                                #LR)
    #optimizer.load_state_dict(run_status_dict['optimizer_state_dict'])

    Vn_LOBPCG, Ln_LOBPCG, iN_LOBPCG = get_LnVn_LOBPCG(
        model, loss_function, optimizer, dataset.data, dataset.data)

    V0, L0, dist_converged_0, dist_converged_stats0, Vn, Ln, dist_converged_n, dist_converged_statsn, i0, iN, second_iter_Ln = get_V0L0_VnLn_finite_differencing(
        model, loss_function, optimizer, dataset.data, dataset.data, MAX_ITER_NUM)

    V0_Lanczos, L0_Lanczos, dist_converged_0_Lanczos, converge_stats0_Lanczos, Vn_Lanczos, Ln_Lanczos, dist_converged_n_Lanczos, converge_statsn_Lanczos, i0_Lanczos, iN_Lanczos, second_iter_Ln_Lanczos = get_V0L0_VnLn_Lanczos(
        model, loss_function, optimizer, dataset.data, dataset.data, 0)

    print("Power method 0 lambda: {} after {} iterations.".format(L0, i0))
    print("Lanczos 0 lambda: {} after {} iterations.".format(L0_Lanczos, i0_Lanczos))

    print("LOBPCG N Lambda: {} after {} iterations.".format(Ln_LOBPCG, iN_LOBPCG))

    print("Lanczos N Lambda: {} after {} iterations (i0={} + iN={}).".format(
        Ln_Lanczos, iN_Lanczos + i0_Lanczos, i0_Lanczos, iN_Lanczos))

    print("Power method N Lambda: {} after {} iterations (i0={} + iN={}).".format(Ln, iN+i0, i0, iN))



def evaluate_run_over_simple_nn_model(to_save, save_dir, iteration_number):

    model = Simple_NN()

    data_x = torch.normal(0.0, 1, size=(10000, 1))
    with torch.no_grad():
        data_y = model(data_x)+torch.normal(0.0, 10, size=(N_HIDDEN*100, 1))

    loss_function = nn.MSELoss(reduction='mean')
    # CustomLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    # func, func_params = functorch.make_functional(model)
    def model_loss_sum_params(params, X, data_y):
        # pred_y: torch.Tensor = _stateless.functional_call(model, {n: p for n, p in zip(names, params)}, data_x)
        pred_y = functional_call(model, params, X)
        loss = loss_function(pred_y, data_y)
        return loss.sum()

    model_parameters = tuple(model.parameters())
    is_comparable = True


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
            HM = (HM + HM.t())/2
        #print("L0 by finite differencing: ", L0)
        #HM_condition_number = torch.linalg.cond(HM)
        #print("condition number of hessian: ", HM_condition_number)
        V0_HM, L0_HM, Vn_HM, Ln_HM = get_V0L0_VnLn(HM)
        L, V = get_sorted_LV_by_abs_L(HM)

        Vn_LOBPCG, Ln_LOBPCG, iN_LOBPCG = get_LnVn_LOBPCG(
            model, loss_function, optimizer, data_x, data_y)

        V0, L0, dist_converged_0, dist_converged_stats0, Vn, Ln, dist_converged_n, dist_converged_statsn, i0, iN, second_iter_Ln = get_V0L0_VnLn_finite_differencing(
            model, loss_function, optimizer, data_x, data_y, MAX_ITER_NUM)

        V0_Lanczos, L0_Lanczos, dist_converged_0_Lanczos, converge_stats0_Lanczos, Vn_Lanczos, Ln_Lanczos, dist_converged_n_Lanczos, converge_statsn_Lanczos, i0_Lanczos, iN_Lanczos, second_iter_Ln_Lanczos = get_V0L0_VnLn_Lanczos(
            model, loss_function, optimizer, data_x, data_y, 0)

        #V0_Lanczos, L0_Lanczos, dist_converged_0_Lanczos, dist_converged_stats0_Lanczos, Vn_Lanczos, Ln_Lanczos, dist_converged_n_Lanczos, dist_converged_statsn_Lanczos, i0_Lanczos, iN_Lanczos, second_iter_Ln_Lanczos  = get_V0L0_VnLn_Lanczos(model, loss_function, optimizer, data_x, data_y, MAX_ITER_NUM)

        #err0_L = torch.matmul(HM, V0_Lanczos) - V0_Lanczos*L0_Lanczos
        #print("Lanczos 0 err: {} after {} iterations".format( torch.linalg.vector_norm(err0_L, ord =2), i0_Lanczos))
        err0_PM = torch.matmul(HM, V0) - V0*L0
        print("Power method 0 err: {} after {} iterations".format(
            torch.linalg.vector_norm(err0_PM, ord=2), i0))
        err0_L = torch.matmul(HM, V0_Lanczos) - V0_Lanczos*L0_Lanczos
        print("Lanczos 0 err: {} after {} iterations".format(
            torch.linalg.vector_norm(err0_L, ord=2), i0_Lanczos))

        print("Target Lambda: {}".format(L.min()))
        errN_LOB = torch.matmul(HM, Vn_LOBPCG) - Vn_LOBPCG*Ln_LOBPCG
        print("LOBPCG N err: {} after {} iterations. Lambda: {}".format(
            torch.linalg.vector_norm(errN_LOB, ord=2), iN_LOBPCG, Ln_LOBPCG))

        errN_L = torch.matmul(HM, Vn_Lanczos) - Vn_Lanczos*Ln_Lanczos
        print("Lanczos N err: {} after {} iterations (i0={} + iN={}). Lambda: {}".format(
            torch.linalg.vector_norm(errN_L, ord=2), iN_Lanczos + i0_Lanczos, i0_Lanczos, iN_Lanczos, Ln_Lanczos))

        # err0_L = torch.matmul(HM, V0_Lanczos) - V0_Lanczos*L0_Lanczos
        # print("Lanczos N err: {} after {} iterations. Lambda: {}".format(torch.linalg.vector_norm(err0_L, ord =2), i0_Lanczos*LANCZOS_DIM, L0_Lanczos))

        errN_PM = torch.matmul(HM, Vn) - Vn*Ln
        print("Power method N err: {} after {} iterations (i0={} + iN={}). Lambda: {}".format(
            torch.linalg.vector_norm(errN_PM, ord=2), iN+i0, i0, iN, Ln))

        HM2_ver_FD = abs(L0)*torch.eye(HM.shape[0], HM.shape[0]) - HM
        HM2_ver_H = abs(L0_HM)*torch.eye(HM.shape[0], HM.shape[0]) - HM

        LL2_ver_FD, VV2_ver_FD = get_sorted_LV(HM2_ver_FD)

        HM2_verFD_LL0 = L0.abs() - LL2_ver_FD[0]

        LL2_ver_H, VV2_ver_H = get_sorted_LV(HM2_ver_H)
        HM2_ver_H_LL0 = L0.abs() - LL2_ver_H[0]

        cosin_dist = get_dist(V0, V0_HM)

        if (abs(cosin_dist) < 0.9):
            correct_cosin = False
        if ((i0 >= MAX_ITER_NUM - 1) or (iN >= MAX_ITER_NUM - 1)):
            convereged = False

        L, V = get_sorted_LV_by_abs_L(HM)
        closest_l_to_l0 = get_closest_l_to_givenL(L, L0_HM)
        closest_l_to_ln = get_closest_l_to_givenL(L, Ln_HM)

        data = {"n hidden": [N_HIDDEN],
                "model size": [HM.shape[0]],
                "FD L0": [L0.item()],
                "True L[0]": [L0_HM.item()],
                "dist_converged_0": [dist_converged_0],
                "number of iterations 0": [i0],
                "Calculated Ln (twice FD)": [Ln.item()],
                "True L[n] of H ": [Ln_HM.item()],
                "Ln = (FD_L[0]*I - HM)": [LL2_ver_FD[0].item()],
                "Ln = (HM_L[0]*I - HM)": [LL2_ver_H[0].item()],
                "Ln = twice FD": [second_iter_Ln.item()],
                "number of iterations n": [iN],
                }
        overview_run_stats = pd.DataFrame(data)

        dist_converged_stats = pd.concat(
            [dist_converged_stats0, dist_converged_statsn])
        dist_converged_stats["n hidden"] = N_HIDDEN
        dist_converged_stats["H size"] = HM.shape[0]
        dist_converged_stats["L0"] = L0_HM.item()
        dist_converged_stats["L0-ref"] = closest_l_to_l0
        dist_converged_stats["Ln"] = Ln_HM.item()
        dist_converged_stats["Ln-ref"] = closest_l_to_ln

    except:
        print("An exception occurred in pytorch's H calculation")
        is_comparable = False

        data = {"n hidden": [N_HIDDEN],
                "model size": [get_model_size(model_parameters)],
                "Calculated L0": [L0.item()],
                "True L[0]": ["Err"],
                "True closest L to l[0]": ["Err"],
                "dist_converged_0": ["Err"],
                "number of iterations 0": [i0],
                "cosin 0": ["Err"],
                "Calculated Ln": [Ln.item()],
                "True L[n]": ["Err"],
                "True closest L to l[n-1]": ["Err"],
                "dist_converged_n": ["Err"],
                "number of iterations n": [iN],
                "cosin n": ["Err"],
                }
        overview_run_stats = pd.DataFrame(data)

        dist_converged_stats = pd.concat(
            [dist_converged_stats0, dist_converged_statsn])
        dist_converged_stats["n hidden"] = N_HIDDEN
        dist_converged_stats["H size"] = get_model_size(model_parameters)
        dist_converged_stats["L0"] = "Err in calculating explicit H"
        dist_converged_stats["L0-ref"] = "Err in calculating explicit H"
        dist_converged_stats["Ln"] = "Err in calculating explicit H"
        dist_converged_stats["Ln-ref"] = "Err in calculating explicit H"
    finally:

        if (to_save):
            if (is_comparable):
                if (correct_cosin):
                    save_dir = save_dir + "\\coverged_to_right_vec"
                else:
                    if (convereged):
                        save_dir = save_dir + "\\coverged_to_wrong_vec"
                    else:
                        save_dir = save_dir + "\\did_not_coverge"
            else:
                save_dir = save_dir + "\\incomperable"
        save_dir = save_dir + "\\" + str(iteration_number)
        if (not os.path.exists(save_dir)):
            os.makedirs(save_dir)
        save_dir = save_dir + "\\model.pth"
        torch.save(model, save_dir)

        return overview_run_stats, dist_converged_stats






def main():

    find_smallest_eig_mnist_model(MODEL_PATH, MODEL_FILENAME)

    
    # overview_run_stats = pd.DataFrame()
    # dist_converged_stats = pd.DataFrame()
    # out_dir = MODEL_PATH + "\\H_calc" + str(N_HIDDEN) + "\\epsilon(mp)" + str(EM)
    # out_stats_filename = out_dir + "\\" + "stats.xlsx"
    # for k in range(1):
        
    #     print("k is: ", k)
    #     overview_run_stats_k, dist_converged_stats_k = evaluate_run_over_simple_nn_model(
    #         True, out_dir, k)
    #     overview_run_stats_k["run num"] = k
    #     dist_converged_stats_k["run num"] = k
    #     overview_run_stats = pd.concat(
    #         [overview_run_stats, overview_run_stats_k])
    #     dist_converged_stats = pd.concat(
    #         [dist_converged_stats, dist_converged_stats_k])

    # with pd.ExcelWriter(out_stats_filename) as writer:
    #     overview_run_stats.to_excel(writer, sheet_name='data_summary')
    #     dist_converged_stats.to_excel(
    #         writer, sheet_name='convergence_iterations')


if __name__ == '__main__':
    main()

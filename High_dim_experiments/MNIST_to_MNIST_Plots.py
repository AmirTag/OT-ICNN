from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from VAE.train_vae_on_mnist import *
import pickle
import glob
import optimal_transport_modules
from optimal_transport_modules.icnn_modules import *
from optimal_transport_modules.all_losses import *
from mnist_data_loader import *
from utils import *
import mnist_utils
from mnist_utils import *

import time
import numpy as np
import pandas as pd
import os
import logging
import torch.utils.data
from utils import *
from datetime import datetime
from ast import literal_eval
from torchvision.utils import save_image
from torchvision.utils import make_grid
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from PIL import Image
from sklearn import decomposition
from scipy.stats import truncnorm
# from torchsummary import summary


# Training settings. Important ones first
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

parser.add_argument('--DATASET_X', type=str, default='MNIST PCA', help='which dataset to use for X')
parser.add_argument('--DATASET_Y', type=str, default='StandardGaussian', help='which dataset to use for Y')

parser.add_argument('--input_dim', type=int, default=784, help='dimensionality of the input x')

parser.add_argument('--latent_dim', type=int, default=16, help='dimensionality of the input x')

parser.add_argument('--batch_size', type=int, default=128, help='size of the batches')

parser.add_argument('--total_iters', type=int, default=100000, help='number of iterations of training')

parser.add_argument('--gen_iters', type=int, default=16, help='number of training steps for discriminator per iter')

parser.add_argument('--num_neurons', type=int, default=1024, help='number of neurons per layer')

parser.add_argument('--num_layers', type=int, default=3, help='number of hidden layers before output')

parser.add_argument('--lambda_cvx', type=float, default=1.0, help='Regularization constant for positive weight constraints')

parser.add_argument('--lambda_fenchel_eq', type=float, default=0.0, help='Regularization constant for making sure that fenchel equality holds for f,g')

parser.add_argument('--lambda_fenchel_ineq', type=float, default=0.0, help='Regularization constant for making sure that fenchel inequality holds')

parser.add_argument('--lambda_inverse_y_side', type=float, default=0.0, help='Regularization constant for making sure that grad g = (grad f)^{-1}')

parser.add_argument('--full_quadratic', type=bool, default=False, help='if the last layer is full quadratic or not')

parser.add_argument('--activation', type=str, default='celu', help='which activation to use for')

parser.add_argument('--initialization', type=str, default='trunc_inv_sqrt', help='which initialization to use for')

parser.add_argument('--trial', type=int, default=1, help='the trail no.')

parser.add_argument('--optimizer', type=str, default='Adam', help='which optimizer to use')

parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

parser.add_argument('--momentum', type=float, default=0.0, metavar='M',
                    help='SGD momentum (default: 0.5)')


parser.add_argument('--mnist_path', type=str, default='./data_mnist')

parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.99)


# Less frequently used training settings 



parser.add_argument('--lambda_mean', type=float, default=0.0, help='Regularization constant for  matching mean and covariance')

parser.add_argument('--have_skip', type=str, default=True, help='if you want skip connections or not')

parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')


parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--SHOW_THE_PLOT', type=bool, default=False, help='Boolean option to show the plots or not')
parser.add_argument('--DRAW_THE_ARROWS', type=bool, default=False, help='Whether to draw transport arrows or not')


parser.add_argument('--N_PLOT', type=int, default=64, help='number of samples for plotting')

parser.add_argument('--SCALE', type=float, default=10.0, help='scale for the gaussian_mixtures')
parser.add_argument('--VARIANCE', type=float, default=0.5, help='variance for each mixture')

parser.add_argument('--N_TEST', type=int, default=2048, help='number of test samples')

parser.add_argument('--N_CPU', type=int, default=8, help='number of cpu threads to use during batch generation')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--gpus', default=3,
                    help='gpus used for training - e.g 0,1,3')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

args.lr_schedule = 4000

# ### Seed stuff
# torch.manual_seed(args.seed)
#
# if args.cuda:
#     torch.cuda.manual_seed(args.seed)

####### Storing stuff

# if args.optimizer is 'SGD':
#     results_save_path = './My_Experimental_results/vaeMNIST->vaeMNIST/input_dim_{5}/activ_{13}/init_{6}/layers_{0}/neuron_{1}/lambda_cvx_{10}_FenchEqY_{11}_Ineq_{14}/optim_{8}lr_{2}momen_{7}/gen_{9}/batch_{3}/trial_{4}_last_{12}_qudr'.format(
#             args.num_layers, args.num_neurons, args.lr, args.batch_size, args.trial, args.latent_dim, args.initialization, args.momentum,
#             'SGD', args.gen_iters, args.lambda_cvx, args.lambda_fenchel_eq, 'full' if args.full_quadratic else 'inp', args.activation, args.lambda_fenchel_ineq)

# elif args.optimizer is 'Adam':
#     results_save_path = './My_Experimental_results/vaeMNIST->vaeMNIST/input_dim_{5}/activ_{14}/init_{6}/layers_{0}/neuron_{1}/lambda_cvx_{11}_FenchEqY_{12}_Ineq_{15}/optim_{9}lr_{2}betas_{7}_{8}/gen_{10}/batch_{3}/trial_{4}_last_{13}_qudr'.format(
#             args.num_layers, args.num_neurons, args.lr, args.batch_size, args.trial, args.latent_dim, args.initialization, args.beta1, args.beta2,
#             'Adam', args.gen_iters, args.lambda_cvx, args.lambda_fenchel_eq, 'full' if args.full_quadratic else 'inp', args.activation, args.lambda_fenchel_ineq)

results_save_path = './My_Experimental_results/vaeMNIST->vaeMNIST/input_dim_16/activ_celu/init_trunc_inv_sqrt/layers_3/neuron_1024/lambda_cvx_1.0_mean_0.0/optim_Adamlr_0.0001betas_0.9_0.99/gen_25/batch_64/trial_1_last_inp_qudr'    
    
    
model_save_path = results_save_path + '/storing_models'
sample_save_path = results_save_path +'/samples'
reconstruction_save_path = results_save_path +'/reconstruction'

# Loading the appropriate VAE
kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

# VAE model import path. For obtaining MNIST latent vectors

vae_model_path = './VAE/results_latent_{0}/vae_model.pt'.format(args.latent_dim)

pretrained_vae_model = VAE(args.latent_dim).cuda() if args.cuda else VAE(args.latent_dim)
pretrained_vae_model.load_state_dict(torch.load(vae_model_path))

################################################################
# Data stuff
################################################################

tf = transforms.Compose([#transforms.Resize(28),
                            transforms.ToTensor()])  # This is because VAE was trained on no transformations
                            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
mnist_dataset = datasets.MNIST('../data', train=True, download=True,
                   transform=tf)

mnist_full_loader = torch.utils.data.DataLoader(mnist_dataset, batch_size=len(mnist_dataset), shuffle=False, **kwargs)

all_digits, all_labels = next(iter(mnist_full_loader))

# This is for VAE projections
all_labels_np = all_labels.data.cpu().numpy()

all_projected_cordinates, _, _, _ = pretrained_vae_model(all_digits.cuda() if args.cuda else all_digits)

all_projected_cordinates_np = all_projected_cordinates.data.cpu().numpy()

print("Finished the VAE on MNIST \n")

# Functions
def compute_optimal_transport_map(y, convex_g):

    g_of_y = convex_g(y).sum()

    grad_g_of_y = torch.autograd.grad(g_of_y, y, create_graph=True)[0]

    return grad_g_of_y


# Defining the model
if args.full_quadratic:
    convex_f = Simple_Feedforward_3Layer_ICNN_LastFull_Quadratic(args.latent_dim, args.num_neurons, args.activation)
    convex_g = Simple_Feedforward_3Layer_ICNN_LastFull_Quadratic(args.latent_dim, args.num_neurons, args.activation)
else:
    convex_f = Simple_Feedforward_3Layer_ICNN_LastInp_Quadratic(args.latent_dim, args.num_neurons, args.activation)
    convex_g = Simple_Feedforward_3Layer_ICNN_LastInp_Quadratic(args.latent_dim, args.num_neurons, args.activation)

if args.cuda:
    convex_f.cuda()
    convex_g.cuda()       
    
## Loading stuff
convex_f.load_state_dict(torch.load(model_save_path+'/convex_f.pt'))
convex_g.load_state_dict(torch.load(model_save_path+'/convex_g.pt'))
    
## For reconstruction results
from numpy.random import seed
from numpy.random import randint

grid_size = 4

# Digit 0
integers = randint(0, 5000, grid_size)

one_indices = torch.ByteTensor(torch.from_numpy(1*(all_labels_np==0)).type(torch.ByteTensor))

one_cordinates = all_projected_cordinates[one_indices][integers, :]

reconstructed_zeros = pretrained_vae_model.decode(one_cordinates)

transported_zeros = pretrained_vae_model.decode(compute_optimal_transport_map(Variable(one_cordinates, requires_grad=True), convex_g))


# Digit 1
integers = randint(0, 5000, grid_size)

one_indices = torch.ByteTensor(torch.from_numpy(1*(all_labels_np==1)).type(torch.ByteTensor))

one_cordinates = all_projected_cordinates[one_indices][integers, :]

reconstructed_ones = pretrained_vae_model.decode(one_cordinates)

transported_ones = pretrained_vae_model.decode(compute_optimal_transport_map(Variable(one_cordinates, requires_grad=True), convex_g))


# Digit 2
integers = randint(0, 5000, grid_size)

one_indices = torch.ByteTensor(torch.from_numpy(1*(all_labels_np==2)).type(torch.ByteTensor))

one_cordinates = all_projected_cordinates[one_indices][integers, :]

reconstructed_twos = pretrained_vae_model.decode(one_cordinates)

transported_twos = pretrained_vae_model.decode(compute_optimal_transport_map(Variable(one_cordinates, requires_grad=True), convex_g))


# Digit 3
integers = randint(0, 5000, grid_size)

one_indices = torch.ByteTensor(torch.from_numpy(1*(all_labels_np==3)).type(torch.ByteTensor))

one_cordinates = all_projected_cordinates[one_indices][integers, :]

reconstructed_threes = pretrained_vae_model.decode(one_cordinates)

transported_threes = pretrained_vae_model.decode(compute_optimal_transport_map(Variable(one_cordinates, requires_grad=True), convex_g))


# Digit 4
integers = randint(0, 5000, grid_size)

one_indices = torch.ByteTensor(torch.from_numpy(1*(all_labels_np==4)).type(torch.ByteTensor))

one_cordinates = all_projected_cordinates[one_indices][integers, :]

reconstructed_fours = pretrained_vae_model.decode(one_cordinates)

transported_fours = pretrained_vae_model.decode(compute_optimal_transport_map(Variable(one_cordinates, requires_grad=True), convex_g))



recons = torch.cat([reconstructed_zeros.view(grid_size, 1, 28, 28), reconstructed_ones.view(grid_size, 1, 28, 28), reconstructed_twos.view(grid_size, 1, 28, 28),\
                   reconstructed_threes.view(grid_size, 1, 28, 28), reconstructed_fours.view(grid_size, 1, 28, 28)])


transp = torch.cat([transported_zeros.view(grid_size, 1, 28, 28), transported_ones.view(grid_size, 1, 28, 28), transported_twos.view(grid_size, 1, 28, 28),\
                   transported_threes.view(grid_size, 1, 28, 28),transported_fours.view(grid_size, 1, 28, 28)])



save_image(recons.cpu(), 'plots_MNIST->MNIST/latent_dim_{0}/all_initial.png'.format(args.latent_dim), nrow= grid_size)
    
save_image(transp.cpu(), 'plots_MNIST->MNIST/latent_dim_{0}/all_morphing.png'.format(args.latent_dim), nrow= grid_size)


save_image(recons.cpu(), 'plots_MNIST->MNIST/latent_dim_{0}/all_initial.pdf'.format(args.latent_dim), nrow= grid_size)
    
save_image(transp.cpu(), 'plots_MNIST->MNIST/latent_dim_{0}/all_morphing.pdf'.format(args.latent_dim), nrow= grid_size)

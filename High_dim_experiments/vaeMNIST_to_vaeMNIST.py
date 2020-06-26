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

if args.optimizer is 'SGD':
    results_save_path = './My_Experimental_results/vaeMNIST->vaeMNIST/input_dim_{5}/activ_{13}/init_{6}/layers_{0}/neuron_{1}/lambda_cvx_{10}_FenchEqY_{11}_Ineq_{14}/optim_{8}lr_{2}momen_{7}/gen_{9}/batch_{3}/trial_{4}_last_{12}_qudr'.format(
            args.num_layers, args.num_neurons, args.lr, args.batch_size, args.trial, args.latent_dim, args.initialization, args.momentum,
            'SGD', args.gen_iters, args.lambda_cvx, args.lambda_fenchel_eq, 'full' if args.full_quadratic else 'inp', args.activation, args.lambda_fenchel_ineq)

elif args.optimizer is 'Adam':
    results_save_path = './My_Experimental_results/vaeMNIST->vaeMNIST/input_dim_{5}/activ_{14}/init_{6}/layers_{0}/neuron_{1}/lambda_cvx_{11}_FenchEqY_{12}_Ineq_{15}/optim_{9}lr_{2}betas_{7}_{8}/gen_{10}/batch_{3}/trial_{4}_last_{13}_qudr'.format(
            args.num_layers, args.num_neurons, args.lr, args.batch_size, args.trial, args.latent_dim, args.initialization, args.beta1, args.beta2,
            'Adam', args.gen_iters, args.lambda_cvx, args.lambda_fenchel_eq, 'full' if args.full_quadratic else 'inp', args.activation, args.lambda_fenchel_ineq)

model_save_path = results_save_path + '/storing_models'
sample_save_path = results_save_path +'/samples'
reconstruction_save_path = results_save_path +'/reconstruction'

#os.makedirs(results_save_path, exist_ok = True)
os.makedirs(model_save_path, exist_ok = True)
os.makedirs(sample_save_path, exist_ok = True)
os.makedirs(reconstruction_save_path, exist_ok= True)

setup_logging(os.path.join(results_save_path , 'log.txt'))
results_file = os.path.join(results_save_path , 'results.%s')
results = ResultsLog(results_file % 'csv', results_file % 'html')

logging.info("saving to %s \n", results_save_path )
logging.debug("run arguments: %s", args)

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
mnist_dataset = datasets.MNIST('./data', train=True, download=True,
                   transform=tf)

mnist_full_loader = torch.utils.data.DataLoader(mnist_dataset, batch_size=len(mnist_dataset), shuffle=False, **kwargs)

all_digits, all_labels = next(iter(mnist_full_loader))

# This is for VAE projections
all_labels_np = all_labels.data.cpu().numpy()

all_projected_cordinates, _, _, _ = pretrained_vae_model(all_digits.cuda() if args.cuda else all_digits)

all_projected_cordinates_np = all_projected_cordinates.data.cpu().numpy()

logging.info("Finished the VAE on MNIST \n")



# # This is for PCA projections
# all_digits_np, all_labels_np = all_digits.reshape(-1, 784).data.cpu().numpy(), all_labels.data.cpu().numpy()

# estimator = decomposition.PCA(n_components=args.latent_dim, svd_solver='randomized', whiten=True)

# all_projected_cordinates = estimator.fit_transform(all_digits_np)

# print(all_projected_cordinates.shape)

# reconstructed_digits = estimator.inverse_transform(all_projected_cordinates)

# logging.info("Finished the PCA on MNIST \n")


# # This is for Gaussian to all MNIST images
# all_projected_cordinates_torch = torch.from_numpy(all_projected_cordinates).float()
# proj_cordinates_generator = RealDataGeneratorDummy(torch.utils.data.DataLoader(all_projected_cordinates_torch, batch_size=args.batch_size, shuffle=True, **kwargs))


# all_projected_cordinates_transp = all_projected_cordinates.transpose()
# proj_mean = torch.from_numpy(np.mean(all_projected_cordinates_transp, axis=1)).float()
# proj_cov = torch.from_numpy(np.cov(all_projected_cordinates_transp)).float()
# gaussian_generator = StandardGaussianGenerator(args.batch_size, proj_mean, proj_cov, lambda_identity=0.)


first_five_digit_cordinates_torch = torch.from_numpy(all_projected_cordinates_np[all_labels_np < 5]).float()
first_five_digit_cordinates_generator = RealDataGeneratorDummy(torch.utils.data.DataLoader(first_five_digit_cordinates_torch, batch_size=args.batch_size, shuffle=True, **kwargs))

last_five_digit_cordinates_torch = torch.from_numpy(all_projected_cordinates_np[all_labels_np > 4]).float()
last_five_digit_cordinates_generator = RealDataGeneratorDummy(torch.utils.data.DataLoader(last_five_digit_cordinates_torch, batch_size=args.batch_size, shuffle=True, **kwargs))

gaussian_generator = first_five_digit_cordinates_generator

logging.info("Created the data loader for both PCA and Gaussian data \n")

# Plotting stuff
fixed_gaussian_plot_data =  mnist_utils.to_var(next(gaussian_generator), requires_grad=True)

print(fixed_gaussian_plot_data.shape)


############################################################
## Model stuff

def get_data(real_data_generator=last_five_digit_cordinates_generator, gaussian_generator=first_five_digit_cordinates_generator):
    real_data = mnist_utils.to_var(next(real_data_generator))
    gaussian_data = mnist_utils.to_var(next(gaussian_generator), requires_grad=True)
    if real_data.size() != gaussian_data.size():
        real_data = mnist_utils.to_var(next(real_data_generator))
        gaussian_data = mnist_utils.to_var(next(gaussian_generator), requires_grad=True)
    return real_data, gaussian_data




## This function computes the optimal transport map given by \nabla convex_g(y)
## Note that 'y' is of size (batch_size, y_dim). Hence the output is also of the same dimension


def compute_optimal_transport_map(y, convex_g):

    g_of_y = convex_g(y).sum()

    grad_g_of_y = torch.autograd.grad(g_of_y, y, create_graph=True)[0]

    return grad_g_of_y


def truncated_normal(size, threshold=1):
    values = truncnorm.rvs(-threshold, threshold, size=size)
    return values



# Everything about model ends here
##############################################################



################################################################
## Everything related to both the convex functions

if args.num_layers == 2:

    if args.full_quadratic:
        convex_f = Simple_Feedforward_2Layer_ICNN_LastFull_Quadratic(args.latent_dim, args.num_neurons, args.activation)
        convex_g = Simple_Feedforward_2Layer_ICNN_LastFull_Quadratic(args.latent_dim, args.num_neurons, args.activation)
    else:
        convex_f = Simple_Feedforward_2Layer_ICNN_LastInp_Quadratic(args.latent_dim, args.num_neurons, args.activation)
        convex_g = Simple_Feedforward_2Layer_ICNN_LastInp_Quadratic(args.latent_dim, args.num_neurons, args.activation)

elif args.num_layers == 3:

    if args.full_quadratic:
        convex_f = Simple_Feedforward_3Layer_ICNN_LastFull_Quadratic(args.latent_dim, args.num_neurons, args.activation)
        convex_g = Simple_Feedforward_3Layer_ICNN_LastFull_Quadratic(args.latent_dim, args.num_neurons, args.activation)
    else:
        convex_f = Simple_Feedforward_3Layer_ICNN_LastInp_Quadratic(args.latent_dim, args.num_neurons, args.activation)
        convex_g = Simple_Feedforward_3Layer_ICNN_LastInp_Quadratic(args.latent_dim, args.num_neurons, args.activation)

elif args.num_layers == 4:
    
    if args.full_quadratic:
        convex_f = Simple_Feedforward_4Layer_ICNN_LastFull_Quadratic(args.latent_dim, args.num_neurons, args.activation)
        convex_g = Simple_Feedforward_4Layer_ICNN_LastFull_Quadratic(args.latent_dim, args.num_neurons, args.activation)
    else:
        convex_f = Simple_Feedforward_4Layer_ICNN_LastInp_Quadratic(args.latent_dim, args.num_neurons, args.activation)
        convex_g = Simple_Feedforward_4Layer_ICNN_LastInp_Quadratic(args.latent_dim, args.num_neurons, args.activation)

elif args.num_layers == 5:

    if args.full_quadratic:
        convex_f = Simple_Feedforward_5Layer_ICNN_LastFull_Quadratic(args.latent_dim, args.num_neurons, args.activation)
        convex_g = Simple_Feedforward_5Layer_ICNN_LastFull_Quadratic(args.latent_dim, args.num_neurons, args.activation)
    else:
        convex_f = Simple_Feedforward_5Layer_ICNN_LastInp_Quadratic(args.latent_dim, args.num_neurons, args.activation)
        convex_g = Simple_Feedforward_5Layer_ICNN_LastInp_Quadratic(args.latent_dim, args.num_neurons, args.activation)

# ### First initialize all weight variables as N(0, 0.01). this initialization seems to be
# bad
# convex_f.apply(weights_init)
# convex_g.apply(weights_init)

# Print model's state_dict
print("Convex_f state_dict:")
# print(convex_f)

for param_tensor in convex_f.state_dict():
    print(param_tensor, "\t", convex_f.state_dict()[param_tensor].size())


f_positive_params = []

### Form a list of positive weight parameters
# and also initialize them with positive values

for p in list(convex_f.parameters()):
    if hasattr(p, 'be_positive'):
        f_positive_params.append(p)
    
    p.data = torch.from_numpy(truncated_normal(p.shape, threshold=1./np.sqrt(p.shape[1] if len(p.shape)>1 else p.shape[0]))).float()

g_positive_params = []

for p in list(convex_g.parameters()):
    if hasattr(p, 'be_positive'):
        g_positive_params.append(p)

    p.data = torch.from_numpy(truncated_normal(p.shape, threshold=1./np.sqrt(p.shape[1] if len(p.shape)>1 else p.shape[0]))).float()

if args.cuda:
    convex_f.cuda()
    convex_g.cuda()    

logging.info("Created and initialized the convex neural networks 'f' and 'g'")
num_parameters = sum([l.nelement() for l in convex_f.parameters()])
logging.info("number of parameters: %d", num_parameters)

# print("before the constraint loss")

if args.optimizer is 'SGD':

    optimizer_f = optim.SGD(convex_f.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer_g = optim.SGD(convex_g.parameters(), lr=args.lr, momentum=args.momentum)

if args.optimizer is 'Adam':

    optimizer_f = optim.Adam(convex_f.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=1e-5)
    optimizer_g = optim.Adam(convex_g.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=1e-5)



###########################################################

#################
## Saving and loading model stuff
#################

# ## Saving stuff
# torch.save(convex_f.state_dict(), model_save_path+'/convex_f.pt')
# torch.save(convex_g.state_dict(), model_save_path+'/convex_g.pt')

# ## Loading stuff
# convex_f.load_state_dict(torch.load(model_save_path+'/convex_f.pt'))
# convex_g.load_state_dict(torch.load(model_save_path+'/convex_g.pt'))
#
# convex_f.eval()
# convex_g.eval()

# print(convex_f.fc1_normal.weight)


###############################################################################

### Training stuff




def train(iteration):

    convex_f.train()
    convex_g.train()

    w_2_loss_value_iter = 0
    g_OT_loss_value_iter = 0
    g_cvx_Constraint_loss_value_iter = 0
    g_FenchEq_Constraint_loss_value_iter = 0
    g_FenchIneq_Constraint_loss_value_iter = 0

    real_data, y = get_data()

    # real_data, y = real_data.view(-1, 28 * 28), y.view(-1, 28 * 28)

    optimizer_f.zero_grad()
    optimizer_g.zero_grad()

    # Train the parameters of 'g'
    for _ in range(1, args.gen_iters+1):

        # First do a forward pass on y and compute grad_g_y
        # Then do a backward pass update on parameters of g
        g_of_y = convex_g(y).sum()

        grad_g_of_y = torch.autograd.grad(g_of_y, y, create_graph=True)[0]

        f_grad_g_y = convex_f(grad_g_of_y)

        loss_g = f_grad_g_y.mean() - torch.dot(grad_g_of_y.reshape(-1), y.reshape(-1)) / y.size(0)
        g_OT_loss_value_iter += loss_g.item()


        if args.lambda_fenchel_eq > 0:

            fenchel_eq_loss, _ =  equality_young_fenchel_loss(grad_g_of_y, f_grad_g_y, real_data, y, convex_g)
            
            g_FenchEq_Constraint_loss_value_iter += fenchel_eq_loss.item()

            loss_g += args.lambda_fenchel_eq * fenchel_eq_loss
        
        if args.lambda_fenchel_ineq > 0:

            fenchel_ineq_loss =  inequality_young_fenchel_loss(convex_f, convex_g, real_data, y)
            
            g_FenchIneq_Constraint_loss_value_iter += fenchel_ineq_loss.item()

            loss_g += args.lambda_fenchel_ineq * fenchel_ineq_loss
        
        if args.lambda_inverse_y_side > 0:

            inverse_constraint_loss_y_side = convex_fn_inverse_constraint_loss_y_side(convex_f, convex_g, y)
            
            g_inv_Constraint_loss_value_iter += inverse_constraint_loss_y_side.item()

            loss_g += args.lambda_inverse_y_side * inverse_constraint_loss_y_side

        # So for the last iteration, gradients of 'f' parameters are also updated
        loss_g.backward()

        # g_params_grad_full = torch.cat([p.grad.reshape(-1).data.cpu() for p in list(convex_g.parameters())])
        # norm_g_parms_grad_full += torch.norm(g_params_grad_full).item()


        ### Constraint loss for g parameters
        if args.lambda_cvx > 0:

            g_positive_constraint_loss = args.lambda_cvx*compute_constraint_loss(g_positive_params)
            g_cvx_Constraint_loss_value_iter += g_positive_constraint_loss.item()/args.lambda_cvx
            g_positive_constraint_loss.backward()        

        optimizer_g.step()

        ## Maintaining the positive constraints on the convex_g_params
        if args.lambda_cvx == 0:
            for p in g_positive_params:
                p.data.copy_(torch.relu(p.data))
        
        optimizer_f.zero_grad()
        optimizer_g.zero_grad()


    g_OT_loss_value_iter /= args.gen_iters
    g_cvx_Constraint_loss_value_iter /= args.gen_iters
    g_FenchEq_Constraint_loss_value_iter /= args.gen_iters
    g_FenchIneq_Constraint_loss_value_iter /= args.gen_iters



    ## Train the parameters of 'f'
    # real_data, y = get_data()
    y.grad.data.zero_()
    g_of_y = convex_g(y).sum()
    grad_g_of_y = torch.autograd.grad(g_of_y, y, create_graph=True)[0]
    f_grad_g_y = convex_f(grad_g_of_y)

    f_real_loss = convex_f(real_data).mean()

    f_loss = f_real_loss - f_grad_g_y.mean()

    if args.lambda_fenchel_eq > 0:

        fenchel_eq_loss, _ =  equality_young_fenchel_loss(grad_g_of_y, f_grad_g_y, real_data, y, convex_g)
    
        f_loss += args.lambda_fenchel_eq * fenchel_eq_loss
        
    if args.lambda_fenchel_ineq > 0:

        fenchel_ineq_loss =  inequality_young_fenchel_loss(convex_f, convex_g, real_data, y)
            
        f_loss += args.lambda_fenchel_ineq * fenchel_ineq_loss

    f_loss.backward()

    optimizer_f.step()

    # Maintain the "f" parameters positive
    for p in f_positive_params:
        p.data.copy_(torch.relu(p.data))

    # print(real_data.shape)
    w_2_loss_value_iter = g_OT_loss_value_iter-f_real_loss.item()+0.5*real_data.pow(2).sum(dim=1).mean().item()+0.5*y.pow(2).sum(dim=1).mean().item()

    # results.add(Iteration=iteration, w2_loss_train_samples=w_2_loss_value_iter, g_OT_train_loss=g_OT_loss_value_iter,\
    #                                  g_cvx_Constraint_loss=g_cvx_Constraint_loss_value_iter, g_FenchEq_Constraint_loss = g_FenchEq_Constraint_loss_value_iter, \
    #                                      g_FenchIneq_Constraint_loss = g_FenchIneq_Constraint_loss_value_iter)
                    
    # results.save()

    return w_2_loss_value_iter, g_OT_loss_value_iter, g_cvx_Constraint_loss_value_iter, g_FenchEq_Constraint_loss_value_iter, g_FenchIneq_Constraint_loss_value_iter


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += criterion(output, target).item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def save_images_as_grid(array_img_vectors, epoch):

    #array_img_vectors is of size (N, PCA_components). So obtain the images first using inverse PCA transform

    array_img_vectors = torch.from_numpy(estimator.inverse_transform(array_img_vectors.data.cpu().numpy())).float()

    array_img_vectors = array_img_vectors.reshape(-1, 1, 28, 28)
    grid = make_grid(array_img_vectors, nrow=8)
    ndarr = grid.mul_(255).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(results_save_path+'/epoch_{0}.png'.format(epoch))


###################################################
## Training stuff

total_w_2_loss_list = []

# total_g_OT_loss_list = []
# total_g_cvx_Constraint_loss_list = []
# total_g_inv_Constraint_loss_list = []


##########
## Plotting the starting set of digits
##########

array_img_vectors = pretrained_vae_model.decode(fixed_gaussian_plot_data).cpu()

array_img_vectors = array_img_vectors.reshape(-1, 1, 28, 28)

save_image(array_img_vectors,
                sample_save_path+ '/starting_0.png')
        



for iteration in range(1, args.total_iters + 1):

    w_2_loss_value_iter, g_OT_loss_value_iter, g_cvx_Constraint_loss_value_iter, g_FenchEq_Constraint_loss_value_iter, g_FenchIneq_Constraint_loss_value_iter = train(iteration)

    total_w_2_loss_list.append(w_2_loss_value_iter)
    # total_g_OT_loss_list.append(g_OT_loss_value_iter)
    # total_g_cvx_Constraint_loss_list.append(g_cvx_Constraint_loss_value_iter)

    if iteration % args.log_interval == 0:
        logging.info('Iteration: {} [{}/{} ({:.0f}%)] g_OT_loss: {:.4f} g_cvx_Loss: {:.4f} g_Eq_loss: {:.4f} g_Ineq_loss: {:.4f} W_2_loss: {:.4f} '.format(
            iteration, iteration, args.total_iters,
            100. * iteration / args.total_iters, g_OT_loss_value_iter, g_cvx_Constraint_loss_value_iter, g_FenchEq_Constraint_loss_value_iter,\
                g_FenchIneq_Constraint_loss_value_iter, w_2_loss_value_iter))

    if iteration % 500 == 0:

        transported_y = compute_optimal_transport_map(fixed_gaussian_plot_data, convex_g)

        # # This line is for PCA
        # array_img_vectors = torch.from_numpy(estimator.inverse_transform(transported_y.data.cpu().numpy())).float()

        array_img_vectors = pretrained_vae_model.decode(transported_y).cpu()

        array_img_vectors = array_img_vectors.reshape(-1, 1, 28, 28)

        #fixed_gz = model.g(fixed_z).view(*fixed_z.size())
        
        save_image(array_img_vectors,
                       sample_save_path+ '/sample_' + str(iteration) + '.png')
        
        # mnist_utils.visualize_single(array_img_vectors, results_save_path+'/iter_{0}.png'.format(iteration), args)


        # save_images_as_grid(transported_y, iteration)
    
    if iteration % args.lr_schedule == 0:
        
        optimizer_g.param_groups[0]['lr'] = optimizer_g.param_groups[0]['lr'] * 0.5

        optimizer_f.param_groups[0]['lr'] = optimizer_f.param_groups[0]['lr'] * 0.5


    # if epoch % 10 == 0:
    # if iteration == 1 or iteration == args.total_iters:
    if iteration % 1000 == 0:
        torch.save(convex_f.state_dict(), model_save_path + '/convex_f.pt')
        torch.save(convex_g.state_dict(), model_save_path + '/convex_g.pt')


logging.info("Training is finished and the models are saved. Good job :)")

# plt.plot(range(1, len(total_w_2_loss_list) + 1), total_w_2_loss_list, label='Training loss')
# plt.xlabel('iterations')
# plt.ylabel(r'$W_2$-loss value')
# plt.savefig(results_save_path+'/training_loss.png')
# plt.show()



###############################################################################################

## For reconstruction results

grid_size = 4

for i in range(5):

    one_indices = torch.ByteTensor(torch.from_numpy(1*(all_labels_np==i)).type(torch.ByteTensor))

    one_cordinates = all_projected_cordinates[one_indices][:grid_size, :]

    reconstructed_ones = pretrained_vae_model.decode(one_cordinates)

    transported_ones = pretrained_vae_model.decode(compute_optimal_transport_map(Variable(one_cordinates, requires_grad=True), convex_g))

    comparison =  torch.cat([reconstructed_ones.view(grid_size, 1, 28, 28), transported_ones.view(grid_size, 1, 28, 28)])

    save_image(comparison.cpu(), reconstruction_save_path+'/{0}_morphing.pdf'.format(i), nrow= grid_size)


#################################################################################################




# ###################################################################################
# ## Checking the distribution stuff

# convex_f.load_state_dict(torch.load(model_save_path+'/convex_f.pt'))
# convex_g.load_state_dict(torch.load(model_save_path+'/convex_g.pt'))

# all_projected_cordinates_torch = torch.from_numpy(all_projected_cordinates).float()

# y = torch.randn(60000, args.latent_dim, requires_grad=True)

# if args.cuda:

#     all_projected_cordinates_torch, y = all_projected_cordinates_torch.cuda(), y.cuda()


# g_of_y = convex_g(y).sum()

# transport_y = torch.autograd.grad(g_of_y, y, create_graph=True)[0]

# mean_transport_y = transport_y.mean(0)
# mean_x = all_projected_cordinates_torch.mean(0)

# mean_loss = (mean_transport_y - mean_x).pow(2).sum().item()

# print("Mean loss is:", mean_loss)

# std_transport_y = transport_y.std(0)
# std_x = all_projected_cordinates_torch.std(0)

# std_loss = (std_transport_y - std_x).pow(2).sum().item()

# print("Standard deviation loss is:", std_loss)

######################################################################
# ### Ploting stuff

# num = 16

# for n in range(num):
#     plt.subplot(4,4,n+1), plt.imshow(np.reshape(255*images_recons[n,:], (28,28)), cmap='gray',
#      interpolation='bicubic',clim=(0,255)), plt.axis('off')

# plt.show()

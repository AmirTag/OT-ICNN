
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import optimal_transport_modules
from optimal_transport_modules.icnn_modules import *
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

parser.add_argument('--DATASET_X', type=str, default='mixtureGaussian', help='which dataset to use for X')
parser.add_argument('--DATASET_Y', type=str, default='StandardGaussian', help='which dataset to use for Y')

parser.add_argument('--INPUT_DIM', type=int, default=784, help='dimensionality of the input x')

parser.add_argument('--BATCH_SIZE', type=int, default=60, help='size of the batches')

parser.add_argument('--epochs', type=int, default=40, metavar='S',
                    help='number_of_epochs')

parser.add_argument('--N_GENERATOR_ITERS', type=int, default=16, help='number of training steps for discriminator per iter')

parser.add_argument('--NUM_NEURON', type=int, default=1024, help='number of neurons per layer')

parser.add_argument('--NUM_LAYERS', type=int, default=3, help='number of hidden layers before output')

parser.add_argument('--full_quadratic', type=bool, default=False, help='if the last layer is full quadratic or not')

parser.add_argument('--activation', type=str, default='leaky_relu', help='which activation to use for')

parser.add_argument('--initialization', type=str, default='trunc_inv_sqrt', help='which initialization to use for')

parser.add_argument('--TRIAL', type=int, default=1, help='the trail no.')

parser.add_argument('--optimizer', type=str, default='Adam', help='which optimizer to use')

parser.add_argument('--LR', type=float, default=1e-4, help='learning rate')

parser.add_argument('--momentum', type=float, default=0.0, metavar='M',
                    help='SGD momentum (default: 0.5)')

parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--beta2', type=float, default=0.99)


# Less frequently used training settings 

parser.add_argument('--LAMBDA_CVX', type=float, default=0.1, help='Regularization constant for positive weight constraints')

parser.add_argument('--LAMBDA_MEAN', type=float, default=0.0, help='Regularization constant for  matching mean and covariance')

parser.add_argument('--have_skip', type=str, default=True, help='if you want skip connections or not')

parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')


parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--SHOW_THE_PLOT', type=bool, default=False, help='Boolean option to show the plots or not')
parser.add_argument('--DRAW_THE_ARROWS', type=bool, default=False, help='Whether to draw transport arrows or not')


parser.add_argument('--N_PLOT', type=int, default=16, help='number of samples for plotting')

parser.add_argument('--SCALE', type=float, default=10.0, help='scale for the gaussian_mixtures')
parser.add_argument('--VARIANCE', type=float, default=0.5, help='variance for each mixture')

parser.add_argument('--N_TEST', type=int, default=2048, help='number of test samples')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--gpus', default=3,
                    help='gpus used for training - e.g 0,1,3')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

args.lr_schedule = 2 if args.BATCH_SIZE == 60 else 4

# ### Seed stuff
# torch.manual_seed(args.seed)
#
# if args.cuda:
#     torch.cuda.manual_seed(args.seed)

####### Storing stuff

if args.optimizer is 'SGD':
    results_save_path = './Results_Stand_Gauss_Gauss/input_dim_{5}/init_{6}/layers_{0}/neuron_{1}/lambda_cvx_{10}_mean_{11}/optim_{8}lr_{2}momen_{7}/gen_{9}/batch_{3}/trial_{4}_last_{12}_qudr'.format(
            args.NUM_LAYERS, args.NUM_NEURON, args.LR, args.BATCH_SIZE, args.TRIAL, args.INPUT_DIM, args.initialization, args.momentum,
            'SGD', args.N_GENERATOR_ITERS, args.LAMBDA_CVX, args.LAMBDA_MEAN, 'full' if args.full_quadratic else 'inp')

elif args.optimizer is 'Adam':
    results_save_path = './Results_Stand_Gauss_Gauss/input_dim_{5}/init_{6}/layers_{0}/neuron_{1}/lambda_cvx_{11}_mean_{12}/optim_{9}lr_{2}betas_{7}_{8}/gen_{10}/batch_{3}/trial_{4}_last_{13}_qudr'.format(
            args.NUM_LAYERS, args.NUM_NEURON, args.LR, args.BATCH_SIZE, args.TRIAL, args.INPUT_DIM, args.initialization, args.beta1, args.beta2,
            'Adam', args.N_GENERATOR_ITERS, args.LAMBDA_CVX, args.LAMBDA_MEAN, 'full' if args.full_quadratic else 'inp')

model_save_path = results_save_path + '/storing_models'


#os.makedirs(results_save_path, exist_ok = True)
os.makedirs(model_save_path, exist_ok = True)

setup_logging(os.path.join(results_save_path , 'log.txt'))
results_file = os.path.join(results_save_path , 'results.%s')
results = ResultsLog(results_file % 'csv', results_file % 'html')

logging.info("saving to %s \n", results_save_path )
logging.debug("run arguments: %s", args)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

################################################################
# Data stuff

mu = args.SCALE * torch.ones(1, args.INPUT_DIM)

X_data = mu + torch.randn(60000, args.INPUT_DIM)

train_loader = torch.utils.data.DataLoader(X_data, batch_size=args.BATCH_SIZE, shuffle=True, **kwargs)


logging.info("Created the data loader for X samples \n")

# Plotting stuff

y_plot = Variable(torch.randn(args.N_PLOT, args.INPUT_DIM), requires_grad=True)

initial_y = y_plot.data.cpu().numpy()


x_plot = X_data[:args.N_PLOT, :]

initial_x = x_plot.data.cpu().numpy()

if args.cuda:
    y_plot = y_plot.cuda()

############################################################
## Model stuff

### This loss is a relaxation of positive constraints on the weights
### Hence we penalize the negative ReLU

def compute_constraint_loss(list_of_params):
    
    loss_val = 0

    for p in list_of_params:
        loss_val += torch.relu(-p).pow(2).sum()
    return loss_val

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

if args.NUM_LAYERS == 2:

    if args.full_quadratic:
        convex_f = Simple_Feedforward_2Layer_ICNN_LastFull_Quadratic(args.INPUT_DIM, args.NUM_NEURON, args.activation)
        convex_g = Simple_Feedforward_2Layer_ICNN_LastFull_Quadratic(args.INPUT_DIM, args.NUM_NEURON, args.activation)
    else:
        convex_f = Simple_Feedforward_2Layer_ICNN_LastInp_Quadratic(args.INPUT_DIM, args.NUM_NEURON, args.activation)
        convex_g = Simple_Feedforward_2Layer_ICNN_LastInp_Quadratic(args.INPUT_DIM, args.NUM_NEURON, args.activation)

elif args.NUM_LAYERS == 3:

    if args.full_quadratic:
        convex_f = Simple_Feedforward_3Layer_ICNN_LastFull_Quadratic(args.INPUT_DIM, args.NUM_NEURON, args.activation)
        convex_g = Simple_Feedforward_3Layer_ICNN_LastFull_Quadratic(args.INPUT_DIM, args.NUM_NEURON, args.activation)
    else:
        convex_f = Simple_Feedforward_3Layer_ICNN_LastInp_Quadratic(args.INPUT_DIM, args.NUM_NEURON, args.activation)
        convex_g = Simple_Feedforward_3Layer_ICNN_LastInp_Quadratic(args.INPUT_DIM, args.NUM_NEURON, args.activation)

elif args.NUM_LAYERS == 4:
    
    if args.full_quadratic:
        convex_f = Simple_Feedforward_4Layer_ICNN_LastFull_Quadratic(args.INPUT_DIM, args.NUM_NEURON, args.activation)
        convex_g = Simple_Feedforward_4Layer_ICNN_LastFull_Quadratic(args.INPUT_DIM, args.NUM_NEURON, args.activation)
    else:
        convex_f = Simple_Feedforward_4Layer_ICNN_LastInp_Quadratic(args.INPUT_DIM, args.NUM_NEURON, args.activation)
        convex_g = Simple_Feedforward_4Layer_ICNN_LastInp_Quadratic(args.INPUT_DIM, args.NUM_NEURON, args.activation)

elif args.NUM_LAYERS == 5:

    if args.full_quadratic:
        convex_f = Simple_Feedforward_5Layer_ICNN_LastFull_Quadratic(args.INPUT_DIM, args.NUM_NEURON, args.activation)
        convex_g = Simple_Feedforward_5Layer_ICNN_LastFull_Quadratic(args.INPUT_DIM, args.NUM_NEURON, args.activation)
    else:
        convex_f = Simple_Feedforward_5Layer_ICNN_LastInp_Quadratic(args.INPUT_DIM, args.NUM_NEURON, args.activation)
        convex_g = Simple_Feedforward_5Layer_ICNN_LastInp_Quadratic(args.INPUT_DIM, args.NUM_NEURON, args.activation)


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

f_positive_constraint_loss = compute_constraint_loss(f_positive_params)

g_positive_constraint_loss = compute_constraint_loss(g_positive_params)

if args.optimizer is 'SGD':

    optimizer_f = optim.SGD(convex_f.parameters(), lr=args.LR, momentum=args.momentum)
    optimizer_g = optim.SGD(convex_g.parameters(), lr=args.LR, momentum=args.momentum)

if args.optimizer is 'Adam':

    optimizer_f = optim.Adam(convex_f.parameters(), lr=args.LR, betas=(args.beta1, args.beta2), weight_decay=1e-5)
    optimizer_g = optim.Adam(convex_g.parameters(), lr=args.LR, betas=(args.beta1, args.beta2), weight_decay=1e-5)



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

def train(epoch):

    convex_f.train()
    convex_g.train()

    # count = 0

    w_2_loss_value_epoch = 0

    g_OT_loss_value_epoch = 0

    g_Constraint_loss_value_epoch = 0

    for batch_idx, real_data in enumerate(train_loader):

        # count += 1

        if args.cuda:

            real_data = real_data.cuda()

        real_data = Variable(real_data)

        y = Variable(torch.randn(args.BATCH_SIZE, args.INPUT_DIM), requires_grad= True)

        if args.cuda:
            y = y.cuda()

        optimizer_f.zero_grad()
        optimizer_g.zero_grad()

        g_OT_loss_val_batch = 0
        g_Constraint_loss_val_batch = 0

        # norm_g_parms_grad_full = 0

        for inner_iter in range(1, args.N_GENERATOR_ITERS+1):
            # First do a forward pass on y and compute grad_g_y
            # Then do a backward pass update on parameters of g

            optimizer_g.zero_grad()

            g_of_y = convex_g(y).sum()

            grad_g_of_y = torch.autograd.grad(g_of_y, y, create_graph=True)[0]

            f_grad_g_y = convex_f(grad_g_of_y).mean()

            loss_g = f_grad_g_y - torch.dot(grad_g_of_y.reshape(-1), y.reshape(-1)) / y.size(0)
            g_OT_loss_val_batch += loss_g.item()

            if args.LAMBDA_MEAN > 0:

                mean_difference_loss = args.LAMBDA_MEAN * (real_data.mean(0) - grad_g_of_y.mean(0)).pow(2).sum()
                variance_difference_loss = args.LAMBDA_MEAN * (real_data.std(0) - grad_g_of_y.std(0)).pow(2).sum()

                loss_g += mean_difference_loss + variance_difference_loss


            loss_g.backward()

            # g_params_grad_full = torch.cat([p.grad.reshape(-1).data.cpu() for p in list(convex_g.parameters())])
            # norm_g_parms_grad_full += torch.norm(g_params_grad_full).item()


            ### Constraint loss for g parameters
            if args.LAMBDA_CVX > 0:
                # print("Came here!")
                g_positive_constraint_loss = args.LAMBDA_CVX*compute_constraint_loss(g_positive_params)
                g_Constraint_loss_val_batch += g_positive_constraint_loss
                g_positive_constraint_loss.backward()            

            optimizer_g.step()


            ## Maintaining the positive constraints on the convex_g_params
            if args.LAMBDA_CVX == 0:
                for p in g_positive_params:
                    p.data.copy_(torch.relu(p.data))

            


            ### Just for the last iteration keep the gradient on f intact
            ### otherwise need to do from scratch
            if inner_iter != args.N_GENERATOR_ITERS:
                optimizer_f.zero_grad()

        g_OT_loss_val_batch /= args.N_GENERATOR_ITERS
        g_Constraint_loss_val_batch /= args.N_GENERATOR_ITERS

        # norm_g_parms_grad_full /= args.N_GENERATOR_ITERS

        ## Flip the gradient sign for parameters in convex_f
        # because we need to solve "sup" maximization for f
        for p in list(convex_f.parameters()):
            p.grad.copy_(-p.grad)

        #print(real_data.size())
        remaining_f_loss = convex_f(real_data).mean()
        remaining_f_loss.backward()

        optimizer_f.step()

        # Maintain the "f" parameters positive
        for p in f_positive_params:
            p.data.copy_(torch.relu(p.data))

        w_2_loss_value_batch = g_OT_loss_val_batch - remaining_f_loss.item() + 0.5*real_data.pow(2).sum(dim=1).mean().item() + 0.5*y.pow(2).sum(dim=1).mean().item()
        w_2_loss_value_epoch += w_2_loss_value_batch

        g_OT_loss_value_epoch += g_OT_loss_val_batch
        g_Constraint_loss_value_epoch += g_Constraint_loss_val_batch

        # w_2_loss_list.append(w_2_loss_value)

        # mean_difference_loss = (real_data.mean(0) - grad_g_of_y.mean(0)).pow(2).sum().item()

        # variance_difference_loss = (real_data.std(0) - grad_g_of_y.std(0)).pow(2).sum().item()

        if batch_idx % args.log_interval == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)] g_OT_loss: {:.4f} g_Constraint_Loss: {:.4f}  W_2_Loss: {:.4f} '.format(
                epoch, batch_idx * len(real_data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), g_OT_loss_val_batch, g_Constraint_loss_val_batch, w_2_loss_value_batch))
    
    w_2_loss_value_epoch /= len(train_loader)
    g_OT_loss_value_epoch/= len(train_loader)
    g_Constraint_loss_value_epoch /= len(train_loader)

    results.add(epoch=epoch, w2_loss_train_samples=w_2_loss_value_epoch, g_OT_train_loss=g_OT_loss_value_epoch, g_Constraint_loss=g_Constraint_loss_value_epoch)
                    
    results.save()


    return w_2_loss_value_epoch, g_OT_loss_value_epoch, g_Constraint_loss_value_epoch


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
    grid = make_grid(array_img_vectors, nrow=4, normalize=True)
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(results_save_path+'/epoch_{0}.png'.format(epoch))


def drawArrow(A, B):
    plt.arrow(A[0], A[1], B[0] - A[0], B[1] - A[1], color=[0.5,0.5,1], alpha = 0.1)
              #head_width=0.01, length_includes_head=False)


def plot_transported_samples(transported_y, epoch):

    transported_y_np = transported_y.data.cpu().numpy()
    
    plt.figure()
    plt.scatter(initial_y[:,0], initial_y[:,1], color='C1', 
               alpha=0.5, label=r'$Y$')
    plt.scatter(initial_x[:,0], initial_x[:,1], color='C2', 
               alpha=0.5, label=r'$X$')
    plt.scatter(transported_y_np[:,0], transported_y_np[:,1], color='C3', 
               alpha=0.5, label=r'$\nabla g(Y)$')
    
    plt.legend()

    for i in range(args.N_PLOT):
       drawArrow(initial_y[i,:], transported_y_np[i,:])

    plt.savefig(results_save_path+'/epoch_{0}.png'.format(epoch))



###################################################
## Training stuff

total_w_2_epoch_loss_list = []
total_g_OT_epoch_loss_list = []
total_g_Constraint_epoch_loss_list = []

for epoch in range(1, args.epochs + 1):

    # transported_y = compute_optimal_transport_map(y_plot, convex_g)

    # plot_transported_samples(transported_y, epoch)

    w_2_loss_value_epoch, g_OT_loss_value_epoch, g_Constraint_loss_value_epoch = train(epoch)

    total_w_2_epoch_loss_list.append(w_2_loss_value_epoch)
    total_g_OT_epoch_loss_list.append(g_OT_loss_value_epoch)
    total_g_Constraint_epoch_loss_list.append(g_Constraint_loss_value_epoch)


    if epoch % args.lr_schedule == 0:
        
        optimizer_g.param_groups[0]['lr'] = optimizer_g.param_groups[0]['lr'] * 0.5

        optimizer_f.param_groups[0]['lr'] = optimizer_f.param_groups[0]['lr'] * 0.5


    # if epoch % 10 == 0:
    if epoch % 1 == 0:
        torch.save(convex_f.state_dict(), model_save_path + '/convex_f_epoch_{0}.pt'.format(epoch))
        torch.save(convex_g.state_dict(), model_save_path + '/convex_g_epoch_{0}.pt'.format(epoch))
    else:
        torch.save(convex_f.state_dict(), model_save_path + '/convex_f_lastepoch.pt')
        torch.save(convex_g.state_dict(), model_save_path + '/convex_g_lastepoch.pt')


plt.plot(range(1, len(total_w_2_epoch_loss_list) + 1), total_w_2_epoch_loss_list, label='Training loss')
plt.xlabel('iterations')
plt.ylabel(r'$W_2$-loss value')
plt.savefig(results_save_path+'/training_loss.png')
plt.show()


logging.info("Training is finished and the models and plots are saved. Good job :)")



# ## Useful for loading and checkign the distributions once the training is done

# convex_f.load_state_dict(torch.load(model_save_path+'/convex_f.pt'))
# convex_g.load_state_dict(torch.load(model_save_path+'/convex_g.pt'))

# y = torch.randn(args.BATCH_SIZE, args.INPUT_DIM, requires_grad=True)

# x = mu + torch.randn(args.BATCH_SIZE, args.INPUT_DIM)

# if args.cuda:

#     x, y = x.cuda(), y.cuda()

# g_of_y = convex_g(y).sum()

# transport_y = torch.autograd.grad(g_of_y, y, create_graph=True)[0]

# mean_transport_y = transport_y.mean(0)
# mean_x = x.mean(0)

# mean_loss = (mean_transport_y - mean_x).pow(2).sum().item()

# print("Mean loss is:", mean_loss)


#########
### Checking stuff


# if args.cuda:
#     real_data = first_batch_data.cuda()
#
# real_data = Variable(real_data)
#
# y = Variable(torch.randn(args.BATCH_SIZE, args.INPUT_DIM), requires_grad=True)
# if args.cuda:
#     y = y.cuda()
#
# g_of_y = convex_g(y).sum()
#
# grad_g_of_y = torch.autograd.grad(g_of_y, y, create_graph=True)[0]
#
# f_grad_g_y = convex_f(grad_g_of_y).mean()
#
# loss_g = f_grad_g_y - torch.dot(grad_g_of_y.reshape(-1), y.reshape(-1)) / y.size(0)
# loss_g.backward()
#
# g_positive_constraint_loss = compute_constraint_loss(g_positive_params)
# g_positive_constraint_loss.backward()
#
# optimizer_g.step()
#
# for p in list(convex_f.parameters()):
#     p.grad.copy_(-p.grad)
#
# remaining_f_loss = convex_f(real_data).mean()
# remaining_f_loss.backward()
#
# optimizer_f.step()
#
# w_2_loss_value = loss_g.item() - remaining_f_loss.item() + 0.5 * real_data.pow(2).sum(
#     dim=1).mean().item() + 0.5 * y.pow(2).sum(dim=1).mean().item()
#
# for p in f_positive_params:
#     p.data.copy_(torch.relu(p.data))
#
# print(w_2_loss_value)

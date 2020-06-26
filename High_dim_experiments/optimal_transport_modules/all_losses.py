import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn._functions as tnnf
import numpy as np


### This loss is a relaxation of positive constraints on the weights
### Hence we penalize the negative ReLU

def compute_constraint_loss(list_of_params):
    loss_val = 0

    for p in list_of_params:
        loss_val += torch.relu(-p).pow(2).sum()
    return loss_val


# Computes E_y |\nabla f (\nabla g(y)) - y|^2

def convex_fn_inverse_constraint_loss_y_side(convex_f, convex_g, y):

    g_of_y = convex_g(y).sum()

    grad_g_of_y = torch.autograd.grad(g_of_y, y, create_graph=True)[0]

    f_grad_g_of_y = convex_f(grad_g_of_y).sum()

    grad_f_grad_g_of_y = torch.autograd.grad(f_grad_g_of_y, grad_g_of_y, create_graph=True)[0]

    constraint_loss = (grad_f_grad_g_of_y - y).pow(2).sum(dim=1).mean()

    return constraint_loss


# Assumes that x is a vector. 
# Computes E_x |\nabla g (\nabla f(x)) - x|^2

def convex_fn_inverse_constraint_loss_x_side(convex_f, convex_g, real_data):

    x = Variable(real_data, requires_grad=True)

    f_of_x = convex_f(x).sum()

    grad_f_of_x = torch.autograd.grad(f_of_x, x, create_graph=True)[0]

    g_grad_f_of_x = convex_g(grad_f_of_x).sum()

    grad_g_grad_f_of_x = torch.autograd.grad(g_grad_f_of_x, grad_f_of_x, create_graph=True)[0]

    constraint_loss = (grad_g_grad_f_of_x - x).pow(2).sum(dim=1).mean()

    return constraint_loss


# Assumes that both (x, y) are vectors. 
# Computes E_{(x,y)} ReLU(<x, y> - f(x) -g(y))^2

def inequality_young_fenchel_loss(convex_f, convex_g, real_data, y):

    size_of_y = y.shape[1]

    return torch.mean((torch.clamp(( torch.bmm(real_data.view(-1, 1, size_of_y), y.view(-1, size_of_y, 1)).reshape(-1, 1) - convex_f(real_data) - convex_g(y) ), min=0))**2)


# Assumes that both (x, y) are vectors
# Computes E_x |f(x) + g(\nabla f(x)) - <x, \nabla f(x)>|^2 + E_y |g(y) + f(\nabla g(y)) - <y, \nabla g(y)>|^2 

def equality_young_fenchel_loss(grad_g_of_y, f_grad_g_y, real_data, y, convex_g):

    size_of_y = y.shape[1]

    # print(grad_f_of_x.shape, grad_g_of_y.shape)

    y_transport_loss = torch.mean((f_grad_g_y + convex_g(y) - torch.bmm(grad_g_of_y.view(-1, 1, size_of_y), y.view(-1, size_of_y, 1)).reshape(-1, 1) )**2)

    # # This is for x-transport loss. This doesn't completely make sense since 'x' doesn't have a density
    # x = Variable(real_data, requires_grad=True)

    # f_of_x = convex_f(x).sum()

    # grad_f_of_x = torch.autograd.grad(f_of_x, x, create_graph=True)[0]

    # x_transport_loss = torch.mean((convex_g(grad_f_of_x) + convex_f(x) - torch.bmm(grad_f_of_x.view(-1, 1, size_of_y), x.view(-1, size_of_y, 1)).reshape(-1, 1) )**2)

    return y_transport_loss, 0 #x_transport_loss
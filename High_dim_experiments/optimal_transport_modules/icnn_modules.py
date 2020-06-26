import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn._functions as tnnf
import numpy as np


def get_activation(activation):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'elu':
        return nn.ELU()
    elif activation == 'leaky_relu':
        return nn.LeakyReLU(0.2)
    elif activation == 'celu':
        return nn.CELU()
    elif activation == 'selu':
        return nn.SELU()
    elif activation == 'tanh':
        return nn.Tanh()
    else:
        raise NotImplementedError('activation [%s] is not found' % activation)


class ConvexLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        # self.wb = kargs[0]
        # self.have_alpha = kargs[1]
        #
        # kargs = kargs[2:]

        super(ConvexLinear, self).__init__(*kargs, **kwargs)

       #self.weight.data.copy_(torch.abs(self.weight.data))

        if not hasattr(self.weight, 'be_positive'):
            self.weight.be_positive = 1.0

    def forward(self, input):

        out = nn.functional.linear(input, self.weight, self.bias)

        return out


class ConvexConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):

        super(ConvexConv2d, self).__init__(*kargs, **kwargs)

        if not hasattr(self.weight, 'be_positive'):
            self.weight.be_positive = 1.0

    def forward(self, input):

        out = nn.functional.conv2d(input, self.weight, self.bias, self.stride,
                                   self.padding, self.dilation, self.groups)
        return out


class Simple_Feedforward_2Layer_ICNN_LastInp_Quadratic(nn.Module):

    def __init__(self, input_dim, hidden_dim, activation):

        super(Simple_Feedforward_2Layer_ICNN_LastInp_Quadratic, self).__init__()

        # For now I am hardcoding it as three hiden layers
        # x, h_1, h_2, h_3, f(x)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation

        # x -> h_1        
        self.fc1_normal = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.activ_1 = get_activation(self.activation)

        self.fc2_normal = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.fc2_convex = ConvexLinear(self.hidden_dim, self.hidden_dim, bias=False)
        self.activ_2 = get_activation(self.activation)

        self.last_convex = ConvexLinear(self.hidden_dim, 1, bias=False)
        self.last_linear = nn.Linear(self.input_dim, 1, bias=True)
        # We might also not want to include an activation in the last  layer.
        #self.activ_4 = nn.LeakyReLU(0.2)


    # Input is of size
    def forward(self, input):

        # input = input.view(-1, self.input_dim)

        # input_image = input.view(-1, 1, 28, 28)

        x = self.activ_1(self.fc1_normal(input)).pow(2)

        x = self.activ_2(self.fc2_convex(x).add(self.fc2_normal(input)))

        x = self.last_convex(x).add(self.last_linear(input).pow(2))

        return x


class Simple_Feedforward_2Layer_ICNN_LastFull_Quadratic(nn.Module):

    def __init__(self, input_dim, hidden_dim, activation):

        super(Simple_Feedforward_2Layer_ICNN_LastFull_Quadratic, self).__init__()

        # For now I am hardcoding it as three hiden layers
        # x, h_1, h_2, h_3, f(x)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation

        # x -> h_1        
        self.fc1_normal = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.activ_1 = get_activation(self.activation)

        self.fc2_normal = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.fc2_convex = ConvexLinear(self.hidden_dim, self.hidden_dim, bias=False)
        self.activ_2 = get_activation(self.activation)

        self.last_convex = ConvexLinear(self.hidden_dim, 1, bias=False)
        self.last_linear = nn.Linear(self.input_dim, 1, bias=True)
        # We might also not want to include an activation in the last  layer.
        #self.activ_4 = nn.LeakyReLU(0.2)


    # Input is of size
    def forward(self, input):

        # input = input.view(-1, self.input_dim)

        # input_image = input.view(-1, 1, 28, 28)

        x = self.activ_1(self.fc1_normal(input)).pow(2)

        x = self.activ_2(self.fc2_convex(x).add(self.fc2_normal(input)))

        x = self.last_convex(x).add(self.last_linear(input)).pow(2)

        return x


class Simple_Feedforward_3Layer_ICNN_LastInp_Quadratic(nn.Module):

    def __init__(self, input_dim, hidden_dim, activation):

        super(Simple_Feedforward_3Layer_ICNN_LastInp_Quadratic, self).__init__()

        # For now I am hardcoding it as three hiden layers
        # x, h_1, h_2, h_3, f(x)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation

        # x -> h_1        
        self.fc1_normal = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        #self.dense1_bn = nn.BatchNorm1d(self.hidden_dim)
        self.activ_1 = get_activation(self.activation)

        self.fc2_normal = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.fc2_convex = ConvexLinear(self.hidden_dim, self.hidden_dim, bias=False)
        self.activ_2 = get_activation(self.activation)

        self.fc3_normal = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.fc3_convex = ConvexLinear(self.hidden_dim, self.hidden_dim, bias=False)
        self.activ_3 = get_activation(self.activation)

        self.last_convex = ConvexLinear(self.hidden_dim, 1, bias=False)
        self.last_linear = nn.Linear(self.input_dim, 1, bias=True)

        # self.dense2_bn = nn.BatchNorm1d(self.hidden_dim)
        # self.dense3_bn = nn.BatchNorm1d(self.hidden_dim)
        
        # We might also not want to include an activation in the last  layer.
        


    # Input is of size
    def forward(self, input):

        # input = input.view(-1, self.input_dim)

        # input_image = input.view(-1, 1, 28, 28)

        #x = self.activ_1(self.dense1_bn(self.fc1_normal(input))).pow(2)
        x = self.activ_1(self.fc1_normal(input)).pow(2)

        x = self.activ_2(self.fc2_convex(x).add(self.fc2_normal(input)))

        x = self.activ_3(self.fc3_convex(x).add(self.fc3_normal(input)))

        x = self.last_convex(x).add(self.last_linear(input).pow(2))

        return x

class Simple_Feedforward_3Layer_ICNN_LastInp_Quadratic_LastLayerCeLU(nn.Module):

    def __init__(self, input_dim, hidden_dim, activation):

        super(Simple_Feedforward_3Layer_ICNN_LastInp_Quadratic_LastLayerCeLU, self).__init__()

        # For now I am hardcoding it as three hiden layers
        # x, h_1, h_2, h_3, f(x)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation

        # x -> h_1        
        self.fc1_normal = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.activ_1 = get_activation(self.activation)

        self.fc2_normal = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.fc2_convex = ConvexLinear(self.hidden_dim, self.hidden_dim, bias=False)
        self.activ_2 = get_activation(self.activation)

        self.fc3_normal = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.fc3_convex = ConvexLinear(self.hidden_dim, self.hidden_dim, bias=False)
        self.activ_3 = get_activation(self.activation)

        self.last_convex = ConvexLinear(self.hidden_dim, 1, bias=False)
        self.last_linear = nn.Linear(self.input_dim, 1, bias=True)
        # We might also not want to include an activation in the last  layer.
        self.activ_4 = get_activation(self.activation)


    # Input is of size
    def forward(self, input):

        # input = input.view(-1, self.input_dim)

        # input_image = input.view(-1, 1, 28, 28)

        x = self.activ_1(self.fc1_normal(input)).pow(2)

        x = self.activ_2(self.fc2_convex(x).add(self.fc2_normal(input)))

        x = self.activ_3(self.fc3_convex(x).add(self.fc3_normal(input)))

        x = self.activ_4(self.last_convex(x).add(self.last_linear(input).pow(2)))

        return x

class Simple_Feedforward_3Layer_ICNN_LastLayerCeLU(nn.Module):

    def __init__(self, input_dim, hidden_dim, activation):

        super(Simple_Feedforward_3Layer_ICNN_LastLayerCeLU, self).__init__()

        # For now I am hardcoding it as three hiden layers
        # x, h_1, h_2, h_3, f(x)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation

        # x -> h_1        
        self.fc1_normal = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.activ_1 = get_activation(self.activation)

        self.fc2_normal = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.fc2_convex = ConvexLinear(self.hidden_dim, self.hidden_dim, bias=False)
        self.activ_2 = get_activation(self.activation)

        self.fc3_normal = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.fc3_convex = ConvexLinear(self.hidden_dim, self.hidden_dim, bias=False)
        self.activ_3 = get_activation(self.activation)

        self.last_convex = ConvexLinear(self.hidden_dim, 1, bias=False)
        self.last_linear = nn.Linear(self.input_dim, 1, bias=True)
        # We might also not want to include an activation in the last  layer.
        self.activ_4 = get_activation(self.activation)


    # Input is of size
    def forward(self, input):

        # input = input.view(-1, self.input_dim)

        # input_image = input.view(-1, 1, 28, 28)

        x = self.activ_1(self.fc1_normal(input)).pow(2)

        x = self.activ_2(self.fc2_convex(x).add(self.fc2_normal(input)))

        x = self.activ_3(self.fc3_convex(x).add(self.fc3_normal(input)))

        x = self.activ_4(self.last_convex(x).add(self.last_linear(input)))

        return x


class Simple_Feedforward_3Layer_ICNN_LastFull_Quadratic(nn.Module):

    def __init__(self, input_dim, hidden_dim, activation):

        super(Simple_Feedforward_3Layer_ICNN_LastFull_Quadratic, self).__init__()

        # For now I am hardcoding it as three hiden layers
        # x, h_1, h_2, h_3, f(x)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation

        # x -> h_1        
        self.fc1_normal = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.activ_1 = get_activation(self.activation)

        self.fc2_normal = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.fc2_convex = ConvexLinear(self.hidden_dim, self.hidden_dim, bias=False)
        self.activ_2 = get_activation(self.activation)

        self.fc3_normal = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.fc3_convex = ConvexLinear(self.hidden_dim, self.hidden_dim, bias=False)
        self.activ_3 = get_activation(self.activation)

        self.last_convex = ConvexLinear(self.hidden_dim, 1, bias=False)
        self.last_linear = nn.Linear(self.input_dim, 1, bias=True)
        # We might also not want to include an activation in the last  layer.
        #self.activ_4 = nn.LeakyReLU(0.2)


    # Input is of size
    def forward(self, input):

        # input = input.view(-1, self.input_dim)

        # input_image = input.view(-1, 1, 28, 28)

        x = self.activ_1(self.fc1_normal(input)).pow(2)

        x = self.activ_2(self.fc2_convex(x).add(self.fc2_normal(input)))

        x = self.activ_3(self.fc3_convex(x).add(self.fc3_normal(input)))

        x = self.last_convex(x).add(self.last_linear(input)).pow(2)

        return x


class Simple_Feedforward_4Layer_ICNN_LastInp_Quadratic(nn.Module):

    def __init__(self, input_dim, hidden_dim, activation):

        super(Simple_Feedforward_4Layer_ICNN_LastInp_Quadratic, self).__init__()

        # For now I am hardcoding it as three hiden layers
        # x, h_1, h_2, h_3, f(x)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation

        # x -> h_1        
        self.fc1_normal = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.activ_1 = get_activation(self.activation)

        self.fc2_normal = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.fc2_convex = ConvexLinear(self.hidden_dim, self.hidden_dim, bias=False)
        self.activ_2 = get_activation(self.activation)

        self.fc3_normal = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.fc3_convex = ConvexLinear(self.hidden_dim, self.hidden_dim, bias=False)
        self.activ_3 = get_activation(self.activation)

        self.fc4_normal = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.fc4_convex = ConvexLinear(self.hidden_dim, self.hidden_dim, bias=False)
        self.activ_4 = get_activation(self.activation)

        self.last_convex = ConvexLinear(self.hidden_dim, 1, bias=False)
        self.last_linear = nn.Linear(self.input_dim, 1, bias=True)
        # We might also not want to include an activation in the last  layer.
        #self.activ_4 = nn.LeakyReLU(0.2)


    # Input is of size
    def forward(self, input):

        # input = input.view(-1, self.input_dim)

        # input_image = input.view(-1, 1, 28, 28)

        x = self.activ_1(self.fc1_normal(input)).pow(2)

        x = self.activ_2(self.fc2_convex(x).add(self.fc2_normal(input)))

        x = self.activ_3(self.fc3_convex(x).add(self.fc3_normal(input)))

        x = self.activ_4(self.fc4_convex(x).add(self.fc4_normal(input)))

        x = self.last_convex(x).add(self.last_linear(input).pow(2))

        return x


class Simple_Feedforward_4Layer_ICNN_LastFull_Quadratic(nn.Module):

    def __init__(self, input_dim, hidden_dim, activation):

        super(Simple_Feedforward_4Layer_ICNN_LastFull_Quadratic, self).__init__()

        # For now I am hardcoding it as three hiden layers
        # x, h_1, h_2, h_3, f(x)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation

        # x -> h_1        
        self.fc1_normal = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.activ_1 = get_activation(self.activation)

        self.fc2_normal = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.fc2_convex = ConvexLinear(self.hidden_dim, self.hidden_dim, bias=False)
        self.activ_2 = get_activation(self.activation)

        self.fc3_normal = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.fc3_convex = ConvexLinear(self.hidden_dim, self.hidden_dim, bias=False)
        self.activ_3 = get_activation(self.activation)

        self.fc4_normal = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.fc4_convex = ConvexLinear(self.hidden_dim, self.hidden_dim, bias=False)
        self.activ_4 = get_activation(self.activation)

        self.last_convex = ConvexLinear(self.hidden_dim, 1, bias=False)
        self.last_linear = nn.Linear(self.input_dim, 1, bias=True)
        # We might also not want to include an activation in the last  layer.
        #self.activ_4 = nn.LeakyReLU(0.2)


    # Input is of size
    def forward(self, input):

        # input = input.view(-1, self.input_dim)

        # input_image = input.view(-1, 1, 28, 28)

        x = self.activ_1(self.fc1_normal(input)).pow(2)

        x = self.activ_2(self.fc2_convex(x).add(self.fc2_normal(input)))

        x = self.activ_3(self.fc3_convex(x).add(self.fc3_normal(input)))

        x = self.activ_4(self.fc4_convex(x).add(self.fc4_normal(input)))

        x = self.last_convex(x).add(self.last_linear(input)).pow(2)

        return x


class Simple_Feedforward_5Layer_ICNN_LastInp_Quadratic(nn.Module):

    def __init__(self, input_dim, hidden_dim, activation):

        super(Simple_Feedforward_5Layer_ICNN_LastInp_Quadratic, self).__init__()

        # For now I am hardcoding it as three hiden layers
        # x, h_1, h_2, h_3, f(x)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation

        # x -> h_1        
        self.fc1_normal = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.activ_1 = get_activation(self.activation)

        self.fc2_normal = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.fc2_convex = ConvexLinear(self.hidden_dim, self.hidden_dim, bias=False)
        self.activ_2 = get_activation(self.activation)

        self.fc3_normal = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.fc3_convex = ConvexLinear(self.hidden_dim, self.hidden_dim, bias=False)
        self.activ_3 = get_activation(self.activation)

        self.fc4_normal = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.fc4_convex = ConvexLinear(self.hidden_dim, self.hidden_dim, bias=False)
        self.activ_4 = get_activation(self.activation)

        self.fc5_normal = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.fc5_convex = ConvexLinear(self.hidden_dim, self.hidden_dim, bias=False)
        self.activ_5 = get_activation(self.activation)

        self.last_convex = ConvexLinear(self.hidden_dim, 1, bias=False)
        self.last_linear = nn.Linear(self.input_dim, 1, bias=True)
        # We might also not want to include an activation in the last  layer.
        #self.activ_4 = nn.LeakyReLU(0.2)


    # Input is of size
    def forward(self, input):

        # input = input.view(-1, self.input_dim)

        # input_image = input.view(-1, 1, 28, 28)

        x = self.activ_1(self.fc1_normal(input)).pow(2)

        x = self.activ_2(self.fc2_convex(x).add(self.fc2_normal(input)))

        x = self.activ_3(self.fc3_convex(x).add(self.fc3_normal(input)))

        x = self.activ_4(self.fc4_convex(x).add(self.fc4_normal(input)))

        x = self.activ_5(self.fc5_convex(x).add(self.fc5_normal(input)))

        x = self.last_convex(x).add(self.last_linear(input).pow(2))

        return x


class Simple_Feedforward_5Layer_ICNN_LastFull_Quadratic(nn.Module):

    def __init__(self, input_dim, hidden_dim, activation):

        super(Simple_Feedforward_5Layer_ICNN_LastFull_Quadratic, self).__init__()

        # For now I am hardcoding it as three hiden layers
        # x, h_1, h_2, h_3, f(x)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation

        # x -> h_1        
        self.fc1_normal = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.activ_1 = get_activation(self.activation)

        self.fc2_normal = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.fc2_convex = ConvexLinear(self.hidden_dim, self.hidden_dim, bias=False)
        self.activ_2 = get_activation(self.activation)

        self.fc3_normal = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.fc3_convex = ConvexLinear(self.hidden_dim, self.hidden_dim, bias=False)
        self.activ_3 = get_activation(self.activation)

        self.fc4_normal = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.fc4_convex = ConvexLinear(self.hidden_dim, self.hidden_dim, bias=False)
        self.activ_4 = get_activation(self.activation)

        self.fc5_normal = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.fc5_convex = ConvexLinear(self.hidden_dim, self.hidden_dim, bias=False)
        self.activ_5 = get_activation(self.activation)

        self.last_convex = ConvexLinear(self.hidden_dim, 1, bias=False)
        self.last_linear = nn.Linear(self.input_dim, 1, bias=True)
        # We might also not want to include an activation in the last  layer.
        #self.activ_4 = nn.LeakyReLU(0.2)


    # Input is of size
    def forward(self, input):

        # input = input.view(-1, self.input_dim)

        # input_image = input.view(-1, 1, 28, 28)

        x = self.activ_1(self.fc1_normal(input)).pow(2)

        x = self.activ_2(self.fc2_convex(x).add(self.fc2_normal(input)))

        x = self.activ_3(self.fc3_convex(x).add(self.fc3_normal(input)))

        x = self.activ_4(self.fc4_convex(x).add(self.fc4_normal(input)))

        x = self.activ_5(self.fc5_convex(x).add(self.fc5_normal(input)))

        x = self.last_convex(x).add(self.last_linear(input)).pow(2)

        return x


class my_own_Conv_ICNN_with_skip(nn.Module):

    def __init__(self, activation):

        super(my_own_Conv_ICNN_with_skip, self).__init__()

        # input_dim = 784 for all MNIST experiments

        # For now I am hardcoding it as three hiden layers
        # x, h_1, h_2, h_3, f(x)

        # x -> h_1
        self.conv1_normal = nn.Conv2d(1, 20, 5, 1, bias=True)
        self.activ_1_squared = get_activation(activation)
        self.max_pool1 = nn.MaxPool2d(2, 2)

        # (x,h_1) -> h_2
        self.conv2_normal = nn.Conv2d(1, 50, 14, 2, bias=True)
        self.conv2_convex = ConvexConv2d(20, 50, 5, 1, bias=False)
        self.activ_2 = get_activation(activation)
        self.max_pool2 = nn.MaxPool2d(2, 2)

        # h_3 -> f(x) \in \reals
        self.fc4_normal = nn.Linear(784, 500, bias=True)
        self.fc4_convex = ConvexLinear(4 * 4 * 50, 500, bias=False)
        self.activ_4 = get_activation(activation)

        self.last_convex = ConvexLinear(500, 1, bias=False)
        self.last_linear = nn.Linear(784, 1, bias=True)
        # We might also not want to include an activation in the last  layer.
        #self.activ_4 = nn.LeakyReLU(0.2)


    # Input is of size
    def forward(self, input):

        # input = input.view(-1, self.input_dim)

        input_image = input.view(-1, 1, 28, 28)

        x = self.conv1_normal(input_image)
        x = self.activ_1_squared(x).pow(2)
        x = self.max_pool1(x)

        # print(x.size())

        x = self.conv2_normal(input_image).add(self.conv2_convex(x))
        x = self.activ_2(x)
        x = self.max_pool2(x)

        # print(x.size())

        x = x.view(-1, 4 * 4 * 50)
        #input = input.view(-1, 784)
        # print(input.size())
        x = self.activ_4(self.fc4_normal(input).add(self.fc4_convex(x)))

        x = self.last_convex(x).add(self.last_linear(input))

        return x


class LeNet_ICNN_without_skip(nn.Module):

    def __init__(self, activation):

        super(LeNet_ICNN_without_skip, self).__init__()

        # input_dim = 784 for all MNIST experiments

        # For now I am hardcoding it as three hiden layers
        # x, h_1, h_2, h_3, f(x)

        # x -> h_1
        self.conv1_normal = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
        self.activ_1_squared = get_activation(activation)
        self.max_pool1 = nn.MaxPool2d(2, 2)

        # (x,h_1) -> h_2
        # self.conv2_normal = nn.Conv2d(1, 50, 14, 2, bias=True)
        self.conv2_convex = ConvexConv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
        self.activ_2 = get_activation(activation)
        self.max_pool2 = nn.MaxPool2d(2, 2)

        # h_3 -> f(x) \in \reals
        # self.fc4_normal = nn.Linear(784, 500, bias=True)
        self.fc1_convex = ConvexLinear(5 * 5 * 16, 120, bias=True)
        self.activ_fc1 = get_activation(activation)

        self.fc2_convex = ConvexLinear(120, 84, bias=True)
        self.activ_fc2 = get_activation(activation)

        self.last_convex = ConvexLinear(84, 1, bias=True)
        self.last_linear = nn.Linear(784, 1, bias=True)
        # We might also not want to include an activation in the last  layer.
        #self.activ_4 = nn.LeakyReLU(0.2)


    # Input is of size
    def forward(self, input):

        # input = input.view(-1, self.input_dim)

        input_image = input.view(-1, 1, 28, 28)

        x = self.conv1_normal(input_image)
        x = self.activ_1_squared(x).pow(2)
        x = self.max_pool1(x)

        # print(x.size())

        x = self.conv2_convex(x)
        x = self.activ_2(x)
        x = self.max_pool2(x)

        # print(x.size())

        x = x.view(-1, 5 * 5 * 16)
        #input = input.view(-1, 784)
        # print(input.size())
        x = self.activ_fc1(self.fc1_convex(x))

        x = self.activ_fc2(self.fc2_convex(x))

        x = self.last_convex(x).add(self.last_linear(input))

        return x


class LeNet_ICNN_with_skip(nn.Module):

    def __init__(self, activation):

        super(LeNet_ICNN_with_skip, self).__init__()

        # input_dim = 784 for all MNIST experiments

        # For now I am hardcoding it as three hiden layers
        # x, h_1, h_2, h_3, f(x)

        # x -> h_1
        self.conv1_normal = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
        self.activ_1_squared = get_activation(activation)
        self.max_pool1 = nn.MaxPool2d(2, 2)

        # (x,h_1) -> h_2
        # self.conv2_normal = nn.Conv2d(1, 50, 14, 2, bias=True)
        self.conv2_convex = ConvexConv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=False)
        self.conv2_normal = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=9, stride=2, padding=0, bias=True)
        self.activ_2 = get_activation(activation)
        self.max_pool2 = nn.MaxPool2d(2, 2)

        # h_3 -> f(x) \in \reals
        # self.fc4_normal = nn.Linear(784, 500, bias=True)
        self.fc1_convex = ConvexLinear(5 * 5 * 16, 120, bias=False)
        self.fc1_normal = nn.Linear(784, 120, bias=True)
        self.activ_fc1 = get_activation(activation)

        self.fc2_convex = ConvexLinear(120, 84, bias=False)
        self.fc2_normal = nn.Linear(784, 84, bias=True)
        self.activ_fc2 = get_activation(activation)

        self.last_convex = ConvexLinear(84, 1, bias=False)
        self.last_linear = nn.Linear(784, 1, bias=True)
        # We might also not want to include an activation in the last  layer.
        #self.activ_4 = nn.LeakyReLU(0.2)


    # Input is of size
    def forward(self, input):

        # input = input.view(-1, self.input_dim)

        input_image = input.view(-1, 1, 28, 28)

        x = self.conv1_normal(input_image)
        x = self.activ_1_squared(x).pow(2)
        x = self.max_pool1(x)

        # print(x.size())

        x = self.conv2_normal(input_image).add(self.conv2_convex(x))
        x = self.activ_2(x)
        x = self.max_pool2(x)

        # print(x.size())

        x = x.view(-1, 5 * 5 * 16)
        #input = input.view(-1, 784)
        # print(input.size())
        x = self.activ_fc1(self.fc1_convex(x).add(self.fc1_normal(input)))

        x = self.activ_fc2(self.fc2_convex(x).add(self.fc2_normal(input)))

        x = self.last_convex(x).add(self.last_linear(input))

        return x

class Simple_Feedforward_3Layer_NN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, activation):

        super(Simple_Feedforward_3Layer_NN, self).__init__()

        # For now I am hardcoding it as three hiden layers
        # x, h_1, h_2, h_3, f(x)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.output_dim = output_dim

        # x -> h_1        
        self.fc1_normal = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.activ_1 = get_activation(self.activation)

        self.fc2_normal = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.activ_2 = get_activation(self.activation)

        self.fc3_normal = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.activ_3 = get_activation(self.activation)

        self.last_linear = nn.Linear(self.hidden_dim, self.output_dim, bias=True)
        # We might also not want to include an activation in the last  layer.
        #self.activ_4 = nn.LeakyReLU(0.2)


    # Input is of size
    def forward(self, input):

        # input = input.view(-1, self.input_dim)

        # input_image = input.view(-1, 1, 28, 28)

        x = self.activ_1(self.fc1_normal(input))

        x = self.activ_2(self.fc2_normal(x))

        x = self.activ_3(self.fc3_normal(x))

        x = self.last_linear(x)

        return x

class Simple_quadratic(nn.Module):

    def __init__(self, input_dim, output_dim):

        super(Simple_quadratic, self).__init__()

        # For now I am hardcoding it as three hiden layers
        # x, h_1, h_2, h_3, f(x)

        self.input_dim = input_dim
        self.output_dim = output_dim

    # Input is of size
    def forward(self, input):

        # input = input.view(-1, self.input_dim)

        # input_image = input.view(-1, 1, 28, 28)

        x = 0.5*input.pow(2).sum(dim=1,keepdim=True)

        return x
'''
class Simple_Feedforward_3Layer_ICNN_All_Quadratic_New(nn.Module):

    def __init__(self, input_dim, hidden_dim, activation):

        super(Simple_Feedforward_3Layer_ICNN_All_Quadratic_New, self).__init__()

        # For now I am hardcoding it as three hiden layers
        # x, h_1, h_2, h_3, f(x)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation

        # x -> h_1        
        self.quad_1 = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.convex_quad_1 = ConvexLinear(self.hidden_dim, self.hidden_dim, bias=False)

        self.quad_2 = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.convex_quad_2 = ConvexLinear(self.hidden_dim, self.hidden_dim, bias=False)
        self.convex_2 = ConvexLinear(self.hidden_dim, self.hidden_dim, bias=True)

        self.quad_3 = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.convex_quad_3 = ConvexLinear(self.hidden_dim, self.hidden_dim, bias=False)
        self.convex_3 = ConvexLinear(self.hidden_dim, self.hidden_dim, bias=True)

        self.quad_last = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.convex_quad_last = ConvexLinear(self.hidden_dim, 1, bias=False)
        self.convex_last = ConvexLinear(self.hidden_dim, 1, bias=False)

        self.activ = get_activation(self.activation)



    # Input is of size
    def forward(self, input):

        # input = input.view(-1, self.input_dim)

        # input_image = input.view(-1, 1, 28, 28)

        h = self.convex_quad_1(self.quad_1(input)).pow(2))/self.hidden_dim

        h = self.convex_quad_2(self.quad_2(input)).pow(2))/self.hidden_dim +  self.convex_2(self.active(h))/self.hidden_dim

        h = self.convex_quad_3(self.quad_3(input)).pow(2))/self.hidden_dim +  self.convex_3(self.active(h))/self.hidden_dim

        f = self.convex_quad_last(self.quad_last(input)).pow(2))/self.hidden_dim +  self.convex_last(self.active(h))/self.hidden_dim

        return x
'''
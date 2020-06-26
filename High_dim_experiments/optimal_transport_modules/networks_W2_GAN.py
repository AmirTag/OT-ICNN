import torch
import torch.nn as nn
from torch import optim

def get_optim(parameters, lr, config):
    return optim.Adam(parameters, lr, [config.beta1, config.beta2])

def get_d(config):
    net = DUAL(config.n_hidden, config.d_n_layers, config.activation)
    net.apply(weights_init_d)
    if torch.cuda.is_available():
        net.cuda()
    return net
# args.input_dim, args.num_neurons, args.activation
def get_g(n_input, n_hidden, n_layers, activation, residual=0, norm='batch'):
    # residual = (config.solver != 'bary_ot')
    net = GEN(n_input, n_hidden, n_layers, activation, residual, norm)
    if residual:
        net.apply(weights_init_g)
    # if torch.cuda.is_available():
    #     net.cuda()
    return net

class DUAL(nn.Module):
    def __init__(self, n_hidden, n_layers, activation):
        super(DUAL, self).__init__()
        in_h = 28*28
        out_h = n_hidden
        modules = []
        for i in range(n_layers):
            modules.append(nn.Linear(in_h, out_h))
            modules.append(get_activation(activation))
            in_h = out_h
            out_h = n_hidden
        modules.append(nn.Linear(n_hidden, 1))
        self.model = nn.Sequential(*modules)

    def forward(self, input):
        input = input.view(input.size(0), -1)
        output = self.model(input)
        return output.squeeze()

class GEN(nn.Module):
    def __init__(self, n_input, n_hidden, n_layers, activation, residual, norm):
        super(GEN, self).__init__()
        self.residual = residual
        in_h = n_input
        out_h = n_hidden
        modules = []
        for i in range(n_layers):
            m = nn.Linear(in_h, out_h)
            norm_ms = apply_normalization(norm, out_h, m)
            for nm in norm_ms:
                modules.append(nm)
            modules.append(get_activation(activation))
            in_h = out_h
            out_h = n_hidden
        modules.append(nn.Linear(n_hidden, n_input))
        modules.append(nn.Tanh())
        self.model = nn.Sequential(*modules)

    def forward(self, input):
        output = self.model(input.view(input.size(0), -1))
        output = output.view(*input.size())
        return 2*output + torch.clamp(input, min=-1, max=1) if self.residual else output

def get_activation(ac):
    if ac == 'relu':
        return nn.ReLU()
    elif ac == 'elu':
        return nn.ELU()
    elif ac == 'leaky_relu':
        return nn.LeakyReLU(0.2)
    elif ac == 'tanh':
        return nn.Tanh()
    else:
        raise NotImplementedError('activation [%s] is not found' % ac)

def apply_normalization(norm, dim, module):
    """
    Applies normalization `norm` to `module`.
    Optionally uses `dim`
    Returns a list of modules.
    """
    if norm == 'none':
        return [module]
    elif norm == 'batch':
        return [module, nn.BatchNorm1d(dim)]
    elif norm == 'layer':
        return [module, nn.GroupNorm(1, dim)]
    elif norm == 'spectral':
        return [torch.nn.utils.spectral_norm(module, name='weight')]
    else:
        raise NotImplementedError('normalization [%s] is not found' % norm)

def weights_init_g(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.fill_(0)

def weights_init_d(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.1)
        m.bias.data.fill_(0)

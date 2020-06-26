import os
import time
import utils
import torch
import losses
import numpy as np
import random
import glob

from torch.backends import cudnn
from data_loader import get_loader
from w1_model import W1
from w2_model import W2
from bot_model import BaryOT
from options import Options
from tensorboardX import SummaryWriter


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import torch
import os
from torch.autograd import Variable
from sklearn.utils.linear_assignment_ import linear_assignment

from collections import OrderedDict

DISPLAY_NUM = 2048

def visualize_single_plotting(x, y, fx, path, data_range=(-2, 2)):
    #fig, ax = plt.subplots()
    #scatter_ax(ax, x=x, y=y, fx=fx, c_x='C1', c_y='C2', c_l='C3', data_range=data_range)
    N_plot = 512
    plt.scatter(x[:N_plot,0],x[:N_plot,1],color='C1', alpha = 0.4)
    plt.scatter(y[:N_plot,0],y[:N_plot,1],color='C2', alpha = 0.4)
    plt.scatter(fx[:N_plot,0],fx[:N_plot,1],color='C3', alpha = 0.4)
    for i in range(len(x[:N_plot,0])):
        plt.arrow(x[i,0], x[i,1], fx[i,0] - x[i,0], fx[i,1] - x[i,1], color='C0', alpha = 0.1, head_width=0.001, width = 0.001)
    
    plt.scatter([],[], color='C1', label = 'Source')
    plt.scatter([],[], color='C2', label = 'Target')
    plt.scatter([],[], color='C3', label= 'Transp.')

    plt.xticks([],'')
    plt.yticks([],'')


    #plt.legend(loc=3, bbox_to_anchor=(-0.015, 0.99, 1.0, 1.2), ncol=3, fontsize=15)
    
    plt.savefig(path) #, bbox_inches='tight')
    plt.clf()

def scatter_ax(ax, x, y, fx, c_x, c_y, c_l, data_range):
    data_min = data_range[0]
    data_max = data_range[1]
    ax.scatter(x[:, 0], x[:, 1], s=1, c=c_x)
    ax.scatter(y[:, 0], y[:, 1], s=1, c=c_y)
    if fx is not None:
        ax.scatter(fx[:, 0], fx[:, 1], s=1, c=c_l)  
        for i in range(DISPLAY_NUM):
            ax.arrow(x[i, 0], x[i, 1], fx[i, 0]-x[i, 0], fx[i, 1]-x[i, 1],
                     head_width=0.03, head_length=0.05, color=[0.5,0.5,1], alpha = 0.1) # fc=c_l, ec=c_l)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_xlim(data_min, data_max)
    ax.set_ylim(data_min, data_max)
    

def get_visuals(model, config):
    gz = model.g(model.fixed_z) if config.gen else model.fixed_z
    x, y = model.fixed_r, gz
    tx, ty = model.get_tx(x), model.get_tx(y, reverse=True)
    images = OrderedDict([('X', x),
                          ('TX', tx),
                          ('Y', y),
                          ('TY', ty)])
    if config.gen:
        images['ZY'] = model.fixed_z
    return images

def visualize_iter_plotting(images, dir, step, config, data_range=(-2, 2)):
    """ visualization for 2D experiment in separate images """
    """ here we assume the following coding and colors:
    X  :  real data  (green color)
    Y  :  fake data  (red color)
    ZY :  noise data (magenta color)
    """
    x, y = to_data(images['X']), to_data(images['Y'])
    fx, fy = to_data(images['TX']), to_data(images['TY'])
    fig, ax = plt.subplots()
    scatter_ax(ax, x=x, y=y, fx=fx, c_x='g', c_y='r', c_l='k', data_range=data_range)
    plt.savefig(os.path.join(dir, 'tx_%06d.png' % (step)), bbox_inches='tight')
    plt.clf()

    fig, ax = plt.subplots()
    scatter_ax(ax, x=y, y=x, fx=fy, c_x='r', c_y='g', c_l='k', data_range=data_range)
    plt.savefig(os.path.join(dir, 'ty_%06d.png' % (step)), bbox_inches='tight')
    plt.clf()

    if config.gen:
        z = to_data(images['ZY'])
        fig, ax = plt.subplots()
        scatter_ax(ax, x=z, y=x, fx=y, c_x='m', c_y='g', c_l='0.5', data_range=data_range)
        plt.savefig(os.path.join(dir, 'gz_%06d.png' % (step)), bbox_inches='tight')
        plt.clf()
    return

gen = 1

config = Options().parse()
config.batch_size = DISPLAY_NUM    # For plotting purposes
config.gen = gen
utils.print_opts(config)

config.solver = 'bary_ot'
#config.solver = 'w1'
#config.solver = 'w2'
#plot_dataset = 'our_checkerboard'
plot_dataset = '8gaussians'
config.data = plot_dataset
config.trial = 3

dir_string = './{0}_{1}/trial_{2}/'.format(config.solver, config.data, config.trial) if config.solver != 'w2' else \
                            './{0}_gen{2}_{1}/trial_{3}/'.format(config.solver, config.data, config.gen, config.trial)

print(dir_string)
exp_dir = dir_string

# ## set up folders
# if config.solver == 'w1':
#     exp_dir = './W1GAN_our_{0}'.format(plot_dataset)   
# if config.solver == 'w2':
#     exp_dir = './w2_gen_{0}_our_{1}'.format(gen,plot_dataset)
# if config.solver == 'bary_ot':
#     exp_dir = './bary_ot_our_{0}'.format(plot_dataset)   


model_dir = exp_dir +'/models'
img_dir = exp_dir +'/images'

## initialize data loaders & model
r_loader, z_loader = get_loader(config)

if config.solver == 'w1':
    model = W1(config, r_loader, z_loader)
elif config.solver == 'w2':
    model = W2(config, r_loader, z_loader)
elif config.solver == 'bary_ot':
    model = BaryOT(config, r_loader, z_loader)

cudnn.benchmark = True
networks = model.get_networks(config)
utils.print_networks(networks)

model.phi.load_state_dict(torch.load(model_dir+'/phi.pkl'))
model.phi.eval()
if config.solver == 'w1':
    model.g.load_state_dict(torch.load(model_dir+'/gen.pkl'))
    model.g.eval()

if config.solver == 'w2':
    model.eps.load_state_dict(torch.load(model_dir+'/eps.pkl'))
    model.eps.eval()
    if gen:
        model.g.load_state_dict(torch.load(model_dir+'/gen.pkl'))
        model.g.eval()

if config.solver == 'bary_ot':
    model.psi.load_state_dict(torch.load(model_dir+'/psi.pkl'))
    model.psi.eval()
    model.g.load_state_dict(torch.load(model_dir+'/gen.pkl'))
    model.g.eval()



fixed_r, fixed_z = model.get_fixed_data()
#visualize_single_plotting(utils.to_data(fixed_z), utils.to_data(fixed_r), None,
#                      exp_dir+'/fixed_source_target_data.png')

# After passing through the model data
images = get_visuals(model, config)

# If there is generator, we should use g(fixed_z)= y for the transported data.
# Otherwise we should use 'TY'
visualize_single_plotting(utils.to_data(fixed_z), utils.to_data(fixed_r), utils.to_data(images['Y']) if gen else utils.to_data(images['TY']),
                      exp_dir+'/final_transport_plot.png')

visualize_single_plotting(utils.to_data(fixed_z), utils.to_data(fixed_r), utils.to_data(images['Y']) if gen else utils.to_data(images['TY']),
                      exp_dir+'/final_transport_plot.pdf')

data_dict = {'Y': utils.to_data(fixed_z), 'X':utils.to_data(fixed_r), 'X_pred':  utils.to_data(images['Y']) if gen else utils.to_data(images['TY'])}
np.save(exp_dir+'/data.npy', data_dict)
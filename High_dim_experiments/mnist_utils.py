import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import torch
import os
import torchvision.utils as vutils
from torch.autograd import Variable

def print_out(losses, curr_iter, total_iter, tbx_writer=None):
    msg = 'Step [%d/%d], ' % (curr_iter, total_iter)
    for k, v in losses.items():
        msg += '%s: %.4f ' % (k, v)
        if tbx_writer is not None:
            tbx_writer.add_scalar(k, v, curr_iter)
    print(msg)

def print_opts(config):
    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

def visualize_iter(images, dir, step, config, data_range=(-1, 1)):
    for k, image in images.items():
        vutils.save_image(image.cpu().data, os.path.join(dir, '%s_%06d.png' % (k, step)), normalize=True, range=data_range, nrow=int(np.sqrt(config.batch_size)))

def visualize_single(image, path, config, data_range=(-1, 1)):
    vutils.save_image(image.cpu().data, path, normalize=True, range=data_range, nrow=int(np.sqrt(config.batch_size)))

def print_networks(networks):
    for name, net in networks.items():
        print('---------- Networks initialized -------------')
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print('[Network %s] Total number of parameters : %d' % (name, num_params))
        print('-----------------------------------------------')

def save_networks(networks, model_dir):
    for k, v in networks.items():
        torch.save(v.state_dict(), os.path.join(model_dir, ('%s.pkl' % k)))

def unsqueeze(tensor, ndim=2):
    for it in range(ndim-1):
        tensor = tensor.unsqueeze(1)
    return tensor

def get_ones(size):
    ones = torch.ones(size)
    if torch.cuda.is_available():
        ones = ones.cuda()
    return ones

def to_var(x, requires_grad=False):
    """Converts numpy to variable."""
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

def to_data(x):
    """Converts variable to numpy."""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()

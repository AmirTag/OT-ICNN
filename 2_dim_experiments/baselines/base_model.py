import torch
import utils
import functools
import losses
import networks
from torch.autograd import Variable
from collections import OrderedDict

class Base(object):
    def __init__(self, config, r_loader, z_loader):
        self.config = config
        self.cost = functools.partial(losses.cost, l=config.l, p=config.p)
        self.z_generator = z_loader
        self.r_generator = r_loader
        self.fixed_z = utils.to_var(next(self.z_generator))
        self.fixed_r = utils.to_var(next(self.r_generator))
        self.define_model(config)

    def get_fixed_data(self):
        return self.fixed_r, self.fixed_z

    def get_data(self, config):
        z = utils.to_var(next(self.z_generator))
        r = utils.to_var(next(self.r_generator))
        return r, z

    def get_cost(self):
        return self.cost

    def define_model(self, config):
        self.define_d(config)
        if config.gen:
            self.define_g(config)

    def define_d(self, config):
        raise NotImplementedError("Please Implement this method")

    def define_g(self, config):
        self.g = networks.get_g(config)
        self.g_optimizer = networks.get_optim(self.g.parameters(), config.g_lr, config)

    def get_tx(self, x, reverse=False):
        x = Variable(x.data, requires_grad=True)
        if reverse:
            ux = self.psi(x)
        else:
            ux = self.phi(x)
        dux = torch.autograd.grad(outputs=ux, inputs=x,
                                  grad_outputs=utils.get_ones(ux.size()),
                                  create_graph=True, retain_graph=True,
                                  only_inputs=True)[0]
        Tx = x - dux
        return Tx

    def train_iter(self, config):
        for it in range(config.d_iters):
            self.train_diter(config)
        if config.gen:
            self.train_giter(config)

    def train_diter(self, config):
        self.d_optimizer.zero_grad()
        x, y = self.get_data(config)
        # this is good for computational reasons:
        x, y = x.detach(), y.detach()
        tx, ty = self.get_tx(x), self.get_tx(y, reverse=True)
        ux, vy = self.phi(x), self.psi(y)
        d_loss = self.calc_dloss(x, y, tx, ty, ux, vy, config)
        d_loss.backward()
        self.d_optimizer.step()
        self.d_loss = d_loss.data.item()

    def train_giter(self, config):
        self.g_optimizer.zero_grad()
        x, y = self.get_data(config)
        tx, ty = self.get_tx(x), self.get_tx(y, reverse=True)
        ux, vy = self.phi(x), self.psi(y)
        g_loss = self.calc_gloss(x, y, ux, vy, config)
        g_loss.backward()
        self.g_optimizer.step()
        self.g_loss = g_loss.data.item()

    def calc_dloss(self, x, y, tx, ty, ux, vy, config):
        raise NotImplementedError("Please Implement this method")

    def calc_gloss(self, x, y, ux, vy, config):
        raise NotImplementedError("Please Implement this method")

    ## model statistics
    def get_stats(self,  config):
        raise NotImplementedError("Please Implement this method")

    def get_visuals(self, config):
        gz = self.g(self.fixed_z) if config.gen else self.fixed_z
        x, y = self.fixed_r, gz
        tx, ty = self.get_tx(x), self.get_tx(y, reverse=True)
        images = OrderedDict([('X', x),
                              ('TX', tx),
                              ('Y', y),
                              ('TY', ty)])
        if config.gen:
            images['ZY'] = self.fixed_z
        return images

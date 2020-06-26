import torch
import utils
import functools
import losses
import networks
import itertools
from collections import OrderedDict
from base_model import Base

class BaryOT(Base):
    """Barycentric-OT from: https://arxiv.org/pdf/1711.02283.pdf"""

    def __init__(self, config, r_loader, z_loader):
        super(BaryOT, self).__init__(config, r_loader, z_loader)
        self.ineq_loss = functools.partial(losses.ineq_loss, reg_type=config.reg_type)
        self.map_loss = functools.partial(losses.map_loss, reg_type=config.reg_type)

    def define_d(self, config):
        self.phi, self.psi = networks.get_d(config), networks.get_d(config)
        self.d_optimizer = networks.get_optim(itertools.chain(list(self.phi.parameters()),
                                                              list(self.psi.parameters())),
                                                              config.d_lr, config)

    def train_diter_only(self, config):
        """dual stage (stage 1)"""
        self.train_diter(config)
        self.g_loss = None

    def train_iter(self, config):
        """map stage (stage 2)"""
        self.train_giter(config)
        self.d_loss = None

    def calc_dloss(self, x, y, tx, ty, ux, vy, config):
        d_loss = -torch.mean(ux + vy)
        d_loss += self.ineq_loss(x, y, ux, vy, self.cost, config.lambda_ineq)
        return d_loss

    def calc_gloss(self, x, y, ux, vy, config):
        fy = self.g(y)
        return self.map_loss(x, y, fy, ux, vy, self.cost, config.lambda_ineq)

    def get_stats(self,  config):
        """print outs"""
        stats = OrderedDict()
        if self.d_loss is not None:
            stats['loss/disc'] = self.d_loss
        if self.g_loss is not None:
            stats['loss/gen'] = self.g_loss
        return stats

    def get_networks(self, config):
        nets = OrderedDict([('phi', self.phi),
                            ('psi', self.psi)])
        nets['gen'] = self.g
        return nets

    def get_visuals(self, config):
        gz = self.g(self.fixed_z)
        x, y = self.fixed_r, gz
        tx, ty = self.get_tx(x), self.get_tx(y, reverse=True)
        images = OrderedDict([('X', x),
                              ('TX', tx),
                              ('Y', y),
                              ('TY', ty)])
        images['ZY'] = self.fixed_z
        return images

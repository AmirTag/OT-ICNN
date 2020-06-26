import torch
import utils
import functools
import losses
import networks
import itertools
from collections import OrderedDict
from base_model import Base

class W2(Base):
    """Wasserstein-2 based models including W2-OT/W2GAN"""

    def get_data(self, config):
        """override z with gz in the case gen=T"""
        z = utils.to_var(next(self.z_generator))
        gz = self.g(z) if config.gen else z
        r = utils.to_var(next(self.r_generator))
        return r, gz

    def define_d(self, config):
        self.phi, self.eps = networks.get_d(config), networks.get_d(config)
        self.d_optimizer = networks.get_optim(itertools.chain(list(self.phi.parameters()),
                                                              list(self.eps.parameters())),
                                                              config.d_lr, config)

    def psi(self, y):
        return -self.phi(y) + self.eps(y)

    def calc_dloss(self, x, y, tx, ty, ux, vy, config):
        d_loss = -torch.mean(ux + vy)
        if config.ineq:
            d_loss += losses.ineq_loss(x, y, ux, vy, self.cost, config.lambda_ineq)
        if config.ineq_interp:
            d_loss += losses.calc_interp_ineq(x, y, self.phi, self.psi, self.cost, config.lambda_ineq, losses.ineq_loss)
        if config.eq_phi:
            d_loss += losses.calc_eq(x, tx, self.phi, self.psi, self.cost, config.lambda_eq)
        if config.eq_psi:
            d_loss += losses.calc_eq(ty, y, self.phi, self.psi, self.cost, config.lambda_eq)
        return d_loss

    def calc_gloss(self, x, y, ux, vy, config):
        return torch.mean(vy)

    def get_stats(self,  config):
        """print outs"""
        stats = OrderedDict()
        stats['loss/disc'] = self.d_loss
        if config.gen:
            stats['loss/gen'] = self.g_loss
        return stats

    def get_networks(self, config):
        nets = OrderedDict([('phi', self.phi),
                            ('eps', self.eps)])
        if config.gen:
            nets['gen'] = self.g
        return nets

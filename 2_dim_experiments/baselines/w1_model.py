import torch
import utils
import losses
import networks
from collections import OrderedDict
from base_model import Base

class W1(Base):
    """Wasserstein-1 based models including WGAN-GP/WGAN-LP"""

    def get_data(self, config):
        """override z with gz in the case gen=T"""
        z = utils.to_var(next(self.z_generator))
        gz = self.g(z) if config.gen else z
        r = utils.to_var(next(self.r_generator))
        return r, gz

    def define_d(self, config):
        self.phi = networks.get_d(config)
        self.d_optimizer = networks.get_optim(self.phi.parameters(), config.d_lr, config)

    def psi(self, y):
        return -self.phi(y)

    def calc_dloss(self, x, y, tx, ty, ux, vy, config):
        d_loss = -torch.mean(ux + vy)
        d_loss += losses.gp_loss(x, y, self.phi, config.lambda_gp, clamp=config.clamp)
        return d_loss

    def calc_gloss(self, x, y, ux, vy, config):
        return torch.mean(vy)

    ## model statistics
    def get_stats(self,  config):
        """print outs"""
        stats = OrderedDict()
        stats['loss/disc'] = self.d_loss
        if config.gen:
            stats['loss/gen'] = self.g_loss
        return stats

    def get_networks(self, config):
        nets = OrderedDict([('phi', self.phi)])
        if config.gen:
            nets['gen'] = self.g
        return nets

import torch
import math
import random
import numpy as np

def get_loader(config):

    # Order is (real_data, z_data)
    """builds and returns generators for 2D source & target datasets."""
    if config.data == '4gaussians':
        return GaussiansGenerator(config.batch_size, scale=1.0, eps_noise=0.1),\
            GaussiansGenerator(config.batch_size, eps_noise=0.1)

    elif config.data == 'swissroll':
        return SwissrollGenerator(config.batch_size, eps_noise=0.1), \
            SwissrollGenerator(config.batch_size, eps_noise=0.1, alternate=True)

    elif config.data == 'checkerboard':
        return CheckerboardGenerator(config.batch_size, eps_noise=0.5, alternate=True), \
            CheckerboardGenerator(config.batch_size, eps_noise=0.5)

    elif config.data == 'our_checkerboard':
        return Our_CheckerboardGenerator(config.batch_size, eps_noise=0.5), \
            Our_CheckerboardGenerator(config.batch_size, eps_noise=0.5, five_squares=True)       

    elif config.data == '8gaussians':
        return Eight_GMM_Generator(config.batch_size),\
            StandardGaussianGenerator(config.batch_size)

    else:
        raise NotImplementedError('requested data: %s is not implemented' % config.data)


class SyntheticDataGenerator(object):
    """superclass of all synthetic data. WARNING: doesn't raise StopIteration so it loops forever!"""

    def __iter__(self):
        return self

    def __next__(self):
        return self.get_batch()

    def get_batch(self):
        raise NotImplementedError()

    def float_tensor(self, batch):
        return torch.from_numpy(batch).type(torch.FloatTensor)


class GaussiansGenerator(SyntheticDataGenerator):
    """samples from four 2D gaussians."""

    def __init__(self,
                 batch_size: int=256,
                 scale: float=2.,
                 center_coor_min: float=-0.25,
                 center_coor_max: float=+0.25,
                 eps_noise: float=0.01):
        self.batch_size = batch_size
        scale = scale
        self.eps_noise = eps_noise
        diag_len = np.sqrt(center_coor_min**2 + center_coor_max**2)
        centers = [
            (center_coor_max / diag_len, center_coor_max / diag_len),
            (center_coor_max / diag_len, center_coor_min / diag_len),
            (center_coor_min / diag_len, center_coor_max / diag_len),
            (center_coor_min / diag_len, center_coor_min / diag_len)
        ]
        self.centers = [(scale * x, scale * y) for x, y in centers]

    def get_batch(self):
        batch = []
        for i in range(self.batch_size):

            point = np.random.randn(2) * self.eps_noise
            center = self.centers[i % 4]
            point[0] += center[0]
            point[1] += center[1]
            batch.append(point)
        batch = np.array(batch, dtype='float32')
        batch = self.float_tensor(batch)
        batch = batch[torch.randperm(batch.size(0)), :]
        return batch


class Eight_GMM_Generator(SyntheticDataGenerator):
    """samples from four 2D gaussians."""

    def __init__(self,
                 batch_size: int=256,
                 scale: float=10.,
                 eps_noise: float=.5):
        self.batch_size = batch_size
        scale = scale
        self.eps_noise = eps_noise

        centers = [
			(1,0),
			(-1,0),
			(0,1),
			(0,-1),
			(1./np.sqrt(2), 1./np.sqrt(2)),
			(1./np.sqrt(2), -1./np.sqrt(2)),
			(-1./np.sqrt(2), 1./np.sqrt(2)),
			(-1./np.sqrt(2), -1./np.sqrt(2))
        ]
        
        self.centers = [(scale*x,scale*y) for x,y in centers]

    def get_batch(self):
        batch = []
        for i in range(self.batch_size):
            
            point = np.random.randn(2) * self.eps_noise
            center = self.centers[i % 8]
            point[0] += center[0]
            point[1] += center[1]
            batch.append(point)

        batch = np.array(batch, dtype='float32')
        batch = self.float_tensor(batch)
        batch = batch[torch.randperm(batch.size(0)), :]
        return batch


class StandardGaussianGenerator(SyntheticDataGenerator):
    """samples from Standard Gaussian."""

    def __init__(self,
                 batch_size: int=256,
                 scale: float=1.,
                 eps_noise: float=1.):
        self.batch_size = batch_size
        scale = scale
        self.eps_noise = eps_noise

    def get_batch(self):
        batch = []
        for _ in range(self.batch_size):
            point = np.random.randn(2) * self.eps_noise
            batch.append(point)
        batch = np.array(batch, dtype='float32')
        batch = self.float_tensor(batch)
        batch = batch[torch.randperm(batch.size(0)), :]
        return batch


class SwissrollGenerator(SyntheticDataGenerator):
    """samples from one of two 2D spirals (depending on alternate=T/F)."""

    def __init__(self,
                 batch_size: int=256,
                 start: float=0.001,
                 eps_noise: float=0.01,
                 alternate: bool=False):
        self.batch_size = batch_size
        self.deg2rad = (2*math.pi)/360
        self.start = start * self.deg2rad
        self.eps_noise = eps_noise/0.15
        self.degrees = 570
        self.alternate = alternate

    def get_batch(self):
        batch = []
        for i in range(self.batch_size):
            n = self.start + np.sqrt(np.random.rand())*self.degrees*self.deg2rad
            if self.alternate:
                batch.append([-math.cos(n)*n + np.random.rand()*self.eps_noise, math.sin(n)*n + np.random.rand()*self.eps_noise])
            else:
                batch.append([math.cos(n)*n + np.random.rand()*self.eps_noise, -math.sin(n)*n + np.random.rand()*self.eps_noise])
        batch = np.array(batch, dtype='float32')
        batch *= 0.15
        return self.float_tensor(batch)


class Our_CheckerboardGenerator(SyntheticDataGenerator):
    """samples from one of two sets 2D squares (depending on alternate=T/F)."""

    def __init__(self,
                 batch_size: int=256,
                 scale: float=5.0,
                 eps_noise: float=0.5,
                 five_squares: bool=False):
        self.batch_size = batch_size
        self.scale = scale
        self.eps_noise = eps_noise
        self.five_squares = five_squares
        if self.five_squares:
            centers = [ (0., 0.), (1., 1.), (-1., 1.),\
                    (-1., -1.), (1., -1.)
                ]
        else:
            centers = [ (1., 0.), (0., 1.), (-1., 0.), (0., -1.)
                ]

        self.centers = [(scale * x, scale * y) for x, y in centers]

    def get_batch(self):
        batch = []
        for i in range(self.batch_size):
            point = 2 * (np.random.rand(2)-0.5) * self.eps_noise
            num = 5 if self.five_squares else 4
            center = self.centers[i % num]
            point[0] += center[0]
            point[1] += center[1]
            batch.append(point)
        batch = np.array(batch, dtype='float32')
        batch = self.float_tensor(batch)
        batch = batch[torch.randperm(batch.size(0)), :]
        return batch


class CheckerboardGenerator(SyntheticDataGenerator):
    """samples from one of two sets 2D squares (depending on alternate=T/F)."""

    def __init__(self,
                 batch_size: int=256,
                 scale: float=1.5,
                 center_coor_min: float=-0.25,
                 center_coor_max: float=+0.25,
                 eps_noise: float=0.01,
                 alternate: bool=False,
                 simple: bool=False):
        self.batch_size = batch_size
        self.scale = scale
        self.eps_noise = eps_noise
        self.alternate = alternate
        self.simple = simple
        diag_len = np.sqrt(center_coor_min**2 + center_coor_max**2)
        center_coor_mid = (center_coor_max + center_coor_min)/2
        if self.simple:
            centers = [(center_coor_mid / diag_len, center_coor_mid / diag_len)]
        else:
            if self.alternate:
                centers = [
                    (center_coor_mid / diag_len, center_coor_max / diag_len),
                    (center_coor_mid / diag_len, center_coor_min / diag_len),
                    (center_coor_max / diag_len, center_coor_mid / diag_len),
                    (center_coor_min / diag_len, center_coor_mid / diag_len),
                    ]
            else:
                centers = [
                    (center_coor_max / diag_len, center_coor_max / diag_len),
                    (center_coor_min / diag_len, center_coor_min / diag_len),
                    (center_coor_max / diag_len, center_coor_min / diag_len),
                    (center_coor_min / diag_len, center_coor_max / diag_len),
                    (center_coor_mid / diag_len, center_coor_mid / diag_len)
                ]
        self.centers = [(scale * x, scale * y) for x, y in centers]

    def get_batch(self):
        batch = []
        for i in range(self.batch_size):
            point = (np.random.rand(2)-0.5) * self.eps_noise
            num = 4 if self.alternate else 5
            center = self.centers[i % num]
            point[0] += center[0]
            point[1] += center[1]
            batch.append(point)
        batch = np.array(batch, dtype='float32')
        batch = self.float_tensor(batch)
        batch = batch[torch.randperm(batch.size(0)), :]
        return batch

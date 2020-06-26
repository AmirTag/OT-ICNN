import torch
import numpy as np
import torch.utils.data as data

from torchvision import datasets
from torchvision import transforms
from torch.distributions.multivariate_normal import MultivariateNormal

def get_loader(config):
    tf = transforms.Compose([transforms.Resize(28),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    mnist = datasets.MNIST(root=config.mnist_path, train=True, download=True, transform=tf)
    mnist_loader = data.DataLoader(dataset=mnist, batch_size=config.batch_size,
                                   shuffle=True,
                                   num_workers=4)
    r_loader = RealDataGenerator(mnist_loader)
    mu, cov = compute_mnist_stats(mnist)
    z_loader = MVGaussianGenerator(config.batch_size, mu, cov)
    return r_loader, z_loader

class DataGenerator(object):
    "superclass of all data. WARNING: doesn't raise StopIteration so it loops forever!"

    def __iter__(self):
        return self

    def __next__(self):
        return self.get_batch()

    def get_batch(self):
        raise NotImplementedError()

    def float_tensor(self, batch):
        return torch.from_numpy(batch).type(torch.FloatTensor)

class StandardGaussianGenerator(DataGenerator):
    """samples from a multivariate gaussian"""
    def __init__(self, batch_size, mu, cov, lambda_identity=1.0):
        self.batch_size = batch_size
        cov = cov + lambda_identity * torch.eye(cov.size(0)) * 1e-1
        self.generator = MultivariateNormal(mu, cov)

    def get_batch(self):
        return self.generator.sample((self.batch_size,)).view(self.batch_size, -1)

class RealDataGeneratorDummy(DataGenerator):
    """samples from real data"""
    def __init__(self, loader):
        self.loader = loader
        self.generator = iter(self.loader)
        self.data_len = len(self.loader)
        self.count = 0

    def get_batch(self):
        if (((self.count + 1) % self.data_len) == 0):
            del self.generator
            self.generator = iter(self.loader)
        self.count += 1
        return next(self.generator)


class MVGaussianGenerator(DataGenerator):
    """samples from a multivariate gaussian"""
    def __init__(self, batch_size, mu, cov, lambda_identity=1.0):
        self.batch_size = batch_size
        self.image_size = 28
        cov = cov + lambda_identity * torch.eye(cov.size(0)) * 1e-1
        self.generator = MultivariateNormal(mu, cov)

    def get_batch(self):
        return self.generator.sample((self.batch_size,)).view(self.batch_size, 1, self.image_size, self.image_size)

class RealDataGenerator(DataGenerator):
    """samples from real data"""
    def __init__(self, loader):
        self.loader = loader
        self.generator = iter(self.loader)
        self.data_len = len(self.loader)
        self.count = 0

    def get_batch(self):
        if (((self.count + 1) % self.data_len) == 0):
            del self.generator
            self.generator = iter(self.loader)
        self.count += 1
        return next(self.generator)[0]

def compute_mnist_stats(mnist_dataset):
    loader = data.DataLoader(dataset=mnist_dataset, batch_size=60000, num_workers=8)
    mnist = next(iter(loader))[0]
    mnist = mnist.view(60000, -1).t().numpy()
    mnist_mean = np.mean(mnist, axis=1)
    mnist_cov = np.cov(mnist)
    return torch.from_numpy(mnist_mean).type(torch.FloatTensor), \
        torch.from_numpy(mnist_cov).type(torch.FloatTensor)

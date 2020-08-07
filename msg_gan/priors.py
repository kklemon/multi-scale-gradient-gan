import torch


class Prior:
    def sample(self, batch_size, latent_sze):
        raise NotImplementedError


class GaussianPrior(Prior):
    def sample(self, batch_size, latent_size):
        return torch.randn(batch_size, latent_size)


class MultinomialBernoulliPrior(Prior):
    def __init__(self, normalize=True):
        self.normalize = normalize

    def sample(self, batch_size, latent_size):
        z = (torch.rand(batch_size, latent_size) > 0.5).type(torch.float)
        if self.normalize:
            z = (z * 2) - 1
        return z

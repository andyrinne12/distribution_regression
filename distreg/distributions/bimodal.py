import numpy as np
import torch

from distreg.distributions.distribution import DIST_CLASSES, UnivarDistribution


class BimodalNormal(UnivarDistribution):
    def __init__(
        self,
        n_distributions: int,
        n_dims: int,
        mean1: int,
        std1: int,
        mean2: int,
        std2: int,
    ):
        super().__init__(
            n_distributions,
            n_dims,
            torch.Tensor([]),
            torch.Tensor([]),
            torch.Tensor([]),
        )
        self.mean1 = mean1
        self.std1 = std1
        self.mean2 = mean2
        self.std2 = std2

    def sample(self, n_samples: int) -> None:
        n_samples_each = n_samples // 2
        self.samples = torch.empty((1, n_samples, self.n_dims))
        samples1 = np.random.normal(
            self.mean1, self.std1, size=(n_samples_each, self.n_dims)
        )
        samples2 = np.random.normal(
            self.mean2,
            self.std2,
            size=(n_samples - n_samples_each, self.n_dims),
        )
        self.samples[0] = torch.from_numpy(
            np.concatenate([samples1, samples2])
        )
        return self.samples

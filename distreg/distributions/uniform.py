import numpy as np
import torch

from distreg.distributions.distribution import DIST_CLASSES, UnivarDistribution

DIST_CLASS_LABEL = "uniform"


class Uniform(UnivarDistribution):
    def __init__(
        self,
        n_distributions: int,
        n_dims: int,
        means: torch.Tensor,
        stds: torch.Tensor,
        classes: torch.Tensor,
    ):
        super().__init__(n_distributions, n_dims, means, stds, classes)

    @classmethod
    def rand_unf(
        cls,
        n_distributions: int,
        n_dims: int,
        mean: int | tuple[int, int],
        std: int | tuple[int, int],
    ):
        means, stds = super().rand_unf_univar(
            n_distributions, n_dims, mean, std
        )
        classes = torch.from_numpy(
            np.full((n_distributions), DIST_CLASSES[DIST_CLASS_LABEL])
        )
        return cls(n_distributions, n_dims, means, stds, classes)

    def sample(self, n_samples: int) -> None:
        self.samples = torch.empty(
            (self.n_distributions, n_samples, self.n_dims)
        )
        for i, [mean, std] in enumerate(zip(self.means, self.stds)):
            std_uniform = torch.from_numpy(
                np.random.uniform(0, 1, size=(n_samples, self.n_dims))
            )
            self.samples[i] = mean + 2 * np.sqrt(3) * std * (
                std_uniform - 1 / 2
            )
        return self.samples

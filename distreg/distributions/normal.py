import numpy as np
import torch

from distreg.distributions.distribution import DIST_CLASSES, UnivarDistribution


class Normal(UnivarDistribution):
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
        subclass="normal",
    ):
        means, stds = super().rand_unf_univar(
            n_distributions, n_dims, mean, std
        )
        classes = torch.from_numpy(
            np.full(n_distributions, DIST_CLASSES[subclass])
        )
        return cls(n_distributions, n_dims, means, stds, classes)

    def sample(self, n_samples: int) -> None:
        self.samples = torch.empty(
            (self.n_distributions, n_samples, self.n_dims)
        )
        for i, [mean, std, _class] in enumerate(
            zip(self.means, self.stds, self.classes)
        ):
            if _class == DIST_CLASSES["normal"]:
                self.samples[i] = torch.from_numpy(
                    np.random.normal(mean, std, size=(n_samples, self.n_dims))
                )
            elif _class == DIST_CLASSES["laplacian"]:
                self.samples[i] = torch.from_numpy(
                    np.random.laplace(mean, std, size=(n_samples, self.n_dims))
                )
            elif _class == DIST_CLASSES["gamma"]:
                theta = std**2 / mean
                k = mean / theta
                self.samples[i] = torch.from_numpy(
                    np.random.gamma(
                        shape=k, scale=theta, size=(n_samples, self.n_dims)
                    )
                )
        return self.samples

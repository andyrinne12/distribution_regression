import numpy as np
import torch

DIST_CLASSES = {
    "uniform": 0,
    "normal": 1,
    "laplacian": 2,
    "cauchy": 3,
    "gamma": 4,
}

DIST_CLASS_LABELS = {DIST_CLASSES[k]: k for k in DIST_CLASSES}


class UnivarDistribution:
    def __init__(
        self,
        n_distributions: int,
        n_dims: int,
        means: np.ndarray,
        stds: np.ndarray,
        classes: np.ndarray,
        samples=None,
    ):
        self.n_distributions = n_distributions
        self.n_dims = n_dims
        self.means = means
        self.stds = stds
        self.classes = classes
        self.samples = samples

    @classmethod
    def rand_unf_univar(
        cls,
        n_distributions: int,
        n_dims: int,
        mean: int | tuple[int, int],
        std: int | tuple[int, int],
    ):
        """Generate N distributions with fixed or uniformly sampled
        mean and variances.

        Args:
            n_distributions (int): N number of distributions
            n_dims (int): dimensionality of the distribution samples
            dist_class (int): distribution class
            mean (int | tuple): mean or uniform range of means
            std (int | tuple): standard deviation or uniform range of
                stds
        """
        means = torch.empty((n_distributions, n_dims))
        stds = torch.empty((n_distributions, n_dims))

        if type(mean) is tuple:
            means[:, :] = torch.from_numpy(
                np.random.uniform(mean[0], mean[1], size=(n_distributions, 1))
            )
        else:
            means[:, :] = mean

        if type(std) is tuple:
            stds[:, :] = torch.from_numpy(
                np.random.uniform(std[0], std[1], size=(n_distributions, 1))
            )
        else:
            stds[:, :] = std
        return (means, stds)

    def bin_dist_class_labels(self, _class: float):
        return (self.classes == _class).type(torch.float)

    def bin_sigma_gt_labels(self, std_threshold: float):
        return (self.stds > std_threshold).type(torch.float)


def concatenate_distributions(distributions):
    n_distributions = sum(
        map(lambda dist: dist.n_distributions, distributions)
    )
    n_dims = distributions[0].n_dims
    for i, d in enumerate(distributions):
        if d.n_dims != n_dims:
            raise ValueError(
                f"Distribution item {i} has a different dimension {d.n_dims}"
                " than the others {n_dims}"
            )
    means = torch.concatenate(list(map(lambda d: d.means, distributions)))
    stds = torch.concatenate(list(map(lambda d: d.stds, distributions)))
    classes = torch.concatenate(list(map(lambda d: d.classes, distributions)))
    samples = torch.concatenate(list(map(lambda d: d.samples, distributions)))
    assert means.shape[0] == n_distributions
    assert stds.shape[0] == n_distributions
    assert classes.shape[0] == n_distributions
    if samples is not None:
        assert samples.shape[0] == n_distributions
    return UnivarDistribution(
        n_distributions, n_dims, means, stds, classes, samples
    )

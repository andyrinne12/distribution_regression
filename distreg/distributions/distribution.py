import numpy as np

DIST_CLASSES = {
    "uniform": 0,
    "normal": 1,
    "laplacian": 2,
    "cauchy": 3,
    "gamma": 4,
}

DIST_CLASS_LABELS = {DIST_CLASSES[k]: k for k in DIST_CLASSES}


class Distribution:
    def __init__(
        self,
        n_distributions: int,
        n_dims: int,
        params: np.ndarray,
        samples: np.ndarray,
    ):
        self.n_distributions = n_distributions
        self.n_dims = n_dims
        self.params = params
        self.samples = samples

    # def sample(self, n_samples: int) -> None:
    #     self.samples = np.empty((self.n_distributions, n_samples, self.n_dims))
    #     for i, [mu, sigma] in enumerate(self.params[:, :2]):
    #         self.samples[i] = np.random.normal(
    #             mu, sigma, size=(n_samples, self.n_dims)
    #         )
    #     return self.samples

    def bin_dist_class_labels(self, _class: float):
        return self.params[:, 2] == _class

    def bin_sigma_gt_labels(self, sigma_threshold: float):
        return (self.params[:, 1] > sigma_threshold).astype(float)


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
    params = np.concatenate(list(map(lambda d: d.params, distributions)))
    samples = np.concatenate(list(map(lambda d: d.samples, distributions)))
    assert params.shape[0] == n_distributions
    if samples is not None:
        assert samples.shape[0] == n_distributions
    return Distribution(n_distributions, n_dims, params, samples)

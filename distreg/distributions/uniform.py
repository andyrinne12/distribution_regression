from distutils.dist import Distribution
import numpy as np

from distreg.distributions.distribution import DIST_CLASSES

DIST_CLASS_KEY = "uniform"


class Uniform(Distribution):
    def __init__(
        self,
        n_distributions: int,
        n_dims: int,
        mu: int | tuple[int, int],
        sigma: int | tuple[int, int],
    ):
        """Generate N uniform distributions with fixed or uniformly sampled
        mean and variances.

        Args:
            n_distributions (int): N number of distributions
            n_dims (int): dimensionality of the distribution samples
            mu (int | tuple): mean or uniform range of means
            sigma (int | tuple): variance or uniform range of variances
        """
        self.n_distributions = n_distributions
        self.n_dims = n_dims
        self.params = np.empty((n_distributions, 3))
        if type(mu) is tuple:
            self.params[:, 0] = np.random.uniform(
                mu[0], mu[1], n_distributions
            )
        else:
            self.params[:, 0] = mu

        if type(sigma) is tuple:
            self.params[:, 1] = np.random.uniform(
                sigma[0], sigma[1], n_distributions
            )
        else:
            self.params[:, 1] = sigma
        self.params[:, 2] = DIST_CLASSES[DIST_CLASS_KEY]

    def sample(self, n_samples: int) -> None:
        self.samples = np.empty((self.n_distributions, n_samples, self.n_dims))
        for i, [mu, sigma] in enumerate(self.params[:, :2]):
            std_uniform = np.random.uniform(
                0, 1, size=(n_samples, self.n_dims)
            )
            self.samples[i] = mu + 2 * np.sqrt(3) * sigma * (
                std_uniform - 1 / 2
            )
        return self.samples

    def bin_sigma_gt_labels(self, sigma_threshold: float):
        return (self.params[:, 1] > sigma_threshold).astype(float)

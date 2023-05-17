import numpy as np

from distreg.distributions.distribution import DIST_CLASSES


class Normal:
    def __init__(
        self,
        n_distributions: int,
        n_dims: int,
        mu: int | tuple[int, int],
        sigma: int | tuple[int, int],
        subclass="normal",
    ):
        """Generate N normal distributions with fixed or uniformly sampled
        mean and variances.

        Args:
            n_distributions (int): N number of distributions
            n_dims (int): dimensionality of the distribution samples
            mu (int | tuple): mean or uniform range of means
            sigma (int | tuple): variance or uniform range of variances
        """
        self.n_distributions = n_distributions
        self.n_dims = n_dims
        self.subclass = subclass
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
        self.params[:, 2] = DIST_CLASSES[self.subclass]

    def sample(self, n_samples: int) -> None:
        self.samples = np.empty((self.n_distributions, n_samples, self.n_dims))
        for i, [mu, sigma] in enumerate(self.params[:, :2]):
            if self.subclass == "normal":
                self.samples[i] = np.random.normal(
                    mu, sigma, size=(n_samples, self.n_dims)
                )
            elif self.subclass == "laplacian":
                self.samples[i] = np.random.laplace(
                    mu, sigma, size=(n_samples, self.n_dims)
                )
            elif self.subclass == "gamma":
                theta = sigma**2 / mu
                k = mu / theta
                self.samples[i] = np.random.gamma(
                    shape=k, scale=theta, size=(n_samples, self.n_dims)
                )
        return self.samples

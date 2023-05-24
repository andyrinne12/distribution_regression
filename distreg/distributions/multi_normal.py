import numpy as np

from distreg.distributions.distribution import DIST_CLASSES


class MultiNormal:
    def __init__(
        self,
        n_distributions: int,
        n_dims: int,
        mean: int | tuple[int, int],
        var: int | tuple[int, int],
        cov: int | tuple[int, int],
        same_var=True,
        subclass="normal",
    ):
        """Generate N normal distributions with fixed or uniformly sampled
        mean and variances.

        Args:
            n_distributions (int): N number of distributions
            n_dims (int): dimensionality of the distribution samples
            mean (int | tuple): mean or uniform range of means
            var (int | tuple): variance or uniform range of variances
            cov (int | tuple): covariance or uniform range of covariances
        """
        self.n_distributions = n_distributions
        self.n_dims = n_dims
        self.subclass = subclass
        self.means = np.empty((n_distributions, n_dims))
        self.covs = np.empty((n_distributions, n_dims, n_dims))
        self.classes = np.empty((n_distributions))

        if type(mean) is tuple:
            self.means[:, :] = np.random.uniform(
                mean[0], mean[1], size=(n_distributions, 1)
            )
        else:
            self.means[:] = mean

        if type(cov) is tuple:
            for i in range(n_distributions):
                covs_mat = np.random.uniform(
                    cov[0], cov[1], size=(n_dims, n_dims)
                )
                self.covs[i] = np.tril(covs_mat) + np.tril(covs_mat, -1).T
        else:
            self.covs[:] = cov

        if type(var) is tuple:
            for i in range(n_distributions):
                if same_var:
                    np.fill_diagonal(
                        self.covs[i], np.random.uniform(var[0], var[1])
                    )
                else:
                    np.fill_diagonal(
                        self.covs[i], np.random.uniform(var[0], var[1], n_dims)
                    )
        else:
            for i in range(n_distributions):
                np.fill_diagonal(self.covs[i], var)

        self.classes[:] = DIST_CLASSES[self.subclass]

    def sample(self, n_samples: int) -> None:
        self.samples = np.empty((self.n_distributions, n_samples, self.n_dims))
        for i in range(self.n_distributions):
            if self.subclass == "normal":
                self.samples[i] = np.random.multivariate_normal(
                    self.means[i], self.covs[i], size=(n_samples)
                )
            # elif self.subclass == "laplacian":
            #     self.samples[i] = np.random.laplace(
            #         mu, sigma, size=(n_samples, self.n_dims)
            #     )
            # elif self.subclass == "gamma":
            #     theta = sigma**2 / mu
            #     k = mu / theta
            #     self.samples[i] = np.random.gamma(
            #         shape=k, scale=theta, size=(n_samples, self.n_dims)
            #     )
        return self.samples

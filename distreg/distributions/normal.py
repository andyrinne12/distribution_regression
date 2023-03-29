import numpy as np
import torch


class Normal:
    def __init__(
        self,
        n_distributions: int,
        mu: int | tuple[int, int],
        sigma: int | tuple[int, int],
    ):
        """Generate N normal distributions with fixed or uniformly sampled
        mean and variances.

        Args:
            n_distributions (int): N number of distributions
            mu (int | tuple): mean or uniform range of means
            sigma (int | tuple): variance or uniform range of variances
        """
        self.n_distributions = n_distributions
        self.params = torch.empty((n_distributions, 2))
        if type(mu) is tuple:
            self.params[:, 0] = torch.from_numpy(
                np.random.uniform(mu[0], mu[1], n_distributions)
            )
        else:
            self.params[:, 0] = mu

        if type(sigma) is tuple:
            self.params[:, 1] = torch.from_numpy(
                np.random.uniform(sigma[0], sigma[1], n_distributions)
            )
        else:
            self.params[:, 1] = sigma

    def sample(self, n_samples: int, n_dims: int) -> None:
        self.samples = torch.empty((self.n_distributions, n_samples, n_dims))
        for i, [mu, sigma] in enumerate(self.params):
            self.samples[i] = torch.from_numpy(
                np.random.normal(mu, sigma, size=(n_samples, n_dims))
            )
        return self.samples

    def binary_sigma_gt_labels(self, sigma_threshold: float) -> torch.Tensor:
        return (self.params[:, 1] > sigma_threshold).type(torch.float32)


# def gen_normals_grid(mu_range, mu_num, sigma_range, sigma_num):
#     mu_freq = (mu_range[1] - mu_range[0]) / mu_num
#     sigma_freq = (sigma_range[1] - sigma_range[0]) / sigma_num
#     grid_xx, grid_yy = np.mgrid[
#         mu_range[0]: mu_range[1]: mu_freq,
#         sigma_range[0]: sigma_range[1]: sigma_freq,
#     ]
#     xx_flat = grid_xx.reshape(-1)
#     yy_flat = grid_yy.reshape(-1)
#     return torch.from_numpy(np.array(list(zip(xx_flat, yy_flat))))

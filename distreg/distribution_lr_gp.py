import numpy as np
import torch

from distreg.distributions.dataset import DistributionDataset
from distreg.kernels.rff import RffEncoder


class DistributionLogisticGP:
    def __init__(
        self, distributions, labels, kme_encoder, reg_encoder, train_ratio=0.7
    ):
        assert distributions.samples is not None

        self.distributions = distributions
        self.samples = distributions.samples
        self.labels = labels

        self.kme_encoder = kme_encoder
        self.reg_encoder = reg_encoder
        self.mean_embeddings = self.kme_encoder.gen_mean_embeddings(
            distributions
        )
        self.features = self.reg_encoder.encode_features(self.mean_embeddings)
        self.train_idx, self.test_idx = self._train_test_split(train_ratio)

        self.gp_mean = None
        self.gp_cov = None

    def set_kme_kernel_params(
        self, log_amplitude, log_length_scale, kernel, n_features=None
    ):
        if n_features is None:
            n_features = self.n_features
        self.kme_encoder = RffEncoder(
            n_features,
            self.encoder.n_dims,
            kernel,
            log_amplitude,
            log_length_scale,
        )
        self.mean_embeddings = self.encoder.gen_mean_embeddings(
            self.distributions
        )

    def as_dataset(self):
        return DistributionDataset(
            self.features, self.labels, self.distributions.params
        )

    def as_split_dataset(self, train_ratio):
        dataset = self.as_dataset()
        return torch.utils.data.random_split(
            dataset, [train_ratio, 1 - train_ratio]
        )

    @property
    def train_dataset(self):
        return (
            self.distributions.features[self.train_idx],
            self.distributions.labels[self.train_idx],
            self.distributions.params[self.train_idx],
        )

    @property
    def test_dataset(self):
        return (
            self.distributions.features[self.test_idx],
            self.distributions.labels[self.test_idx],
            self.distributions.params[self.test_idx],
        )

    def _train_test_split(self, ratio):
        indices = np.arange(self.distributions.n_distributions)
        np.random.seed(2023)
        np.random.shuffle(indices)
        train_size = int(self.distributions.n_distributions * ratio)
        return indices[:train_size], indices[train_size:]

    #
    #
    #
    #
    # def fit(self, f_tolerance=1e-8):
    #     N = self.distributions.n_distributions
    #     train_data, y, _ = self.train_dataset
    #     self.kernel = K = train_data @ train_data.T
    #     f = np.zeros(N)
    #     delta_f = float("inf")
    #     while delta_f > f_tolerance:
    #         y_hat = sigmoid(f)
    #         grad = -(y_hat - y)
    #         W = np.diag(y_hat(1 - y_hat))
    #         W_sqrt = np.diag(np.sqrt(y_hat(1 - y_hat)))
    #         self.B = np.identity(N) + W_sqrt @ K @ W_sqrt

    #         b = W @ f + grad

    #         L = scipy.linalg.cholesky(self.B, lower=True)
    #         s1 = scipy.linalg.solve_triangular(L, W_sqrt @ K @ b, lower=True)
    #         s2 = scipy.linalg.solve_triangular(L.T, s1, lower=False)

    #         a = b - W_sqrt @ s2
    #         f_new = K @ a
    #         delta_f = np.abs(f_new - f)
    #         if delta_f > f_tolerance:
    #             f = f_new

    #     self.lml = -1 / 2 * a.T @ f + grad - np.sum(np.log(np.diagonal(L)))
    #     self.gp_mean = f
    #     return self.gp_mean, self.lml

    # def predict(self, data):
    #     pass

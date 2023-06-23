import numpy as np
import torch

from distreg.distributions.dataset import DistributionDataset
from distreg.kernels.rff import RffEncoder


class SynthDistributionSetEncoder:
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
        if self.reg_encoder is not None:
            self.features = self.reg_encoder.encode_features(
                self.mean_embeddings, single=True
            )
        else:
            self.features = self.mean_embeddings
        self.train_idx, self.test_idx = self._train_test_split(train_ratio)

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

    def samples_to_features(self, samples):
        mean_embeddings = self.kme_encoder.gen_mean_embeddings_from_tensor(
            samples
        )
        features = self.reg_encoder.encode_features(mean_embeddings)
        return features

    def as_dataset(self):
        return DistributionDataset(
            self.features,
            self.labels,
            self.distributions.means,
            self.distributions.stds,
            self.distributions.classes,
        )

    def as_multi_dataset(self):
        return DistributionDataset(
            self.features,
            self.labels,
            self.distributions.means,
            self.distributions.covs,
            self.distributions.classes,
        )

    def as_split_dataset(self, train_ratio):
        dataset = self.as_dataset()
        return torch.utils.data.random_split(
            dataset, [train_ratio, 1 - train_ratio]
        )

    def as_split_multi_dataset(self, train_ratio):
        dataset = self.as_multi_dataset()
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

import torch
from distreg.kernels import RffEncoder
from distreg.distributions import UnivarDistribution


class GanDistributionSet:
    def __init__(
        self,
        kme_encoder: RffEncoder,
        reg_encoder: RffEncoder,
        ground_samples: torch.Tensor,
        noise_dists: UnivarDistribution,
        fixed_noise_dists: UnivarDistribution,
        samples_per_dist: int,
    ):
        self.kme_encoder = kme_encoder
        self.reg_encoder = reg_encoder
        self.ground_samples = ground_samples
        self.fixed_noise_dists = fixed_noise_dists
        self.noise_dists = noise_dists
        self.samples_per_dist = samples_per_dist

        self.dist_features = torch.empty((1, reg_encoder.n_features))
        self.labels = torch.empty(1)

        mean_embedding = self.kme_encoder.gen_mean_embeddings_from_tensor(
            ground_samples
        )
        features = self.reg_encoder.encode_features(
            mean_embedding, single=True
        )
        self.dist_features[0] = features
        self.labels[0] = 1.0

    def encode_and_append(self, samples, label):
        features = self.encode(samples)
        self.dist_features = torch.cat((self.dist_features, features), 0)
        self.labels = torch.cat(
            (self.labels, torch.Tensor([label] * features.shape[0])), 0
        )

    def encode(self, samples):
        mean_embedding = self.kme_encoder.gen_mean_embeddings_from_tensor(
            samples
        )
        features = self.reg_encoder.encode_features(
            mean_embedding, single=True
        )
        return features

    def resample_noise(self):
        self.noise_dists.sample(self.samples_per_dist)

    @property
    def loss_weights(self):
        n_fake_samples = self.labels.shape[0] - 1
        weights = (self.labels) / 2 + (1 - self.labels) / (2 * n_fake_samples)
        return weights

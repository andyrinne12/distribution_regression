from ctypes import ArgumentError

import numpy as np


class RffEncoder:
    def __init__(
        self,
        n_dims,
        n_features,
        kernel="gaussian",
        log_amplitude=1.0,
        log_length_scale=1.0,
    ):
        self._n_features = n_features
        self.n_dims = n_dims
        self.kernel = kernel
        self._log_amplitude = log_amplitude
        self._log_length_scale = log_length_scale
        self.omega = None
        self._gen_features()

    def encode_features(self, X):
        if self.omega is None:
            raise ArgumentError("Weights not initialized")
        assert self._n_features == self.omega.shape[0]

        features = np.cos(np.einsum("sd, fd -> sf", X, self.omega) + self.phi)
        features = self.amplitude * (2 / self.omega.shape[0]) ** 0.5 * features
        return features

    def gen_mean_embeddings(self, distributions):
        n_distributions = distributions.n_distributions

        mean_embeddings = np.empty((n_distributions, self._n_features))
        for idx, sample_arr in enumerate(distributions.samples):
            features = self.encode_features(sample_arr)
            mean_embeddings[idx] = np.mean(features, axis=0)

        # mean_embeddings = (
        #     mean_embeddings - np.mean(mean_embeddings, axis=0)
        # ) / np.std(mean_embeddings, axis=0)
        return mean_embeddings

    @property
    def log_amplitude(self):
        return self._log_amplitude

    @property
    def amplitude_squared(self):
        return np.exp(self._log_amplitude * 2)

    @property
    def log_length_scale(self):
        return self._log_length_scale

    @property
    def length_scale(self):
        return np.exp(self._log_length_scale)

    @property
    def amplitude(self):
        return np.exp(self._log_amplitude)

    @property
    def n_features(self):
        return self._n_features

    def _gen_features(self):
        omega_shape = (self.n_features, self.n_dims)
        if self.kernel == "gaussian":
            self.omega = np.random.normal(size=omega_shape)
        elif self.kernel == "laplacian":
            self.omega = np.random.laplace(size=omega_shape)
        elif self.kernel == "cauchy":
            self.omega = np.random.standard_cauchy(size=omega_shape)
        else:
            raise ArgumentError(f"Invalid kernel: {self.kernel}")

        self.omega /= self.length_scale
        self.phi = np.random.uniform(0, 2 * np.pi, (self.n_features))
from ctypes import ArgumentError

import numpy as np
import torch


class RffEncoder:
    def __init__(
        self, n_features, n_dim, kernel="gaussian", coeff=1.0, lengthscale=1.0
    ):
        self.n_features = n_features
        self.n_dim = n_dim
        self.kernel = kernel
        self.coeff = coeff
        self.lengthscale = lengthscale
        self.omega = None

    def gen_features(self):
        omega_shape = (self.n_features, self.n_dim)
        if self.kernel == "gaussian":
            self.omega = torch.from_numpy(np.random.normal(size=omega_shape))
        elif self.kernel == "laplacian":
            self.omega = torch.from_numpy(np.random.laplace(size=omega_shape))
        elif self.kernel == "cauchy":
            self.omega = torch.from_numpy(
                np.random.standard_cauchy(size=omega_shape))
        else:
            raise ArgumentError(f"Invalid kernel: {self.kernel}")

        self.omega /= self.lengthscale
        self.phi = torch.from_numpy(
            np.random.uniform(0, 2 * np.pi, (self.n_features)))

    def encode_features(self, X):
        if self.omega is None:
            raise ArgumentError('Weights not initialized')

        features = torch.cos(torch.einsum(
            "sd, fd -> sf", X, self.omega) + self.phi)
        features = self.coeff * (2 / self.omega.shape[0]) ** 0.5 * features
        return features

    def get_n_features(self):
        return self.omega.shape[0]

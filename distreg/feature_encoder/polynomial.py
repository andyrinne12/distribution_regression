import numpy as np
import torch


class PolynomialEncoder:
    def __init__(self, degree=2):
        self.degree = degree

    def gen_features(self):
        pass

    def encode_features(self, X):
        features = np.empty(
            (X.shape[0], self.degree + 1))
        for d in range(self.degree + 1):
            features[:, d] = np.power(X, d).reshape(-1)
        return torch.from_numpy(features)

    def get_n_features(self):
        return self.degree + 1

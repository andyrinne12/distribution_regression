import numpy as np


class PolynomialEncoder:
    def __init__(self, degree=2):
        self.degree = degree

    def gen_features(self):
        pass

    def encode_features(self, X):
        features = np.empty((X.shape[0], self.degree))
        for d in range(1, self.degree + 1):
            features[:, d - 1] = np.power(X, d).reshape(-1)
        return features

    def gen_mean_embeddings(self, distributions):
        n_distributions = distributions.n_distributions

        mean_embeddings = np.empty((n_distributions, 4))
        for idx, sample_arr in enumerate(distributions.samples):
            features = self.encode_features(sample_arr)
            mean = np.mean(features, axis=0, dtype=np.float64)
            mean_squared = mean**2
            mean_embeddings[idx] = np.concatenate([mean, mean_squared])

        mean_embeddings = (
            mean_embeddings - np.mean(mean_embeddings, axis=0)
        ) / np.std(mean_embeddings, axis=0)
        return mean_embeddings

    @property
    def n_features(self):
        return self.degree

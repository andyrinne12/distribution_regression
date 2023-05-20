import torch


class DistributionDataset(torch.utils.data.Dataset):
    def __init__(self, mean_embeddings, labels, means, covs, classes):
        self.mean_embeddings = torch.tensor(
            mean_embeddings, dtype=torch.float64
        )
        self.labels = torch.tensor(labels, dtype=torch.float64)
        self.means = torch.tensor(means, dtype=torch.float64)
        self.covs = torch.tensor(covs, dtype=torch.float64)
        self.classes = torch.tensor(classes, dtype=torch.float64)

    def __len__(self):
        return len(self.mean_embeddings)

    def __getitem__(self, idx):
        return (
            self.mean_embeddings[idx],
            self.labels[idx],
            (self.means[idx], self.covs[idx], self.classes[idx]),
        )

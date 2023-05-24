import torch


class DistributionDataset(torch.utils.data.Dataset):
    def __init__(self, mean_embeddings, labels, means, stds, classes):
        self.mean_embeddings = mean_embeddings
        self.labels = labels
        self.means = means
        self.stds = stds
        self.classes = classes

    def __len__(self):
        return len(self.mean_embeddings)

    def __getitem__(self, idx):
        return (
            self.mean_embeddings[idx],
            self.labels[idx],
            (self.means[idx], self.stds[idx], self.classes[idx]),
        )

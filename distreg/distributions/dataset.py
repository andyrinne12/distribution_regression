import torch


class DistributionDataset(torch.utils.data.Dataset):
    def __init__(self, mean_embeddings, labels, dist_data):
        self.mean_embeddings = torch.tensor(
            mean_embeddings, dtype=torch.float64
        )
        self.labels = torch.tensor(labels, dtype=torch.float64)
        self.dist_data = torch.tensor(dist_data, dtype=torch.float64)

    def __len__(self):
        return len(self.mean_embeddings)

    def __getitem__(self, idx):
        return self.mean_embeddings[idx], self.labels[idx], self.dist_data[idx]

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy

from distreg.distributions.distribution import DIST_CLASS_LABELS


def sigmoid(X):
    return np.clip(1 / (1 + np.exp(X)), 1e-6, 1 - 1e-6)


def plot_normals_predictions(
    model, train_dataset, test_dataset, xlabel, ylabel
):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for dataset, set_label in [
        (train_dataset, "Train"),
        (test_dataset, "Test"),
    ]:
        data, labels, (means, covs, _) = next(
            iter(DataLoader(train_dataset, batch_size=len(dataset)))
        )
        with torch.no_grad():
            y = model(data).flatten()
            plot = sns.scatterplot(
                x=means[:, 0],
                y=covs[:, 0, 1],
                hue=y[:],
                palette="RdPu",
            )
            plot.set_title(f"{set_label} predictions (unf)")
            plt.show()

            plot = sns.scatterplot(
                x=means[:, 0],
                y=covs[:, 0, 1],
                hue=torch.round(y[:]),
            )
            plot.set_title(f"{set_label} predictions (bin)")
            plt.show()

        acc = round(100 * (BinaryAccuracy()(y, labels)).item(), 3)
        print(f"{set_label} Accuracy: {acc}")


def plot_bin_class_predictions(
    model,
    train_dataset,
    test_dataset,
):
    for dataset, set_label in [
        (train_dataset, "Train"),
        (test_dataset, "Test"),
    ]:
        data, labels, dists = next(
            iter(DataLoader(train_dataset, batch_size=len(dataset)))
        )

        with torch.no_grad():
            y = model(data).flatten()
            unique_dist_classes = torch.unique(dists[:, 2]).tolist()
            n_dist_classes = len(unique_dist_classes)
            fig, axs = plt.subplots(
                1, n_dist_classes, figsize=(5 * n_dist_classes, 5)
            )
            for i, dist_class in enumerate(unique_dist_classes):
                idx = dists[:, 2] == dist_class
                plot = sns.scatterplot(
                    x=dists[idx, 0],
                    y=dists[idx, 1],
                    hue=y[idx],
                    hue_norm=(0, 1),
                    palette="RdPu",
                    ax=axs[i],
                )
                axs[i].set(xlabel="mu", ylabel="sigma")
                plot.set_title(f"{DIST_CLASS_LABELS[dist_class]}")
                fig.suptitle(f"{set_label} predictions (unf)")
            plt.show()

            fig, axs = plt.subplots(
                1, n_dist_classes, figsize=(5 * n_dist_classes, 5)
            )
            for i, dist_class in enumerate(unique_dist_classes):
                idx = dists[:, 2] == dist_class
                plot = sns.scatterplot(
                    x=dists[idx, 0],
                    y=dists[idx, 1],
                    hue=torch.round(y[idx]),
                    hue_norm=(0, 1),
                    palette="RdPu",
                    ax=axs[i],
                )
                axs[i].set(xlabel="mu", ylabel="sigma")
                plot.set_title(f"{DIST_CLASS_LABELS[dist_class]}")
                fig.suptitle(f"{set_label} predictions (bin)")
            plt.show()

            acc = round(100 * (BinaryAccuracy()(y, labels)).item(), 3)
        print(f"{set_label} Accuracy: {acc}")

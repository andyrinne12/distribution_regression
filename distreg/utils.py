import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from scipy import stats
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy

from distreg.distributions.distribution import DIST_CLASS_LABELS


def sigmoid(X):
    return np.clip(1 / (1 + np.exp(X)), 1e-6, 1 - 1e-6)


def plot_multi_normals_predictions(
    model, train_dataset, test_dataset, xlabel, ylabel
):
    for dataset, set_label in [
        (train_dataset, "Train"),
        (test_dataset, "Test"),
    ]:
        data, labels, (means, covs, _) = dataset[:]
        labels = labels.flatten()
        loss_criterion = nn.BCELoss(reduction="mean")
        with torch.no_grad():
            y = model(data).flatten()
            plot = sns.scatterplot(
                x=means[:, 0],
                y=covs[:, 0, 1],
                hue=y[:],
                palette="RdPu",
            )
            plot.set_title(f"{set_label} predictions (cont)")
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.show()

            plot = sns.scatterplot(
                x=means[:, 0],
                y=covs[:, 0, 1],
                hue=torch.round(y[:]),
            )
            plot.set_title(f"{set_label} predictions (bin)")
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.show()

            acc = round(100 * (BinaryAccuracy()(y, labels)).item(), 3)
        loss = round(loss_criterion(y, labels).item(), 3)
        print(f"{set_label} Accuracy: {acc}, loss: {loss}")


def plot_normals_predictions(
    model, train_dataset, test_dataset, xlabel, ylabel
):
    for dataset, set_label in [
        (train_dataset, "Train"),
        (test_dataset, "Test"),
    ]:
        data, labels, (means, stds, _) = dataset[:]
        labels = labels.flatten()
        loss_criterion = nn.BCELoss(reduction="mean")
        with torch.no_grad():
            y = model(data).flatten()
            plot = sns.scatterplot(
                x=means[:, 0],
                y=stds[:, 0],
                hue=y[:],
                palette="RdPu",
            )
            plot.set_title(f"{set_label} predictions (cont)")
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.show()

            plot = sns.scatterplot(
                x=means[:, 0],
                y=stds[:, 0],
                hue=torch.round(y[:]),
            )
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plot.set_title(f"{set_label} predictions (bin)")
            plt.show()

            acc = round(100 * (BinaryAccuracy()(y, labels)).item(), 3)
        loss = round(loss_criterion(y, labels).item(), 3)
        print(f"{set_label} Accuracy: {acc}, loss: {loss}")


def plot_bin_class_predictions(
    model,
    train_dataset,
    test_dataset,
):
    for dataset, set_label in [
        (train_dataset, "Train"),
        (test_dataset, "Test"),
    ]:
        data, labels, (means, stds, classes) = dataset[:]
        means = means.flatten()
        stds = stds.flatten()
        classes = classes.flatten()
        labels = labels.flatten()
        loss_criterion = nn.BCELoss(reduction="mean")

        with torch.no_grad():
            y = model(data).flatten()
            unique_dist_classes = torch.unique(classes).tolist()
            n_dist_classes = len(unique_dist_classes)
            fig, axs = plt.subplots(
                1, n_dist_classes, figsize=(5 * n_dist_classes, 5)
            )
            for i, dist_class in enumerate(unique_dist_classes):
                c_axs = axs[i] if n_dist_classes > 1 else axs
                idx = classes == dist_class
                plot = sns.scatterplot(
                    x=means[idx],
                    y=stds[idx],
                    hue=y[idx],
                    hue_norm=(0, 1),
                    palette="RdPu",
                    ax=c_axs,
                )
                c_axs.set(xlabel="mean", ylabel="std")
                plot.set_title(f"{DIST_CLASS_LABELS[dist_class]}")
                fig.suptitle(f"{set_label} predictions (cont)")
            plt.show()

            fig, axs = plt.subplots(
                1, n_dist_classes, figsize=(5 * n_dist_classes, 5)
            )
            for i, dist_class in enumerate(unique_dist_classes):
                c_axs = axs[i] if n_dist_classes > 1 else axs
                idx = classes == dist_class
                plot = sns.scatterplot(
                    x=means[idx],
                    y=stds[idx],
                    hue=torch.round(y[idx]),
                    ax=c_axs,
                )
                c_axs.set(xlabel="mean", ylabel="std")
                plot.set_title(f"{DIST_CLASS_LABELS[dist_class]}")
                fig.suptitle(f"{set_label} predictions (bin)")
            plt.show()

            acc = round(100 * (BinaryAccuracy()(y, labels)).item(), 3)
        loss = round(loss_criterion(y, labels).item(), 3)
        print(f"{set_label} Accuracy: {acc}, loss: {loss}")


def gamma_pdf(mean, std):
    var = std**2
    mean_sq = mean**2
    a = mean_sq / var
    scale = mean / a
    return stats.gamma(a, scale=scale).pdf

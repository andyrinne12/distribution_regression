import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy


def sigmoid(X):
    return np.clip(1 / (1 + np.exp(X)), 1e-6, 1 - 1e-6)


def plot_normals_predictions(model, train_dataset, test_dataset):
    train_data, labels, train_dists = next(
        iter(DataLoader(train_dataset, batch_size=len(train_dataset)))
    )
    with torch.no_grad():
        y_train = model(train_data).flatten()
        plot = sns.scatterplot(
            x=train_dists[:, 0],
            y=train_dists[:, 1],
            hue=y_train[:],
            palette="RdPu",
        )
        plot.set_title("Train predictions (unf)")
        plt.show()

        plot = sns.scatterplot(
            x=train_dists[:, 0],
            y=train_dists[:, 1],
            hue=torch.round(y_train[:]),
        )
        plot.set_title("Train predictions (bin)")
        plt.show()
        train_acc = round(100 * (BinaryAccuracy()(y_train, labels)).item(), 3)
    print(f"Train Accuracy: {train_acc}")

    test_data, labels, test_dists = next(
        iter(DataLoader(test_dataset, batch_size=len(test_dataset)))
    )
    with torch.no_grad():
        y_test = model(test_data).flatten()
        plot = sns.scatterplot(
            x=test_dists[:, 0],
            y=test_dists[:, 1],
            hue=y_test[:],
            palette="RdPu",
        )
        plot.set_title("Test predictions (unf)")
        plt.show()

        plot = sns.scatterplot(
            x=test_dists[:, 0],
            y=test_dists[:, 1],
            hue=torch.round(y_test[:]),
        )
        plot.set_title("Test predictions (bin)")
        plt.show()
        test_acc = round(100 * (BinaryAccuracy()(y_test, labels)).item(), 3)
    print(f"Test accuracy: {test_acc}")


def plot_bin_class_predictions(
    model,
    train_dataset,
    test_dataset,
    class_labels,
):
    class1, class2 = class_labels
    train_data, labels, train_dists = next(
        iter(DataLoader(train_dataset, batch_size=len(train_dataset)))
    )

    with torch.no_grad():
        y_train = model(train_data).flatten()
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        plot = sns.scatterplot(
            x=train_dists[labels == 0, 0],
            y=train_dists[labels == 0, 1],
            hue=y_train[labels == 0],
            hue_norm=(0, 1),
            palette="RdPu",
            ax=axs[0],
        )
        axs[0].set(xlabel="mu", ylabel="sigma")
        plot.set_title(f"{class1}")
        plot = sns.scatterplot(
            x=train_dists[labels == 1, 0],
            y=train_dists[labels == 1, 1],
            hue=y_train[labels == 1],
            palette="RdPu",
            hue_norm=(0, 1),
            ax=axs[1],
        )
        axs[1].set(xlabel="mu", ylabel="sigma")
        plot.set_title(f"{class2}")
        fig.suptitle("Train predictions (unf)")
        plt.show()

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        plot = sns.scatterplot(
            x=train_dists[labels == 0, 0],
            y=train_dists[labels == 0, 1],
            hue=np.round(y_train[labels == 0]),
            ax=axs[0],
        )
        axs[0].set(xlabel="mu", ylabel="sigma")
        plot.set_title(f"{class1}")
        plot = sns.scatterplot(
            x=train_dists[labels == 1, 0],
            y=train_dists[labels == 1, 1],
            hue=np.round(y_train[labels == 1]),
            ax=axs[1],
        )
        axs[1].set(xlabel="mu", ylabel="sigma")
        plot.set_title(f"{class2}")
        fig.suptitle("Train predictions (bin)")
        plt.show()

        train_acc = round(100 * (BinaryAccuracy()(y_train, labels)).item(), 3)
    print(f"Train Accuracy: {train_acc}")

    test_data, labels, test_dists = next(
        iter(DataLoader(test_dataset, batch_size=len(test_dataset)))
    )
    with torch.no_grad():
        y_test = model(test_data).flatten()
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        plot = sns.scatterplot(
            x=test_dists[labels == 0, 0],
            y=test_dists[labels == 0, 1],
            hue=y_test[labels == 0],
            hue_norm=(0, 1),
            palette="RdPu",
            ax=axs[0],
        )
        axs[0].set(xlabel="mu", ylabel="sigma")
        plot.set_title(f"{class1}")
        plot = sns.scatterplot(
            x=test_dists[labels == 1, 0],
            y=test_dists[labels == 1, 1],
            hue=y_test[labels == 1],
            hue_norm=(0, 1),
            palette="RdPu",
            ax=axs[1],
        )
        axs[1].set(xlabel="mu", ylabel="sigma")
        plot.set_title(f"{class2}")
        fig.suptitle("Test predictions (unf)")
        plt.show()

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        plot = sns.scatterplot(
            x=test_dists[labels == 0, 0],
            y=test_dists[labels == 0, 1],
            hue=np.round(y_test[labels == 0]),
            ax=axs[0],
        )
        axs[0].set(xlabel="mu", ylabel="sigma")
        plot.set_title(f"{class1}")
        plot = sns.scatterplot(
            x=test_dists[labels == 1, 0],
            y=test_dists[labels == 1, 1],
            hue=np.round(y_test[labels == 1]),
            ax=axs[1],
        )
        axs[1].set(xlabel="mu", ylabel="sigma")
        plot.set_title(f"{class2}")
        fig.suptitle("Test predictions (bin)")
        plt.show()

        test_acc = round(100 * (BinaryAccuracy()(y_test, labels)).item(), 3)
    print(f"Test accuracy: {test_acc}")

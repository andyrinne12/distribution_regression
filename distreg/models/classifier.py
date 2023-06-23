import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torch import nn
from torchmetrics.classification import BinaryAccuracy


class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=1, bias=False):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x):
        out = self.linear(x)
        return torch.sigmoid(out)


def train_classifier(
    model,
    optimizer,
    train_dataset,
    test_dataset,
    epochs,
    verbose=False,
    plot_curve=False,
):
    criterion = nn.BCELoss(reduction="sum")

    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    for e in range(1, epochs + 1):
        train_loss = 0.0
        train_acc = 0.0
        (data, labels, _) = train_dataset[:]
        labels = labels.flatten()

        def closure1():
            model.zero_grad()
            optimizer.zero_grad()
            y = model(data).flatten()
            loss = criterion(y, labels)
            loss.backward()
            return loss

        optimizer.step(closure1)
        y = model(data).flatten()
        loss = criterion(y, labels)
        train_loss = loss.item() / y.shape[0]
        train_acc = BinaryAccuracy()(y, labels).item()

        with torch.no_grad():
            (test_data, labels_test, _) = test_dataset[:]
            labels_test = labels_test.flatten()

            y_test = model(test_data).flatten()
            loss = criterion(y_test, labels_test)
            test_loss = loss.item() / y_test.shape[0]
            test_acc = BinaryAccuracy()(y_test, labels_test).item()

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        train_accs.append(train_acc)
        test_accs.append(test_acc)

        if e % 50 == 0 and verbose:
            print(
                "Epoch: %d, train loss = %.4f, test loss = %.4f"
                % (e, train_loss, test_loss)
            )

    if plot_curve:
        sns.lineplot(train_losses, label="train")
        plot = sns.lineplot(test_losses, label="test")
        plot.set_title("Loss curve")
        plt.xlabel("epoch")
        plt.ylabel("BCE loss")
        plt.show()

    if plot_curve:
        sns.lineplot(train_accs, label="train")
        plot = sns.lineplot(test_accs, label="test")
        plot.set_title("Accuracy curve")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.show()

    return train_losses, test_losses

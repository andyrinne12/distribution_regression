import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
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
    device,
    train_dataset,
    test_dataset,
    epochs,
    batch_size=16,
    bfgs=False,
    verbose=False,
    plot_curve=False,
):
    criterion = nn.BCELoss(reduction="sum")

    batch_size = len(train_dataset) if bfgs else batch_size
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=len(test_dataset), shuffle=False
    )

    train_size = len(train_loader.dataset)
    test_size = len(test_loader.dataset)

    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    for e in range(1, epochs + 1):
        train_loss = 0.0
        train_acc = 0.0
        for _, (data, labels, _) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device).flatten()

            def closure1():
                model.zero_grad()
                optimizer.zero_grad()
                y = model(data).flatten()
                loss = criterion(y, labels)
                loss.backward()
                return loss

            if not bfgs:
                model.zero_grad()
                optimizer.zero_grad()

                y = model(data).flatten()
                loss = criterion(y, labels)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
            else:
                optimizer.step(closure1)
                y = model(data).flatten()
                loss = criterion(y, labels)
                train_loss += loss.item()
                train_acc += data.shape[0] * BinaryAccuracy()(y, labels).item()

        with torch.no_grad():
            test_loss = 0.0
            for _, (test_data, labels_test, _) in enumerate(test_loader):
                test_data = test_data.to(device)
                labels_test = labels_test.to(device).flatten()

                y_test = model(test_data).flatten()
                loss = criterion(y_test, labels_test)
                test_loss += loss.item()
                test_acc = BinaryAccuracy()(y_test, labels_test).item()

        train_loss /= train_size
        test_loss /= test_size

        train_acc /= train_size

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
        plt.show()

    if plot_curve:
        sns.lineplot(train_accs, label="train")
        plot = sns.lineplot(test_accs, label="test")
        plot.set_title("Accuracy curve")
        plt.show()

    return train_losses, test_losses

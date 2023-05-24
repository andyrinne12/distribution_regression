import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torch import nn


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim=1, bias=False):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64, bias=bias),
            nn.ReLU(),
            nn.Linear(64, 128, bias=bias),
            nn.ReLU(),
            nn.Linear(128, 256, bias=bias),
            nn.ReLU(),
            nn.Linear(256, output_dim, bias=bias),
        )

    def forward(self, x):
        out = self.layers(x)
        return out


def train_generator_only(
    distribution_lr,
    classifier,
    generator,
    optimizer,
    device,
    fixed_noise,
    epochs,
    verbose=False,
    plot_curve=False,
):
    criterion = nn.BCELoss(reduction="sum")
    losses = []
    preds = []

    for e in range(1, epochs + 1):
        generator.zero_grad()
        optimizer.zero_grad()

        fake_samples = generator(fixed_noise)

        features = distribution_lr.encode_to_features(fake_samples)
        pred_class = classifier(features)

        loss = criterion(pred_class[0], torch.tensor([1.0]))
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        preds.append(pred_class.item())

        if e % 50 == 0 and verbose:
            print(
                "Epoch: %d, train loss = %.4f, pred = %.4f"
                % (e, loss.item(), pred_class.item())
            )

            sns.histplot(fake_samples.detach().numpy())
            plt.show()

    if plot_curve:
        plot = sns.lineplot(losses)
        plot.set_title("Loss curve")
        plt.show()

    if plot_curve:
        plot = sns.lineplot(preds)
        plot.set_title("Predictions curve")
        plt.show()

    return losses, preds

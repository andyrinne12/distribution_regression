from collections import deque
import numpy as np

import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torch import nn

from distreg.gan_dist_set import GanDistributionSet
from distreg.models.classifier import Classifier


def roundl(arr, dec):
    return [round(n, dec) for n in arr]


def absl(arr):
    return [abs(n) for n in arr]


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim=1, bias=False):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 32, bias=bias),
            nn.GELU(),
            nn.Linear(32, 16, bias=bias),
            nn.GELU(),
            nn.Linear(16, 8, bias=bias),
            nn.GELU(),
            nn.Linear(8, output_dim, bias=bias),
        )

    def forward(self, x):
        out = self.layers(x)
        return out


def train_gan(
    gan_dist_set: GanDistributionSet,
    classifier,
    classifier_opt,
    classifier_lr,
    generator: Generator,
    generator_opt,
    batch_size,
    n_features,
    epochs,
    classifier_subepochs,
    generator_subepochs,
    device,
    verbose=False,
    save_plots=False,
):
    gen_losses = []
    gen_preds = []
    fixed_noise_dists = gan_dist_set.fixed_noise_dists
    noise_dists = gan_dist_set.noise_dists

    true_labels = torch.ones((noise_dists.n_distributions, 1)).flatten()
    means = [torch.mean(gan_dist_set.ground_samples).item()]
    stds = [torch.std(gan_dist_set.ground_samples).item()]
    hues = [1.0]

    for e in range(1, epochs + 1):
        for g in generator_opt.param_groups:
            g["lr"] = g["lr"] * 0.96
            print(f"LR: {g['lr']}")

        classifier = Classifier(n_features).to(device)
        classifier_opt = torch.optim.LBFGS(
            classifier.parameters(), lr=classifier_lr
        )

        gan_dist_set.resample_noise()
        output_sample = generator(noise_dists.samples).detach()

        means.append(torch.mean(output_sample).item())
        stds.append(torch.std(output_sample).item())
        hues.append(0.0)

        gan_dist_set.encode_and_append(output_sample[:5], 0)

        classifier_criterion = nn.BCELoss(
            reduction="mean"
            # , weight=gan_dist_set.loss_weights
        )
        data = gan_dist_set.dist_features
        labels = gan_dist_set.labels

        print("Classifier:")

        losses_queue = deque(maxlen=5)
        last_loss = 0.0
        for se in range(1, classifier_subepochs + 1):

            def closure():
                classifier.zero_grad()
                classifier_opt.zero_grad()
                y = classifier(data).flatten()
                loss = classifier_criterion(y, labels)
                loss.backward()
                return loss

            classifier_opt.step(closure)

            y = classifier(data).flatten()
            loss = classifier_criterion(y, labels)

            losses_queue.appendleft(loss.item() - last_loss)
            last_loss = loss.item()
            if max(absl(list(losses_queue))) < 1e-3:
                print(
                    f"    SE {se}: pred: {round(y[0].item(), 3)}, "
                    f"loss: {round(loss.item(), 3)}"
                )
                break

            if se % 50 == 0 and verbose:
                print(
                    f"    SE {se}: pred: {roundl(y.tolist(), 3)}, "
                    f"loss: {round(loss.item(), 3)}"
                )

        print("Generator:")
        generator_criterion = nn.BCELoss(reduction="sum")

        losses_queue = deque(maxlen=3)
        last_loss = 0.0
        for se in range(1, generator_subepochs + 1):
            total_loss = 0.0
            train_size = noise_dists.samples.shape[0]
            for batch_id in range(train_size // batch_size):
                idx = np.arange(
                    batch_id * batch_size,
                    min((batch_id + 1) * batch_size, train_size),
                )
                generator.zero_grad()
                generator_opt.zero_grad()

                output_sample = generator(noise_dists.samples[idx])
                features = gan_dist_set.encode(output_sample)

                y = classifier(features).flatten()
                loss = generator_criterion(y, true_labels[idx])

                loss.backward()
                generator_opt.step()

                total_loss += loss.item()

            losses_queue.appendleft(total_loss - last_loss)
            last_loss = total_loss
            if max(absl(list(losses_queue))) < 1e-2:
                print(
                    f"    SE {se}: pred: {round(y[0].item(), 3)}, "
                    f"loss: {round(total_loss, 3)}"
                )
                break

            if (se == 1 or se % 5 == 0) and verbose:
                print(
                    f"    SE {se}: pred: {round(y[0].item(), 3)}, "
                    f"loss: {round(total_loss, 3)}"
                )

        gen_losses.append(loss.item())
        gen_preds.append(y[0].item())

        if e % 1 == 0 and verbose:
            print(
                "Epoch: %d, train loss = %.4f, pred = %.4f"
                % (e, gen_losses[-1], gen_preds[-1])
            )

            print(
                "Output dist mean: %.2f std: %.2f"
                % (
                    torch.mean(output_sample[0], dim=0),
                    torch.std(output_sample[0], dim=0),
                )
            )

            fixed_output_sample = generator(fixed_noise_dists.samples)
            fig, axs = plt.subplots(1, 2, figsize=(6, 3))
            sns.histplot(fixed_output_sample[0].detach().numpy(), ax=axs[0])
            sns.histplot(
                gan_dist_set.ground_samples[0].detach().numpy(), ax=axs[1]
            )
            plt.show()
            if save_plots:
                fig.savefig(f"train_plots/histplot_{e}.png")


def train_generator_only(
    distribution_lr,
    classifier,
    generator,
    optimizer,
    device,
    fixed_noise_dists,
    ground_sample,
    epochs,
    verbose=False,
    plot_curve=False,
):
    criterion = nn.BCELoss(reduction="sum")
    losses = []
    preds = []
    true_labels = torch.ones(fixed_noise_dists.n_distributions)

    for e in range(1, epochs + 1):
        generator.zero_grad()
        optimizer.zero_grad()

        fake_samples = generator(fixed_noise_dists.samples)

        features = distribution_lr.samples_to_features(fake_samples)
        pred_class = classifier(features).flatten()

        loss = criterion(pred_class, true_labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        preds.append(pred_class[0].item())

        if e % 50 == 0 and verbose:
            print(
                "Epoch: %d, train loss = %.4f, pred = %.4f"
                % (e, loss.item(), pred_class[0].item())
            )

            print(
                "Output dist mean: %.2f std: %.2f"
                % (
                    torch.mean(fake_samples[0], dim=0),
                    torch.std(fake_samples[0], dim=0),
                )
            )

            _, axs = plt.subplots(1, 2, figsize=(8, 4))
            # axs[0].set_xlim(0, 400)
            # axs[1].set_xlim(0, 400)
            sns.histplot(fake_samples[0].detach().numpy(), ax=axs[0])
            sns.histplot(ground_sample.detach().numpy(), ax=axs[1])
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

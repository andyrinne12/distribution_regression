import glob
import os
from collections import deque

import numpy as np
import scipy
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
            nn.Linear(input_dim, output_dim, bias=bias),
        )
        with torch.no_grad():
            self.layers[0].bias += 1

    def forward(self, x):
        out = self.layers(x)
        return out


class GeneratorSimple(Generator):
    def __init__(self, input_dim, output_dim=1, bias=False):
        super().__init__(input_dim, output_dim=output_dim, bias=bias)
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 32, bias=bias),
            nn.LeakyReLU(0.2),
            nn.Linear(32, output_dim, bias=bias),
        )


class GeneratorDoubleHidden(Generator):
    def __init__(self, input_dim, output_dim=1, bias=False):
        super().__init__(input_dim, output_dim=output_dim, bias=bias)
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16, bias=bias),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 8, bias=bias),
            nn.LeakyReLU(0.2),
            nn.Linear(8, output_dim, bias=bias),
        )


def train_gan(
    gan_dist_set: GanDistributionSet,
    classifier_lr,
    generator: Generator,
    generator_opt,
    generator_scheduler,
    generator_lr_min,
    n_features,
    epochs,
    classifier_subepochs,
    generator_subepochs,
    verbose=False,
    save_plots=False,
    resample_each_subepoch=False,
    print_output_params=False,
):
    fixed_noise_sample = gan_dist_set.fixed_noise_dists.samples[0]
    noise_dists = gan_dist_set.noise_dists
    ground_sample = gan_dist_set.ground_samples[0]
    ground_params = torch.Tensor(
        [
            [
                torch.mean(ground_sample, dim=0).detach(),
                torch.std(ground_sample, dim=0).detach(),
            ]
        ]
    )
    true_labels = torch.ones((noise_dists.n_distributions, 1)).flatten()
    generator_criterion = nn.BCELoss(reduction="sum")

    class_losses = np.empty((epochs))

    gen_losses = np.empty((epochs))
    gen_preds = np.empty((epochs))

    wass_dists = np.empty((epochs))

    output_params = np.empty((epochs, 2))

    # Clean up plots folder
    files = glob.glob("train_plots/*")
    for f in files:
        os.remove(f)

    for e in range(1, epochs + 1):
        # Produce a generator output sample and add it to the distribution set
        generator.eval()
        output_sample = generator(noise_dists.samples[0]).detach()
        output_sample = output_sample.reshape(1, *output_sample.shape)
        gan_dist_set.encode_and_append(output_sample, 0)

        # Initialize the classifier
        classifier = Classifier(n_features)
        classifier_opt = torch.optim.LBFGS(
            classifier.parameters(), lr=classifier_lr
        )
        classifier_criterion = nn.BCELoss(
            reduction="sum", weight=gan_dist_set.loss_weights
        )

        data = gan_dist_set.dist_features
        labels = gan_dist_set.labels
        print("Classifier train:")

        # classifier_subepochs += 0.3
        # Train the classifier
        for se in range(1, int(classifier_subepochs) + 1):

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

            # Print training log
            if se % 5 == 0 and verbose:
                print(
                    f"    {se}: loss: {round(loss.item() / y.shape[0], 3)}, "
                    f"pred: {roundl(y.tolist(), 3)}"
                )

        # Append classifier losses
        class_losses[e - 1] = loss.item() / y.shape[0]

        # Generator training
        print(
            f"Generator training: (lr="
            f"{round(generator_scheduler.get_last_lr()[0], 3)})"
        )
        generator.train()
        # Resample noise (default)
        if not resample_each_subepoch:
            gan_dist_set.resample_noise()

        for se in range(1, int(generator_subepochs) + 1):
            # Resample noise (optionally)
            if resample_each_subepoch:
                gan_dist_set.resample_noise()

            generator.zero_grad()
            generator_opt.zero_grad()

            # Encode generated output
            output_sample = generator(noise_dists.samples[0])
            features = gan_dist_set.encode(
                output_sample.reshape(1, *output_sample.shape)
            )

            # Classify generated sample
            y = classifier(features).flatten()
            loss = generator_criterion(y, true_labels[:1])

            loss.backward()
            generator_opt.step()

            # Print training log
            if (se == 1 or se % 100 == 0) and verbose:
                print(
                    f"    SE {se}: loss: {round(loss.item(), 3)}, "
                    f"pred: {round(y[0].item(), 3)}"
                )

        # Update generator LR
        if generator_scheduler.get_last_lr()[0] > generator_lr_min:
            generator_scheduler.step()
        generator.eval()

        # Append generator losses
        gen_losses[e - 1] = loss.item()
        gen_preds[e - 1] = y[0].item()

        # Compute training statistics
        fixed_output_sample = generator(fixed_noise_sample).detach()

        output_mean = torch.mean(fixed_output_sample, dim=0).item()
        output_std = torch.std(fixed_output_sample, dim=0).item()
        output_params[e - 1, 0], output_params[e - 1, 1] = (
            output_mean,
            output_std,
        )

        wass_dists[e - 1] = scipy.stats.wasserstein_distance(
            fixed_output_sample.flatten(), ground_sample.flatten()
        )

        # Master training log
        if e % 1 == 0 and verbose:
            print(
                "### Epoch: %d\n train loss = %.4f, pred = %.4f, wass_d = %.4f"
                % (e, gen_losses[e - 1], gen_preds[e - 1], wass_dists[e - 1])
            )

            print("output mean: %.2f std: %.2f" % (output_mean, output_std))
            print(
                "ground mean: %.2f std: %.2f"
                % (ground_params[0, 0], ground_params[0, 1])
            )

            # Plot both histograms
            axs = sns.histplot(
                x=fixed_output_sample[:, 0],
                label="generated",
                color="#ff6a9e",
                kde=True,
            )
            sns.histplot(
                ground_sample[:, 0].detach(),
                color="#57b884",
                label="true",
                kde=True,
            )
            plt.title(f"Fixed-noise output and true sample (e. {e})")
            plt.legend()
            plt.show()

            # Save histograms plot
            if save_plots:
                axs.get_figure().savefig(f"train_plots/histplot_{e}.png")
            if print_output_params:
                sns.scatterplot(
                    x=output_params[:e, 0],
                    y=output_params[:e, 1],
                    alpha=(np.arange(e) + 1) / e * 0.9 + 0.1,
                    color="#ff6a9e",
                    label="generated",
                )
                sns.scatterplot(
                    x=ground_params[:, 0],
                    y=ground_params[:, 1],
                    color="#57b884",
                    label="true",
                )
                plt.title(
                    f"Fixed-noise generated output in the mean-std space (e."
                    f" {e})"
                )
                plt.legend()
                plt.xlabel("mean")
                plt.ylabel("std")
                plt.show()

    plt.xlabel("epoch")
    plt.ylabel("W1 dist")
    plot = sns.lineplot(wass_dists)
    plot.set(xticks=np.arange(0, wass_dists.shape[0] + 1, 5))
    plt.title("Wasserstein 1-dist between fixed-noise output and true sample")
    plt.show()

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plot = sns.lineplot(gen_losses, label="generator")
    sns.lineplot(class_losses, label="classifier")
    plot.set(xticks=np.arange(0, wass_dists.shape[0] + 1, 5))
    plt.title("Generator and classifier loss at end of epoch")
    plt.show()

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plot = sns.lineplot(class_losses, label="classifier")
    plot.set(xticks=np.arange(0, wass_dists.shape[0] + 1, 5))
    plt.title("Classifier loss at end of epoch")
    plt.show()

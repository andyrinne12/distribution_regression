import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torch import nn

from distreg.gan_dist_set import GanDistributionSet


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim=1, bias=False):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64, bias=bias),
            nn.ReLU(),
            nn.Linear(64, 32, bias=bias),
            nn.ReLU(),
            nn.Linear(32, 16, bias=bias),
            nn.ReLU(),
            nn.Linear(16, 8, bias=bias),
            nn.ReLU(),
            nn.Linear(8, output_dim, bias=bias),
        )

    def forward(self, x):
        out = self.layers(x)
        return out


def train_gan(
    gan_dist_set: GanDistributionSet,
    classifier,
    classifier_opt,
    generator: Generator,
    generator_opt,
    epochs,
    classifier_subepochs,
    generator_subepochs,
    device,
    verbose=False,
):
    gen_losses = []
    gen_preds = []
    noise_dists = gan_dist_set.fixed_noise_dists

    true_labels = torch.ones((noise_dists.n_distributions, 1))

    for e in range(1, epochs + 1):
        output_sample = generator(noise_dists.samples).detach()
        gan_dist_set.encode_and_append(output_sample, 0)

        classifier_criterion = nn.BCELoss(
            reduction="mean"
            # , weight=gan_dist_set.loss_weights
        )
        data = gan_dist_set.dist_features
        labels = gan_dist_set.labels
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
            if se % 200 == 0 and verbose:
                print(f"Classif loss: {round(loss.item(), 3)}")

        generator_criterion = nn.BCELoss(reduction="sum")
        for se in range(1, generator_subepochs + 1):
            generator.zero_grad()
            generator_opt.zero_grad()

            output_sample = generator(noise_dists.samples)
            features = gan_dist_set.encode(output_sample)
            y = classifier(features)
            loss = generator_criterion(y, true_labels)
            loss.backward()

            # for p in generator.parameters():
            #     print(p.grad)

            generator_opt.step()

            if se % 400 == 0 and verbose:
                print(
                    f"Gen pred: {round(y[0].item(), 3)}, loss: "
                    f"{round(loss.item(), 3)}"
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

            _, axs = plt.subplots(1, 2, figsize=(8, 4))
            # axs[0].set_xlim(0, 400)
            # axs[1].set_xlim(0, 400)
            sns.histplot(output_sample[0].detach().numpy(), ax=axs[0])
            sns.histplot(
                gan_dist_set.ground_samples[0].detach().numpy(), ax=axs[1]
            )
            plt.show()


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

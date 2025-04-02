# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-01-27)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
#
# Significant extension by Søren Hauberg, 2024

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from tqdm import tqdm
from copy import deepcopy
import os
import math
import matplotlib.pyplot as plt
from tqdm import trange

from geodesics import compute_geodesic, curve
from utils import get_all_labels_and_latents

class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int]
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(GaussianDecoder, self).__init__()
        self.decoder_net = decoder_net
        # self.std = nn.Parameter(torch.ones(28, 28) * 0.5, requires_grad=True) # In case you want to learn the std of the gaussian.

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor]
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        means = self.decoder_net(z)
        return td.Independent(td.Normal(loc=means, scale=1e-1), 3)


class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """

    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module]
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """

        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x)
        z = q.rsample()

        elbo = torch.mean(
            self.decoder(z).log_prob(x) - q.log_prob(z) + self.prior().log_prob(z)
        )
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.

        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()

    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)


def energy(weights, decoders, c0, c1, N, num_models):
    """
    Computes the energy approximation of the curve for multiple models using Monte Carlo estimation.
    (eq. 8.7 in the book)
    """
    t = torch.linspace(0, 1, N + 1)
    c_points = curve(t, c0, c1, weights)

    total_energy = 0.0
    
    # Looping over all decoders and computing the energy for each model
    for i in range(num_models):
        decoder_fun = decoders[i]
        f_vals = decoder_fun(c_points).mean
        diffs = f_vals[1:] - f_vals[:-1]
        total_energy += (diffs ** 2).sum()
    
    # Averaging the energy over all models (Monte Carlo estimation)
    return total_energy / num_models

def compute_geodesic_ensemble(c0, c1, decoders, num_models, polyn_order=4, latent_dim=2, N=100, num_iterations=2500, lr=1e-3, debug=False):
    """
    Computes an approximate geodesic between two latent points using energy minimization via Adam.
    Includes optional early stopping, and Monte Carlo estimation of the energy for multiple models.
    """
    weights = torch.randn(latent_dim, polyn_order - 1, requires_grad=True)
    
    if debug:
        print(f'Init Energy for {weights}: {energy(weights, decoders, c0, c1, N, num_models).item()}')
    
    optimizer = torch.optim.Adam([weights], lr)
    
    best_energy = float('inf')
    steps_since_improvement = 0

    for i in trange(num_iterations, desc="Optimizing geodesic"):
        optimizer.zero_grad()
        E = energy(weights, decoders, c0, c1, N, num_models)
        E.backward()
        optimizer.step()

        current_energy = E.item()

        if best_energy - current_energy > 1e-4:
            best_energy = current_energy
            steps_since_improvement = 0
        else:
            steps_since_improvement += 1

        if steps_since_improvement >= 200:
            if debug:
                print(f"Early stopping at iteration {i}, Energy: {current_energy}")
            break
        
        if i % 100 == 0 and debug:
            print(f"Iteration {i}, Energy: {current_energy}")
    
    if debug:
        print(f"Final weights: {weights}. Final Energy: {energy(weights, decoders, c0, c1, N, num_models).item()}")
    
    return curve(torch.linspace(0, 1, N + 1), c0, c1, weights)

def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """

    num_steps = len(data_loader) * epochs
    epoch = 0

    def noise(x, std=0.05):
        eps = std * torch.randn_like(x)
        return torch.clamp(x + eps, min=0.0, max=1.0)

    with tqdm(range(num_steps)) as pbar:
        for step in pbar:
            try:
                x = next(iter(data_loader))[0]
                x = noise(x.to(device))
                model = model
                optimizer.zero_grad()
                # from IPython import embed; embed()
                loss = model(x)
                loss.backward()
                optimizer.step()

                # Report
                if step % 5 == 0:
                    loss = loss.detach().cpu()
                    pbar.set_description(
                        f"total epochs ={epoch}, step={step}, loss={loss:.1f}"
                    )

                if (step + 1) % len(data_loader) == 0:
                    epoch += 1
            except KeyboardInterrupt:
                print(
                    f"Stopping training at total epoch {epoch} and current loss: {loss:.1f}"
                )
                break


if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        default="train",
        choices=["train", "sample", "eval", "geodesics", "ensemble_geodesics", "cov_analysis"],
        help="what to do when running the script (default: %(default)s)",
    )
    parser.add_argument(
        "--experiment-folder",
        type=str,
        default="experiment",
        help="folder to save and load experiment results in (default: %(default)s)",
    )
    parser.add_argument(
        "--samples",
        type=str,
        default="samples.png",
        help="file to save samples in (default: %(default)s)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="torch device (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="batch size for training (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs-per-decoder",
        type=int,
        default=50,
        metavar="N",
        help="number of training epochs per each decoder (default: %(default)s)",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=2,
        metavar="N",
        help="dimension of latent variable (default: %(default)s)",
    )
    parser.add_argument(
        "--num-decoders",
        type=int,
        default=3,
        metavar="N",
        help="number of decoders in the ensemble (default: %(default)s)",
    )
    parser.add_argument(
        "--num-reruns",
        type=int,
        default=10,
        metavar="N",
        help="number of reruns (default: %(default)s)",
    )
    parser.add_argument(
        "--num-curves",
        type=int,
        default=10,
        metavar="N",
        help="number of geodesics to plot (default: %(default)s)",
    )
    parser.add_argument(
        "--num-t",  # number of points along the curve
        type=int,
        default=20,
        metavar="N",
        help="number of points along the curve (default: %(default)s)",
    )

    args = parser.parse_args()
    print("# Options")
    for key, value in sorted(vars(args).items()):
        print(key, "=", value)

    device = args.device

    # Load a subset of MNIST and create data loaders
    def subsample(data, targets, num_data, num_classes):
        idx = targets < num_classes
        new_data = data[idx][:num_data].unsqueeze(1).to(torch.float32) / 255
        new_targets = targets[idx][:num_data]

        return torch.utils.data.TensorDataset(new_data, new_targets)

    num_train_data = 2048
    num_classes = 3
    train_tensors = datasets.MNIST(
        "data/",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    test_tensors = datasets.MNIST(
        "data/",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    train_data = subsample(
        train_tensors.data, train_tensors.targets, num_train_data, num_classes
    )
    test_data = subsample(
        test_tensors.data, test_tensors.targets, num_train_data, num_classes
    )

    mnist_train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True
    )
    mnist_test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False
    )

    # Define prior distribution
    M = args.latent_dim

    def new_encoder():
        encoder_net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(512, 2 * M),
        )
        return encoder_net

    def new_decoder():
        decoder_net = nn.Sequential(
            nn.Linear(M, 512),
            nn.Unflatten(-1, (32, 4, 4)),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=0),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
        )
        return decoder_net

    # Choose mode to run
    if args.mode == "train":

        experiments_folder = args.experiment_folder
        os.makedirs(f"{experiments_folder}", exist_ok=True)
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(
            model,
            optimizer,
            mnist_train_loader,
            args.epochs_per_decoder,
            args.device,
        )
        os.makedirs(f"{experiments_folder}", exist_ok=True)

        torch.save(
            model.state_dict(),
            f"{experiments_folder}/model.pt",
        )

    elif args.mode == "sample":
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt"))
        model.eval()

        with torch.no_grad():
            samples = (model.sample(64)).cpu()
            save_image(samples.view(64, 1, 28, 28), args.samples)

            data = next(iter(mnist_test_loader))[0].to(device)
            recon = model.decoder(model.encoder(data).mean).mean
            save_image(
                torch.cat([data.cpu(), recon.cpu()], dim=0), "reconstruction_means.png"
            )

    elif args.mode == "eval":
        # Load trained model
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt"))
        model.eval()

        elbos = []
        with torch.no_grad():
            for x, y in mnist_test_loader:
                x = x.to(device)
                elbo = model.elbo(x)
                elbos.append(elbo)
        mean_elbo = torch.tensor(elbos).mean()
        print("Print mean test elbo:", mean_elbo)

    elif args.mode == "geodesics":

        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt", weights_only=True))
        model.eval()

        with torch.no_grad():
            x, y = next(iter(mnist_test_loader))
            x = x.to(device)
            latent = model.encoder(x).rsample()
        
        indices = torch.randperm(latent.size(0))[:(2 * args.num_curves)]

        chosen_pairs = list(zip(latent[indices[:args.num_curves]], latent[indices[args.num_curves:]]))
                        
        decoder_fun = lambda x: model.decoder(x).mean
        
        geodesics = tuple(map(lambda pair: compute_geodesic(pair[0], pair[1], decoder_fun),chosen_pairs))
        
        for i, curve in enumerate(geodesics):
            if curve is not None:
                plt.plot(curve[:, 0].detach().numpy(), curve[:, 1].detach().numpy(), linestyle='-', linewidth=1, label=str(i), color='black')
        
        # plotting the entire space 
        all_latents, all_labels = get_all_labels_and_latents(model, mnist_test_loader)
        plt.scatter(all_latents[:, 0].cpu(), all_latents[:, 1].cpu(), c=all_labels.cpu(), cmap='winter', alpha=0.3)

        plt.title('Latent Space')
        plt.show()

    elif args.mode == "ensemble_geodesics":
        experiment_folder = args.experiment_folder
        model_range = range(0,2)
        num_curves = args.num_curves
        decoders = []
        for i in model_range:
            model = VAE(
                GaussianPrior(M), 
                GaussianDecoder(new_decoder()),
                GaussianEncoder(new_encoder())
            ).to(device)
            model.load_state_dict(torch.load(f"{experiment_folder}/model{i}.pt", weights_only=True))
            model.eval()
            
            decoder_fun = lambda x: model.decoder(x)
            decoders.append(decoder_fun)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        with torch.no_grad():
            x, y = next(iter(mnist_test_loader))
            x = x.to(device)
            latent = model.encoder(x).rsample()
        
        # Randomly selecting pairs of latent points
        indices = torch.randperm(latent.size(0))[:(2 * num_curves)]
        chosen_pairs = list(zip(latent[indices[:num_curves]], latent[indices[num_curves:]]))
        
        # Computing geodesics for the chosen pairs
        geodesics = tuple(map(lambda pair: compute_geodesic_ensemble(pair[0], pair[1], decoders, len(decoders)), chosen_pairs))

        for i, curve in enumerate(geodesics):
            if curve is not None:
                plt.plot(curve[:, 0].detach().numpy(), curve[:, 1].detach().numpy(), linestyle='-', linewidth=1, label=str(i), color='black')
        
        all_latents, all_labels = get_all_labels_and_latents(model, mnist_test_loader)
        plt.scatter(all_latents[:, 0].cpu(), all_latents[:, 1].cpu(), c=all_labels.cpu(), cmap='winter', alpha=0.3)

        plt.title('Latent Space and Geodesics')
        plt.show()

    elif args.mode == "cov_analysis":
        import numpy as np
        import pandas as pd

        from euclidean import euclidean_distance 

        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder())
        ).to(device)
        model.load_state_dict(torch.load(f"{args.experiment_folder}/model0.pt", map_location=device))
        model.eval()

        def compute_distance_stats(c0, c1, decoder_modules):
            euclidean_dists = []
            geodesic_dists = []

            for decoder in decoder_modules:
                z0 = decoder(c0.unsqueeze(0)).mean.squeeze()  # [1, M] -> [28,28]
                z1 = decoder(c1.unsqueeze(0)).mean.squeeze()
                euclidean_dists.append(euclidean_distance(z0, z1).item())

            # Geodesic is computed across ensemble
            path = compute_geodesic_ensemble(c0, c1, decoder_modules, len(decoder_modules), N=50)
            diff = path[1:] - path[:-1]
            geodesic_dists.append(diff.norm(dim=1).sum().item())

            return euclidean_dists, geodesic_dists

        def compute_cov(values):
            mean = np.mean(values)
            std = np.std(values)
            return std / mean if mean != 0 else 0

        num_pairs = 10
        test_pairs = []

        with torch.no_grad():
            x, _ = next(iter(mnist_test_loader))
            x = x.to(device)
            latents = model.encoder(x).rsample()
            indices = torch.randperm(latents.size(0))[:2 * num_pairs]
            test_pairs = list(zip(latents[indices[:num_pairs]], latents[indices[num_pairs:]]))

        results = []

        for num_dec in [1, 2, 3]:
            model_paths = [f"{args.experiment_folder}/model{i}.pt" for i in range((num_dec - 1) * 10, num_dec * 10)]
            decoder_fns = []
            for path in model_paths:
                model = VAE(
                    GaussianPrior(M),
                    GaussianDecoder(new_decoder()),
                    GaussianEncoder(new_encoder())
                ).to(device)
                model.load_state_dict(torch.load(path, map_location=device))
                model.eval()
                decoder_fns.append(model.decoder)

            cov_e_list = []
            cov_g_list = []

            for c0, c1 in tqdm(test_pairs, desc=f"CoV: {num_dec} decoders"):
                eucl_dists, geo_dists = compute_distance_stats(c0, c1, decoder_fns)
                cov_e_list.append(compute_cov(eucl_dists))
                cov_g_list.append(compute_cov(geo_dists))

            results.append({
                "num_decoders": num_dec,
                "cov_euclidean": np.mean(cov_e_list),
                "cov_geodesic": np.mean(cov_g_list),
            })

        df = pd.DataFrame(results)
        print(df)

        # Plotting
        plt.plot(df["num_decoders"], df["cov_euclidean"], label="Euclidean", marker='o')
        plt.plot(df["num_decoders"], df["cov_geodesic"], label="Geodesic", marker='s')
        plt.xlabel("Number of Ensemble Decoders")
        plt.ylabel("Average Coefficient of Variation (CoV)")
        plt.title("CoV of Euclidean vs Geodesic Distance")
        plt.legend()
        plt.grid(True)
        plt.savefig("cov_plot.png")
        plt.show()
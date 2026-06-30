"""
Autoencoder for DeepADEFCM - learns latent representation for fuzzy clustering.
Architecture: Input -> Encoder -> Latent (bottleneck) -> Decoder -> Reconstruction
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from loguru import logger


class Encoder(nn.Module):
    """Encoder network: input_dim -> latent_dim through hidden layers."""

    def __init__(self, input_dim, latent_dim=10, hidden_dims=None, activation=nn.ReLU, dropout=0.0):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, latent_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    """Decoder network: latent_dim -> input_dim (mirrors encoder)."""

    def __init__(self, latent_dim, output_dim, hidden_dims=None, activation=nn.ReLU, dropout=0.0):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128, 256]

        layers = []
        prev_dim = latent_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        return self.decoder(z)


class Autoencoder(nn.Module):
    """
    Full autoencoder: input -> encode -> latent z -> decode -> reconstruction.

    Args:
        input_dim: Input feature dimensionality
        latent_dim: Bottleneck latent dimension (default: 10, can be auto-discovered)
        hidden_dims: List of hidden layer sizes for encoder (decoder mirrors)
        activation: Activation function class
        dropout: Dropout probability (0 = no dropout)
    """

    def __init__(self, input_dim, latent_dim=10, hidden_dims=None, activation=nn.ReLU, dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = [max(256, input_dim), max(128, input_dim // 2), max(64, input_dim // 4)]
        decoder_hidden_dims = list(reversed(hidden_dims))

        self.encoder = Encoder(input_dim, latent_dim, hidden_dims, activation, dropout)
        self.decoder = Decoder(latent_dim, input_dim, decoder_hidden_dims, activation, dropout)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def encode(self, x):
        """Encode input to latent representation."""
        return self.encoder(x)

    def decode(self, z):
        """Decode latent to reconstruction."""
        return self.decoder(z)

    def get_latent(self, x):
        """Get latent representation (numpy array)."""
        self.eval()
        with torch.no_grad():
            z = self.encoder(x if isinstance(x, torch.Tensor) else torch.FloatTensor(x))
        return z.cpu().numpy()


class VariationalEncoder(nn.Module):
    """Variational encoder: outputs mu and log_var for reparameterization."""

    def __init__(self, input_dim, latent_dim=10, hidden_dims=None, activation=nn.ReLU):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(activation())
            prev_dim = h_dim

        self.shared = nn.Sequential(*layers)
        self.mu = nn.Linear(prev_dim, latent_dim)
        self.log_var = nn.Linear(prev_dim, latent_dim)

    def forward(self, x):
        h = self.shared(x)
        mu = self.mu(h)
        log_var = self.log_var(h)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, log_var


class VAutoencoder(nn.Module):
    """Variational Autoencoder for DeepADEFCM."""

    def __init__(self, input_dim, latent_dim=10, hidden_dims=None, activation=nn.ReLU):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        decoder_hidden_dims = list(reversed(hidden_dims))

        self.encoder = VariationalEncoder(input_dim, latent_dim, hidden_dims, activation)

        dec_layers = []
        prev_dim = latent_dim
        for h_dim in decoder_hidden_dims:
            dec_layers.append(nn.Linear(prev_dim, h_dim))
            dec_layers.append(activation())
            prev_dim = h_dim
        dec_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z, mu, log_var = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z, mu, log_var

    def encode(self, x):
        z, _, _ = self.encoder(x if isinstance(x, torch.Tensor) else torch.FloatTensor(x))
        return z

    def get_latent(self, x):
        self.eval()
        with torch.no_grad():
            z = self.encode(x)
        return z.cpu().numpy()


def pretrain_autoencoder(model, X, epochs=100, batch_size=256, lr=1e-3, device='cpu',
                         weight_decay=1e-5, verbose=True):
    """
    Pretrain autoencoder with reconstruction loss only.

    Args:
        model: Autoencoder or VAutoencoder instance
        X: numpy array of shape (n_samples, n_features)
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        device: 'cpu' or 'cuda'
        weight_decay: L2 regularization
        verbose: Print progress

    Returns:
        losses: List of loss values per epoch
    """
    device = torch.device(device)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    X_tensor = torch.FloatTensor(X).to(device)
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch in dataloader:
            x = batch[0]
            optimizer.zero_grad()

            output = model(x)
            if len(output) == 2:  # Autoencoder
                x_recon, z = output
                recon_loss = nn.MSELoss()(x_recon, x)
            else:  # VAutoencoder
                x_recon, z, mu, log_var = output
                recon_loss = nn.MSELoss()(x_recon, x)
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                recon_loss += 0.001 * kl_loss

            recon_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += recon_loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        if verbose and (epoch + 1) % 20 == 0:
            logger.info(f"AE Pretrain Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f}")

    return losses


def compute_reconstruction_error(model, X, device='cpu'):
    """Compute reconstruction error for dataset."""
    device = torch.device(device)
    model.eval()
    X_tensor = torch.FloatTensor(X).to(device)
    with torch.no_grad():
        output = model(X_tensor)
        if len(output) == 2:
            x_recon, _ = output
        else:
            x_recon, _, _, _ = output
        error = nn.MSELoss(reduction='mean')(x_recon, X_tensor).item()
    return error

import torch
import torch.nn as nn
import numpy as np


class KYCAutoencoder(nn.Module):
    """
    Autoencoder for KYC document feature reconstruction.
    High reconstruction error = document is anomalous (forged/synthetic).
    """

    def __init__(self, input_dim: int = 128, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.ReLU(),
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 96),
            nn.ReLU(),
            nn.Linear(96, input_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        reconstructed, _ = self.forward(x)
        return nn.functional.mse_loss(reconstructed, x, reduction="none").mean(dim=-1)


class KYCDiscriminator(nn.Module):
    """
    GAN Discriminator trained to distinguish legitimate KYC documents
    from synthesized or physically altered forgeries.
    Detects pixel-level artifacts, metadata mismatches,
    font irregularities, and biometric divergence.
    """

    def __init__(self, input_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class KYCGenerator(nn.Module):
    """
    GAN Generator — synthesizes realistic fraudulent KYC document features.
    Used during training to continuously improve Discriminator sensitivity.
    Also used for synthetic data augmentation to address class imbalance.
    """

    def __init__(self, latent_dim: int = 64, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class GANAutoencoderKYC(nn.Module):
    """
    Combined GAN + Autoencoder KYC fraud detector.
    Inference uses both the Discriminator score and the
    Autoencoder reconstruction error to produce a final
    fraud probability — two independent signals fused together.
    """

    def __init__(self, input_dim: int = 128, latent_dim: int = 32, gan_latent: int = 64):
        super().__init__()
        self.autoencoder = KYCAutoencoder(input_dim, latent_dim)
        self.discriminator = KYCDiscriminator(input_dim)
        self.generator = KYCGenerator(gan_latent, input_dim)

        # Fusion layer: combine discriminator score + reconstruction error
        self.fusion = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        disc_score = self.discriminator(x)
        recon_error = self.autoencoder.reconstruction_error(x).unsqueeze(-1)
        # Discriminator outputs high for REAL docs; invert for fraud score
        fraud_from_disc = 1.0 - disc_score
        fused_input = torch.cat([fraud_from_disc, recon_error], dim=-1)
        return self.fusion(fused_input)

    @torch.no_grad()
    def predict(self, doc_features: np.ndarray) -> float:
        self.eval()
        x = torch.tensor(doc_features, dtype=torch.float32).unsqueeze(0)
        return self.forward(x).item()

    def generate_synthetic_fraud(self, n_samples: int = 100) -> np.ndarray:
        """Generate synthetic fraudulent KYC feature vectors for data augmentation."""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.generator.net[0].in_features)
            return self.generator(z).numpy()
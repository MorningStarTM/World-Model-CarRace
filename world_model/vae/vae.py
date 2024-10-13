import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvVAE(nn.Module):

    def __init__(self, input_channels=1, hidden_dim=256, latent_dim=64):
        super(ConvVAE, self).__init__()

        # Encoder: using conv layers
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),  # (batch_size, 32, 14, 14)
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # (batch_size, 64, 7, 7)
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # (batch_size, 128, 3, 3)
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, hidden_dim, kernel_size=3)  # (batch_size, hidden_dim, 1, 1)
        )

        # Latent mean and log variance
        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
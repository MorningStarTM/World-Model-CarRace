import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from world_model.logger import logger
import torch.optim as optim


class ConvVAE(nn.Module):
    def __init__(self, latent_dim):
        super(ConvVAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # (B, 32, 48, 48)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # (B, 64, 24, 24)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # (B, 128, 12, 12)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # (B, 256, 6, 6)
            nn.ReLU(),
            nn.Flatten(),  # (B, 256*6*6)
        )
        
        self.fc_mu = nn.Linear(256*6*6, latent_dim)
        self.fc_logvar = nn.Linear(256*6*6, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, 256*6*6)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (B, 128, 12, 12)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # (B, 64, 24, 24)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # (B, 32, 48, 48)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),     # (B, 3, 96, 96)
            nn.Sigmoid()  # Output values in the range [0, 1]
        )
        
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.fc_dec(z).view(-1, 256, 6, 6)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, logvar
    
    def save(self, path, optimizer:optim=None):
        os.makedirs(path, exist_ok=True)
        checkpoint = {
            'model': self.state_dict(),
        }
        if optimizer is not None:
            checkpoint['optimizer'] = optimizer.state_dict()
        torch.save(checkpoint, os.path.join(path, 'vae_checkpoint.pth'))
        logger.info(f"VAE checkpoint (model+optimizer) saved at {os.path.join(path, 'vae_checkpoint.pth')}")

    def load(self, path, optimizer=None):
        checkpoint_path = os.path.join(path, 'vae_checkpoint.pth')
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.load_state_dict(checkpoint['model'])
        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info(f"VAE checkpoint loaded from {checkpoint_path}")


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

        


class WMVAE(nn.Module):
    def __init__(self, img_channel, latent_dim) -> None:
        super(WMVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(img_channel, 32, kernel_size=4, stride=2),  # (B, 32, 48, 48)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # (B, 64, 24, 24)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2), # (B, 128, 12, 12)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2), # (B, 256, 6, 6)
            nn.ReLU(),
            nn.Flatten(),  # (B, 256*6*6)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2),  # (B, 128, 12, 12)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),   # (B, 64, 24, 24)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),    # (B, 32, 48, 48)
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_channel, kernel_size=6, stride=2),     # (B, img_channel, 96, 96)
            nn.Sigmoid()  # Output values in the range [0, 1]
        )

        self.fc1 = nn.Linear(latent_dim, 1024)
        self.fc_mu = nn.Linear(256*2*2, latent_dim)
        self.fc_logvar = nn.Linear(256*2*2, latent_dim)


    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        sigma = logvar.exp()
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)
        
        x = F.relu(self.fc1(z))
        x = x.unsqueeze(-1).unsqueeze(-1)
        recon = self.decoder(x)
        return recon, mu, logvar
    
    def save(self, path, optimizer:optim=None):
        os.makedirs(path, exist_ok=True)
        checkpoint = {
            'model': self.state_dict(),
        }
        if optimizer is not None:
            checkpoint['optimizer'] = optimizer.state_dict()
        torch.save(checkpoint, os.path.join(path, 'wm_vae_checkpoint.pth'))
        logger.info(f"VAE checkpoint (model+optimizer) saved at {os.path.join(path, 'wm_vae_checkpoint.pth')}")

    def load(self, path, optimizer=None):
        checkpoint_path = os.path.join(path, 'wm_vae_checkpoint.pth')
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.load_state_dict(checkpoint['model'])
        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info(f"World Model VAE checkpoint loaded from {checkpoint_path}")

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvVAE(nn.Module):
    def __init__(self, input_channels=3, latent_dim=200):
        super(ConvVAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),  # 96x96 -> 48x48
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 48x48 -> 24x24
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 24x24 -> 12x12
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 12x12 -> 6x6
            nn.LeakyReLU(0.2)
        )
        
        # Latent space: mean and variance
        self.fc_mean = nn.Linear(256 * 6 * 6, latent_dim)
        self.fc_logvar = nn.Linear(256 * 6 * 6, latent_dim)
        
        # Decoder: FC layer followed by transposed convolutions
        self.fc_decode = nn.Linear(latent_dim, 256 * 6 * 6)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 6x6 -> 12x12
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 12x12 -> 24x24
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 24x24 -> 48x48
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),  # 48x48 -> 96x96
            nn.Sigmoid()  # Output between 0 and 1 for images
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)


    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar



    def reparameterization(self, mean, logvar):
        std = torch.exp(0.5 * logvar)  # Standard deviation
        epsilon = torch.randn_like(std).to(self.device)  # Sample epsilon
        z = mean + std * epsilon  # Reparameterization trick
        return z
    
    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(x.size(0), 256, 6, 6)  # Reshape back to convolutional size
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar
import torch
import torch.nn as nn
import torch.optim as optim


class Encoder(nn.Module):
    def __init__(self, latent_dim=(3, 3, 3)):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),  # (48, 48, 16)
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # (24, 24, 32)
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # (12, 12, 64)
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # (6, 6, 128)
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # (3, 3, 256)
            nn.ReLU(True),
            nn.Conv2d(256, latent_dim[2], kernel_size=1)  # (3, 3, latent_dim[2])
        )

    
    def forward(self, x):
        return self.encoder(x)
    

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))



# Decoder class
class Decoder(nn.Module):
    def __init__(self, latent_dim=(3, 3, 3)):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim[2], 256, kernel_size=4, stride=2, padding=1), # (6, 6, 256)
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # (12, 12, 128)
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (24, 24, 64)
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # (48, 48, 32)
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # (96, 96, 16)
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1),   # (96, 96, 3)
            nn.Sigmoid()  # To ensure pixel values are in range [0, 1]
        )
    
    def forward(self, x):
        return self.decoder(x)
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))



# Autoencoder class that combines Encoder and Decoder
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=(3, 3, 3)):
        super(Autoencoder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
    

    def save(self, encoder_path, decoder_path):
        self.encoder.save(encoder_path)
        self.decoder.save(decoder_path)

    def load(self, encoder_path, decoder_path):
        self.encoder.load(encoder_path)
        self.decoder.load(decoder_path)



"""# Example usage
if __name__ == '__main__':
    # Instantiate the autoencoder
    latent_dim = (3, 3, 3)
    autoencoder = Autoencoder(latent_dim)
    autoencoder.to(autoencoder.device)

    # Example input (batch size of 1 for simplicity)
    input_image = torch.rand((1, 3, 96, 96)).to(autoencoder.device)  # Random image with shape (1, 3, 96, 96)

    # Forward pass
    output_image = autoencoder(input_image)
    print("Input shape:", input_image.shape)
    print("Output shape:", output_image.shape)"""
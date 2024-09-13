from ae import Autoencoder
import torch


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
    print("Output shape:", output_image.shape)
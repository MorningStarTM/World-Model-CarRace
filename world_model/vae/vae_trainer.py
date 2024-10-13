import torch
import os
from .vae import ConvVAE
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F


class VAETrainer:
    def __init__(self, model:ConvVAE, batch_size:int) -> None:
        pass

    def vae_loss(self, reconstructed, original, mu, logvar):
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstructed, original, reduction='sum')
        
        # KL Divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
        return recon_loss + kl_loss

    
    def train_vae(self, model, dataloader, epochs=20, learning_rate=1e-3, device='cuda'):
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        model.to(device)
        model.train()

        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(dataloader):
                images = data
                images = images.to(device)

                optimizer.zero_grad()
                reconstructed, mu, logvar = model(images)
                loss = self.vae_loss(reconstructed, images, mu, logvar)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(dataloader.dataset):.4f}")

            # Generate images at the end of each epoch
            self.generate_and_plot_images(model, epoch, device)

    
    def generate_and_plot_images(self, num_images=8):
        """Generate and plot images after each epoch."""
        self.model.eval()  # Set the model to evaluation mode

        with torch.no_grad():
            # Sample random latent vectors
            z = torch.randn(num_images, self.model.latent_dim).to(self.device)
            
            # Decode the latent vectors into images
            generated_images = self.model.decode(z).cpu()

        # Plot the generated images
        fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 2))
        
        for i in range(num_images):
            axes[i].imshow(generated_images[i].permute(1, 2, 0))  # Adjusting tensor shape for plotting
            axes[i].axis('off')  # Turn off axis labels

        plt.show()
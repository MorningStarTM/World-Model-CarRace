import torch
import os
from world_model.vae import ConvVAE
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F


class VAETrainer:
    def __init__(self, model:ConvVAE, batch_size:int, save_path: str, beta=2) -> None:
        self.model = model
        self.batch_size = batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.save_path = save_path
        self.best_loss = float('inf')
        self.beta = beta

    def vae_loss(self, reconstructed, original, mu, logvar):
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstructed, original, reduction='sum')
        
        # KL Divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
        return recon_loss + self.beta * kl_loss

    
    def train_vae(self, dataloader, epochs=20, learning_rate=1e-3):
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.model.to(self.device)
        self.model.train()

        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(dataloader):
                images = data[0] if isinstance(data, (list, tuple)) else data
                images = images.to(self.device)

                optimizer.zero_grad()
                reconstructed, mu, logvar = self.model(images)
                loss = self.vae_loss(reconstructed, images, mu, logvar)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            
            avg_loss = running_loss / self.batch_size
            
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(dataloader.dataset):.4f}")

            # Check if the model's loss has reduced
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.model.save(self.save_path)
                print(f"model saved at {self.save_path}")

            # Generate images at the end of each epoch
            self.generate_and_plot_images()



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
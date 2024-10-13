import torch
import os
from .vae import ConvVAE
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt


class VAETrainer:
    def __init__(self, model:ConvVAE, batch_size:int) -> None:
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=1e-3)
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size


    def loss_function(self, x, x_hat, mean, log_var):
        reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
        KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

        return reproduction_loss + KLD
    
    def train(self, train_loader, epochs):
        self.model.train()
        for epoch in range(epochs):
            overall_loss = 0
            for batch_idx, (x, _) in enumerate(train_loader):
                x = x.to(self.device)

                self.optimizer.zero_grad()

                x_hat, mean, log_var = self.model(x)
                loss = self.loss_function(x, x_hat, mean, log_var)
                
                overall_loss += loss.item()
                
                loss.backward()
                self.optimizer.step()

            print("\tEpoch", epoch + 1, "\tAverage Loss: ", overall_loss/(batch_idx*self.batch_size))
            self.generate_and_plot_images()
        return overall_loss

    
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
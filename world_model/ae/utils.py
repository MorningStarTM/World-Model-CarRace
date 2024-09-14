import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

class Utils:
    def __init__(self) -> None:
        pass

    def train_autoencoder(self, autoencoder, dataloader, num_epochs=10, learning_rate=0.001, device='cuda'):
        # Move model to the device (GPU or CPU)
        autoencoder = autoencoder.to(device)

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(num_epochs):
            autoencoder.train()
            running_loss = 0.0

            for images in dataloader:
                # Move images to the device
                images = images.to(device)

                # Forward pass
                outputs = autoencoder(images)
                loss = criterion(outputs, images)  # Compare outputs to original images

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Accumulate the loss
                running_loss += loss.item()

            # Print the average loss for this epoch
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

        print("Training completed.")


    def plot_result(self, original, reconstructed, n=8):
        """
        Plot n original images and their corresponding reconstructed images.
        
        :param original: Batch of original images (Tensor)
        :param reconstructed: Batch of reconstructed images (Tensor)
        :param n: Number of images to display
        """
        # Select the first n images
        original = original[:n].cpu().detach()
        reconstructed = reconstructed[:n].cpu().detach()

        # Set up the plot grid
        fig, axes = plt.subplots(2, n, figsize=(15, 5))

        for i in range(n):
            # Plot original images
            axes[0, i].imshow(original[i].permute(1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
            axes[0, i].axis('off')
            axes[0, i].set_title("Original")

            # Plot reconstructed images
            axes[1, i].imshow(reconstructed[i].permute(1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
            axes[1, i].axis('off')
            axes[1, i].set_title("Reconstructed")

        plt.show()
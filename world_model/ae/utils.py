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


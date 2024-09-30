import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .mdn_rnn import LSTM_MDN_Model

class MDNRNNTrainer:
    def __init__(self, latent_dim, action_dim, hidden_dim, num_gaussians, lr=0.001):
        """
        Initialize the MDN-RNN Trainer.
        
        Parameters:
        - latent_dim: Dimensionality of the latent space (output of VAE).
        - action_dim: Dimensionality of the action space.
        - hidden_dim: Hidden size of the LSTM.
        - num_gaussians: Number of Gaussians in the mixture.
        - lr: Learning rate for optimizer.
        """
        self.model = LSTM_MDN_Model(latent_dim, action_dim, hidden_dim, num_gaussians)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = self.mdn_loss
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)


    def train(self, train_loader, epochs=10):
        """
        Train the MDN-RNN model.
        
        Parameters:
        - train_loader: DataLoader providing batches of (latent_vector, action, next_latent_vector).
        - epochs: Number of training epochs.
        """
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for latent_vector, action, next_latent_vecctor in train_loader:
                latent_vector = latent_vector.to(self.device)
                action = action.to(self.device)
                next_latent_vecctor = next_latent_vecctor.to(self.device)

                # Initialize the hidden state
                batch_size = latent_vector.size(0)
                hidden_state = self.model.init_hidden(batch_size)
                hidden_state = (hidden_state[0].to(self.device), hidden_state[1].to(self.device))

                self.optimizer.zero_grad()
                pi, mu, sigma, hidden_state = self.model(latent_vector, action, hidden_state)

                loss = self.criterion(pi, mu, sigma, next_latent_vecctor)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader)}")


    def save_model(self, path):
        """
        Save the trained MDN-RNN model.
        
        Parameters:
        - path: File path to save the model.
        """
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    

    def load_model(self, path):
        """
        Load a pre-trained MDN-RNN model.
        
        Parameters:
        - path: File path to load the model.
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        print(f"Model loaded from {path}")
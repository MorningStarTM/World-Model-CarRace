import torch
import os
from vae import ConvVAE
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

    

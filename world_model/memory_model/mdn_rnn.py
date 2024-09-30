import torch
import torch.nn as nn
import numpy as np

class LSTM_MDN_Model(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim, num_gaussians):
        super(LSTM_MDN_Model, self).__init__()
        self.num_gaussians = num_gaussians
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        #LSTM layer
        self.lstm = nn.LSTM(latent_dim + action_dim, hidden_dim, self.num_gaussians)
        
        # MDN layer
        self.mdn_pi = nn.Linear(hidden_dim, self.num_gaussians)
        self.mdn_mu = nn.Linear(hidden_dim, self.num_gaussians * latent_dim)
        self.mdn_sigma = nn.Linear(hidden_dim, self.num_gaussians * latent_dim)

        #Activation Function for MDN
        self.softmax = nn.Softmax(dim=1)
        self.softplus = nn.Softplus()

    

    def forward(self, latent_vector, action, hidden_state):
        lstm_input = torch.cat([latent_vector, action], dim=-1).unsqueeze(1)

        lstm_out, hidden_state = self.lstm(lstm_input, hidden_state)

        pi = self.softmax(self.mdn_pi(lstm_out.squeeze(1)))
        mu = self.mdn_mu(lstm_out.squeeze(1))
        sigma = self.softplus(self.mdn_sigma(lstm_out.squeeze(1)))


        mu = mu.view(-1, self.num_gaussians, self.latent_dim)
        sigma = sigma.view(-1, self.num_gaussians, self.latent_dim)

        return pi, mu, sigma, hidden_state
    
    def init_hidden(self, batch_size):
        # Initialize hidden and cell states to zeros
        return (torch.zeros(1, batch_size, self.hidden_dim),
                torch.zeros(1, batch_size, self.hidden_dim))
    

    def mdn_loss(pi, mu, sigma, target):
        # Reshape target to match the shape of mu
        target = target.unsqueeze(1).expand_as(mu)
        
        # Compute Gaussian probabilities
        prob = (1.0 / torch.sqrt(2.0 * np.pi * sigma**2)) * torch.exp(-0.5 * ((target - mu) / sigma)**2)
        
        # Sum probabilities across all Gaussians
        prob = torch.sum(pi * prob, dim=1)
        
        # Compute negative log-likelihood
        nll = -torch.log(prob + 1e-8)  # Add small epsilon to prevent log(0)
        return torch.mean(nll)


import torch
import torch.nn as nn

class LSTM_MDN_Model(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim, num_gaussians):
        super(LSTM_MDN_Model, self).__init__()
        self.num_gaussians = num_gaussians
        self.latent_dim = latent_dim

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
    
    
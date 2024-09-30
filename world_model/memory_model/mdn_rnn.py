import torch
import torch.nn as nn

class LSTM_MDN_Model(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim, num_gaussians):
        super(LSTM_MDN_Model, self).__init__()
        
        #LSTM layer
        self.lstm = nn.LSTM(latent_dim + action_dim, hidden_dim, num_gaussians)

        # MDN layer
        self.mdn_pi = nn.Linear(hidden_dim, num_gaussians)
        self.mdn_mu = nn.Linear(hidden_dim, num_gaussians * latent_dim)
        self.mdn_sigma = nn.Linear(hidden_dim, num_gaussians * latent_dim)

        #Activation Function for MDN
        self.softmax = nn.Softmax(dim=1)
        self.softplus = nn.Softplus()

    
    
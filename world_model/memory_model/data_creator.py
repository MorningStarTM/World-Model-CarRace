import os
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

class MDNRNNDatasetCreator:
    def __init__(self, vae_encoder, image_size, latent_dim, output_folder):
        """
        Initialize the dataset creator for MDN-RNN training.
        
        Parameters:
        - vae_encoder: The pre-trained VAE encoder for compressing images.
        - image_size: The size of the input images (expected by the VAE).
        - latent_dim: Dimension of the latent space produced by the VAE encoder.
        - output_folder: Folder to save the processed dataset.
        """
        self.vae_encoder = vae_encoder
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.output_folder = output_folder
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])


    def load_image(self, image_path):
        """
        Load an image and apply the necessary transformations.
        """
        image = Image.open(image_path).convert('RGB')
        return self.transform(image).unsqueeze(0)  # Add batch dimension
    
    def compress_to_latent(self, image_tensor):
        """
        Pass an image through the VAE encoder to get the latent vector.
        """
        with torch.no_grad():
            latent_vector, _ = self.vae_encoder(image_tensor)
        return latent_vector.squeeze(0).cpu().numpy()  # Remove batch dimension


    def create_dataset(self, data_file):
        """
        Create a dataset with latent vectors, actions, and next latent vectors.
        
        Parameters:
        - data_file: CSV file containing the original dataset with 'obs', 'action', and 'next_obs' columns.
        
        Returns:
        - DataFrame with latent_space, action, and next_latent_space columns.
        """
        data = pd.read_csv(data_file)

        latent_vectors = []
        actions = []
        next_latent_vectors = []

        for index, row in data.iterrows():
            # Load and compress the observation image
            obs_path = os.path.join(self.output_folder, row['obs'])
            obs_image = self.load_image(obs_path)
            latent_vector = self.compress_to_latent(obs_image)

            # Load and compress the next observation image
            next_obs_path = os.path.join(self.output_folder, row['next_obs'])
            next_obs_image = self.load_image(next_obs_path)
            next_latent_vector = self.compress_to_latent(next_obs_image)

            # Collect latent vectors and corresponding action
            latent_vectors.append(latent_vector)
            actions.append(row['action'])  # Assuming actions are stored as integers/floats
            next_latent_vectors.append(next_latent_vector)

        # Create a DataFrame with the latent vectors, actions, and next latent vectors
        dataset = pd.DataFrame({
            'latent_space': latent_vectors,
            'action': actions,
            'next_latent_space': next_latent_vectors
        })

        # Save the DataFrame to a CSV or pickle file
        dataset.to_pickle(os.path.join(self.output_folder, 'mdn_rnn_dataset.pkl'))
        print(f"Dataset saved to {self.output_folder}/mdn_rnn_dataset.pkl")
        
        return dataset
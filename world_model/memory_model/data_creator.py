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

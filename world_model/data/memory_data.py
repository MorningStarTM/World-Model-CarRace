import torch
import pandas as pd
import cv2
import os

class MemoryData:
    def __init__(self, model, image_folder, csv_file, device='cpu'):
        """
        Initialize the DataProcessor with the VAE model, image folder, and CSV file.
        
        Args:
            model: The pre-trained VAE model.
            image_folder: The root folder where images are stored.
            csv_file: The path to the CSV file containing the image names and actions.
            latent_dim: The dimensionality of the latent space of the model.
            device: The device to run the model on ('cpu' or 'cuda').
        """
        self.model = model.to(device)
        self.model.eval()  # Set the model to evaluation mode
        self.image_folder = image_folder
        self.csv_file = csv_file
        self.device = device


    def load_data(self):
        """
        Load the dataset CSV file into a DataFrame.
        """
        return pd.read_csv(self.csv_file)
    
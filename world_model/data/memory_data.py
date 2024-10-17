import torch
import pandas as pd
import cv2
import os
from world_model.ae.autoencoder import Autoencoder
from world_model.vae.vae import ConvVAE

class MemoryData:
    def __init__(self, model, image_folder, csv_file, device='cpu'):
        """
        Initialize the DataProcessor with the VAE model, image folder, and CSV file.
        
        Args:
            model: The model type.
            image_folder: The root folder where images are stored.
            csv_file: The path to the CSV file containing the image names and actions.
            latent_dim: The dimensionality of the latent space of the model.
            device: The device to run the model on ('cpu' or 'cuda').
        """
        
        self.model = model
        self.model.to(device)

        self.model.eval()  # Set the model to evaluation mode
        self.image_folder = image_folder
        self.csv_file = csv_file
        self.device = device


    def load_data(self):
        """
        Load the dataset CSV file into a DataFrame.
        """
        return pd.read_csv(self.csv_file)
    

    def encode_observation(self, img_path):
        """
        Encode an image to its latent representation using the VAE model.
        
        Args:
            img_path: The path to the image file.
        
        Returns:
            The latent vector as a numpy array.
        """
        image = cv2.imread(img_path)
        image = cv2.resize(image, (96, 96))
        img_tensor = torch.from_numpy(image).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)  # Change shape to (C, H, W)
        img_batch = img_tensor.unsqueeze(0).to(self.device)  # Add batch dimension

        with torch.no_grad():
            latent = self.model.encoder(img_batch)
        
        return latent.squeeze().cpu().numpy()
    

    def process_and_save(self, output_csv):
        """
        Process the images and save the encoded data into a new CSV file.
        
        Args:
            output_csv: The path where the encoded CSV file will be saved.
        """
        df = self.load_data()
        encoded_data = []

        for idx, row in df.iterrows():
            current_img_name = row['image']
            next_img_name = row['next_image']
            action = row['action']

            # Construct full paths for the current and next images
            current_img_path = os.path.join(self.image_folder, current_img_name)
            next_img_path = os.path.join(self.image_folder, next_img_name)

            # Encode the images to latent space
            encoded_current = self.encode_observation(current_img_path)
            encoded_next = self.encode_observation(next_img_path)

            # Add the encoded data to the list
            encoded_data.append([encoded_current.tolist(), action, encoded_next.tolist()])

        # Convert the encoded data into a DataFrame
        encoded_df = pd.DataFrame(encoded_data, columns=['encoded_current', 'action', 'encoded_next'])

        # Save the DataFrame to a CSV file
        encoded_df.to_csv(output_csv, index=False)

        print(f"Encoded dataset saved successfully to {output_csv}")
    
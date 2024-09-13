import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os


# Custom dataset class for loading images
class ImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.image_files = [os.path.join(image_folder, f) for f in sorted(os.listdir(image_folder)) if f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image
    

# Define image transformations (resize and convert to tensor)
transform = transforms.Compose([
    transforms.Resize((96, 96)),  # Resize images to (96, 96)
    transforms.ToTensor()         # Convert images to PyTorch tensors
])
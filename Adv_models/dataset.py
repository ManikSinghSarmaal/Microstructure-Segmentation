import torch
from torchvision.transforms import ToTensor
from PIL import Image
import os
class ImageDataset(Dataset):
    def __init__(self, image_dir):
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
        self.transform = ToTensor()  # Define the transformation to apply

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        image = image.convert('RGB')  # Convert to RGB if necessary
        
        # Apply the transformation (convert to tensor)
        image = self.transform(image)
        
        return image
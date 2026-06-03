"""
Phase 2: Data Pipeline

This script handles loading the Kaggle dataset, preprocessing the images, 
and setting up PyTorch DataLoaders for training and validation.

Why do this here? 
Keeping data loading separate from the model definition (model.py) and 
training loop (train.py) makes the codebase much cleaner and easier to maintain.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class KaggleMNISTDataset(Dataset):
    """
    A custom PyTorch Dataset to load our Kaggle CSV data.
    PyTorch requires datasets to implement __len__ (how many items) 
    and __getitem__ (how to get a single item by index).
    """
    def __init__(self, csv_path):
        # We load the entire CSV into memory (it's small enough for MNIST)
        self.data_frame = pd.read_csv(csv_path)
        
        # The first column is the label (0-9)
        self.labels = self.data_frame.iloc[:, 0].values
        
        # The remaining 784 columns are the pixel values (28x28)
        self.pixels = self.data_frame.iloc[:, 1:].values
        
    def __len__(self):
        # Returns the total number of images in this dataset
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        # 1. Get the label for this specific index
        label = self.labels[idx]
        
        # 2. Get the 784 pixel values for this image
        pixel_row = self.pixels[idx]
        
        # 3. Normalise the pixels from [0, 255] to [0.0, 1.0]
        # This helps the neural network learn faster and more stably
        pixel_row = pixel_row / 255.0
        
        # 4. Standard MNIST normalisation
        # We subtract the mean and divide by standard deviation of the MNIST dataset.
        # This centres the data around 0, making gradients well-behaved during training.
        mean = 0.1307
        std = 0.3081
        pixel_row = (pixel_row - mean) / std
        
        # 5. Reshape and convert to PyTorch Tensor
        # We reshape the flat 784 array into (1, 28, 28)
        # 1 = channel (grayscale), 28 = height, 28 = width
        # PyTorch expects the channel dimension first!
        pixel_tensor = torch.tensor(pixel_row, dtype=torch.float32).reshape(1, 28, 28)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return pixel_tensor, label_tensor

def get_dataloaders(csv_path="../data/digi_rec_train.csv", batch_size=64):
    """
    Loads the dataset, splits it into training (80%) and validation (20%),
    and returns DataLoaders for both.
    """
    # 1. Load the full dataset
    full_dataset = KaggleMNISTDataset(csv_path)
    
    # 2. Calculate sizes for an 80/20 split
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    # 3. Split the dataset randomly, but with a fixed seed (42)
    # The fixed seed ensures that every time we run this code, we get the EXACT SAME split.
    # This reproducibility is crucial for debugging and comparing model changes.
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size], 
        generator=generator
    )
    
    # 4. Create DataLoaders
    # A DataLoader groups the individual images into "batches" of 64.
    # It also handles shuffling (randomising the order) which is important for training.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def get_train_loader(csv_path="../data/digi_rec_train.csv", batch_size=64):
    train_loader, _ = get_dataloaders(csv_path, batch_size)
    return train_loader

def get_val_loader(csv_path="../data/digi_rec_train.csv", batch_size=64):
    _, val_loader = get_dataloaders(csv_path, batch_size)
    return val_loader

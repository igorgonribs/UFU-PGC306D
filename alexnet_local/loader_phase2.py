#!/usr/bin/env python3

"""A simple python script template.
"""

import os
import sys
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
import os

save_dir = './aloader'
os.makedirs(save_dir, exist_ok=True)

class SavedTensorDataset(Dataset):
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.image_files = sorted([f for f in os.listdir(save_dir) if f.startswith('images_batch_')])
        self.label_files = sorted([f for f in os.listdir(save_dir) if f.startswith('labels_batch_')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        images = torch.load(os.path.join(self.save_dir, self.image_files[idx]))
        labels = torch.load(os.path.join(self.save_dir, self.label_files[idx]))
        return images, labels

# Load the saved data using the custom dataset
loaded_dataset = SavedTensorDataset(save_dir)
loaded_dataloader = DataLoader(loaded_dataset, batch_size=1, shuffle=False) # Batch size 1 as we saved batches

# Example of accessing loaded data
for images, labels in loaded_dataloader:
    print(f"Loaded images shape: {images.shape}, Loaded labels shape: {labels.shape}")
    break # Just show one batch





sys.exit(0)

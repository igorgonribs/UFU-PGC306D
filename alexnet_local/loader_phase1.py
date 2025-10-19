#!/usr/bin/env python3

"""A simple python script template.
"""

import os
import sys
import argparse


import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# Define your transformations
transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load your dataset using ImageFolder
data_root = './GTSRB_novo/Final_Training/Images/' # Replace with your dataset root directory
dataset = datasets.ImageFolder(root=data_root, transform=transform)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

# Create a directory to save the processed data
save_dir = './aloader'
os.makedirs(save_dir, exist_ok=True)

# Iterate and save each batch
for i, (images, labels) in enumerate(dataloader):
    torch.save(images, os.path.join(save_dir, f'images_batch_{i}.pt'))
    torch.save(labels, os.path.join(save_dir, f'labels_batch_{i}.pt'))

print(f"Processed data saved to {save_dir}")


sys.exit(0)

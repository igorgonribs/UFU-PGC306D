#!/usr/bin/env python3

"""dataloaders_gen.py

"""

import os
import sys
import glob
import yaml
import datetime
import argparse

# Own modules (local sources)
sys.path.append("../../")
sys.path.append("./")
from lib.config import *
from lib.decor import *

import torch
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
import numpy as np

# https://discuss.pytorch.org/t/why-do-we-need-subsets-at-all/49391/7
# https://discuss.pytorch.org/t/using-imagefolder-random-split-with-multiple-transforms/79899/2
# adapted from ptrblck post
class MyLazyDataset(Dataset):
  def __init__(self, dataset, transform=None):
    self.dataset = dataset
    self.transform = transform

  def __getitem__(self, index):
    if self.transform:
      x = self.transform(dataset[index][0])
    else:
      x = dataset[index][0]
      y = dataset[index][1]
    return x, y

  def __len__(self):
    return len(dataset)

dataloaders_dir = './dataloaders/'

# -= init =-
banner(f"Program -> {os.path.basename(__file__)}")

data_transforms = {
  # dados de treinamento
  'train': transforms.Compose([
    transforms.Resize((227,227)), # Pré-processamento para formatar as imagens em tamanho (30x30)
    transforms.ToTensor(), # Transforma no tipo de dado do pytorch
    ]),

  # Podemos fazer cortes diferentes, mas a imagem final tem que ser 30x30
  # dados de validação
  'val': transforms.Compose([
    transforms.Resize((227,227)), # Pré-processamento para formatar as imagens em tamanho (30x30)
    transforms.ToTensor(),  # Transforma no tipo de dado do pytorch
    ]),
}

# Carrega dataset completo
dataset = datasets.ImageFolder(root='GTSRB_novo/Final_Training/Images', transform=data_transforms['train'])

# Define split ratios
train_ratio = 0.7
val_ratio = 0.20
test_ratio = 0.10

# Calculate lengths for each split
total_size = len(dataset)
train_size = int(train_ratio * total_size)
val_size = int(val_ratio * total_size)
test_size = total_size - train_size - val_size # Ensure all samples are accounted for

# Perform the split
train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42) # For reproducibility
)

# Create DataLoaders for each split
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Dataset Sumary:")
print(f"Total:      {len(dataset):>5}")
print(f"Train size: {len(train_dataset):>5} ({ (len(train_dataset) * 100)/len(dataset):>5.2f}%)")
print(f"Val   size: {len(val_dataset):>5} ({ (len(val_dataset) * 100)/len(dataset):>5.2f}%)")
print(f"Test  size: {len(test_dataset):>5} ({ (len(test_dataset) * 100)/len(dataset):>5.2f}%)")
print(f'Class names: {dataset.classes}')
print("")


# Accessing the shape of a single sample within a batch
for batch_data, _ in train_loader:
  print(f"Shape of a single sample within the batch: {batch_data[0].shape}")
  break



print("Saving the DataLoaders...")
# for dataloader_name, dataloader_object  in dataloaders.items():
#   print(f"  {dataloader_name}")
#   file_to_save = dataloaders_dir+dataloader_name+'_dataloader.pt'
#   print(file_to_save)
#   torch.save(dataloader_object, file_to_save)

#torch.save(dataset, dataloaders_dir+'all'+'_dataloader.pt')
#torch.save(train_loader, dataloaders_dir+'train'+'_dataloader.pt')
#torch.save(val_loader, dataloaders_dir+'val'+'_dataloader.pt')
#torch.save(test_loader, dataloaders_dir+'test'+'_dataloader.pt')


# # Convert to Tensors
# all_images = []
# all_labels = []
# for image, label in dataset:
#   all_images.append(image)
#   all_labels.append(label)
#
# images_tensor = torch.stack(all_images)
# labels_tensor = torch.tensor(all_labels)
#
# # Create a TensorDataset
# tensor_dataset = TensorDataset(images_tensor, labels_tensor)
#
# # Save the TensorDataset
# torch.save(tensor_dataset, dataloaders_dir+'tensor_dataset.pth')


# ----
# Convert to Tensors
print("Saving train_loader")
all_images = []
all_labels = []
for image, label in train_loader:
  all_images.append(image)
  all_labels.append(label)

images_tensor = torch.stack(all_images)
labels_tensor = torch.tensor(all_labels)

# Create a TensorDataset
tensor_dataset = TensorDataset(images_tensor, labels_tensor)

# Save the TensorDataset
torch.save(tensor_dataset, dataloaders_dir+'train'+'_dataloader.pth')

print("Saving val_loader")
all_images = []
all_labels = []
for image, label in val_loader:
  all_images.append(image)
  all_labels.append(label)

images_tensor = torch.stack(all_images)
labels_tensor = torch.tensor(all_labels)

# Create a TensorDataset
tensor_dataset = TensorDataset(images_tensor, labels_tensor)

# Save the TensorDataset
torch.save(tensor_dataset, dataloaders_dir+'val'+'_dataloader.pth')

print("Saving test_loader")
all_images = []
all_labels = []
for image, label in test_loader:
  all_images.append(image)
  all_labels.append(label)

images_tensor = torch.stack(all_images)
labels_tensor = torch.tensor(all_labels)

# Create a TensorDataset
tensor_dataset = TensorDataset(images_tensor, labels_tensor)

# Save the TensorDataset
torch.save(tensor_dataset, dataloaders_dir+'test'+'_dataloader.pth')


sys.exit(0)

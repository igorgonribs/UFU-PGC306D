#!/usr/bin/env python3

"""dataloaders_gen.py

"""

import os
import sys
import glob
import yaml
import argparse

#sys.path.append("../../")
from configs.config import *

import torch
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split

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

# Proporções de 80%, 15% e 5% respectivamente
train_ratio, val_ratio, test_ratio = (0.80, 0.18, 0.02)

#train_size = int(train_ratio * len(dataset))
#val_size = int(val_ratio * len(dataset))
#test_size = int(test_ratio * len(dataset))

train_size = int(len(dataset) * train_ratio)
val_size = int(len(dataset) * val_ratio)
test_size = len(dataset) - train_size - val_size

# Divide de forma aleatória
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Cria dataloaders
train_loader = DataLoader(train_dataset.dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset.dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset.dataset, batch_size=64, shuffle=True)

dataset_sizes = {'train': train_size, 'val': val_size, 'test': test_size}
dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

print("Dataset Sumary:")
print(f"Total:      {len(dataset):>5}")
print(f"train_size: {train_size:>5} ({ (train_size * 100)/len(dataset):>5.2f}%)")
print(f"val_size:   {val_size:>5} ({ (val_size * 100)/len(dataset):>5.2f}%)")
print(f"test_size:  {test_size:>5} ({ (test_size * 100)/len(dataset):>5.2f}%)")
#print(f'Dataset sizes: {dataset_sizes}')
print(f'class names: {dataset.classes}')
print("")


print("Saving the DataLoaders...")
dataloaders_dir = './dataloaders/'

for dataloader_name, dataloader_object  in dataloaders.items():
  print(f"  {dataloader_name}")
  file_to_save = dataloaders_dir+dataloader_name+'_dataloader.pt'
  print(file_to_save)
  torch.save(dataloader_object, file_to_save)


sys.exit(0)

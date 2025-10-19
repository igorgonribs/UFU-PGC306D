#!/usr/bin/env python3

"""alexnet_train.py

"""

import os
import sys
import glob
import yaml
import argparse
import copy
import shutil
import random

# Own modules (local sources)
sys.path.append("../../")
sys.path.append("./")
from lib.config import *
from lib.decor import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from alexnet_define import AlexNet

import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from torch.optim import lr_scheduler
from torchmetrics.classification import F1Score, Recall, BinaryF1Score, MulticlassF1Score, MultilabelF1Score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


from torch import optim
import time
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from imageio import imread
from PIL import Image
from skimage.transform import resize
from skimage.util import crop

#===========================
def imshow(inp, title=None):
  """Imshow for Tensor."""

  # Reorganizar o dado para o formato adequado do matplotlib
  inp = inp.numpy().transpose((1, 2, 0))
  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])
  inp = std * inp + mean
  inp = np.clip(inp, 0, 1) # limita os dados da imagem entre [0,1]
  # plota a imagem
  plt.imshow(inp)
  if title is not None:
    plt.title(title)
  plt.pause(0.01)  # pause a bit so that plots are updated

def imshow2(inp, title=None):
  """Imshow for Tensor."""

  # Reorganizar o dado para o formato adequado do matplotlib
  inp = inp.numpy().transpose((1, 2, 0))
  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])
  inp = std * inp + mean
  inp = np.clip(inp, 0, 1) # limita os dados da imagem entre [0,1]
  # plota a imagem
  plt.imshow(inp)
  if title is not None:
    plt.title(title)
  plt.pause(0.01)  # pause a bit so that plots are updated
  plt.show()

# Apresenta a dimensão de saída da imagem após a convolução (qual o tamanho do feature map?)
def out_conv2d(dim_input, kernel_size, padding=0, dilation=1, stride=1):
  dim_output = ((dim_input + 2 * padding - dilation * (kernel_size-1) - 1)/stride) + 1
  return dim_output


def train_model(model, criterion, optimizer, dataloaders, device, num_epochs=10):
  # Históricos de perda e acurácia
  history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
  }

  since = time.time()
  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0

  for epoch in range(num_epochs):
    print('Epoch \033[0;32m{}\033[0m/{}'.format(epoch +1, num_epochs))
    print('-' * 12)
    sinceepoch = time.time()

    for phase in ['train', 'val']:
      if phase == 'train':
        model.train()
      else:
        model.eval()

      running_loss = 0.0
      running_corrects = 0
      num_samples = 0   # contador real de exemplos

      for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
          outputs = model(inputs)
          _, preds = torch.max(outputs, 1)
          loss = criterion(outputs, labels)

          if phase == 'train':
            loss.backward()
            optimizer.step()

        # acumula estatísticas
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()
        num_samples += labels.size(0)

      # calcula médias com base no número real de amostras
      epoch_loss = running_loss / num_samples
      epoch_acc = running_corrects / num_samples

      # Salva no histórico
      if phase == 'train':
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)
      else:
        history["val_loss"].append(epoch_loss)
        history["val_acc"].append(epoch_acc)

      print(f"{phase:>6} Loss: {epoch_loss:>6.4f} Acc: \033[1;32m{epoch_acc:>6.4f}\033[0m")

      if phase == 'val' and epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())

    # verifica o tempo de treinamento desta epoca
    time_elapsedepoch = time.time() - sinceepoch
    print('       Epoch training complete in {:.0f}m {:.0f}s'.format(
          time_elapsedepoch // 60, time_elapsedepoch % 60))
    print("")

  time_elapsed = time.time() - since
  print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
  print('Best val Acc: \033[0;32m{:4f}\033[0m'.format(best_acc))

  model.load_state_dict(best_model_wts)
  return model, history

def plot_training(history):
  epochs = range(1, len(history["train_loss"]) + 1)

  plt.figure(figsize=(12, 5))

  # Plot Loss
  plt.subplot(1, 2, 1)
  plt.plot(epochs, history["train_loss"], label="Treino")
  plt.plot(epochs, history["val_loss"], label="Validação")
  plt.xlabel("Época")
  plt.ylabel("Loss")
  plt.title("Evolução da Loss")
  plt.legend()
  plt.grid(True)

  # Plot Accuracy em %
  plt.subplot(1, 2, 2)
  plt.plot(epochs, [a * 100 for a in history["train_acc"]], label="Treino")
  plt.plot(epochs, [a * 100 for a in history["val_acc"]], label="Validação")
  plt.xlabel("Época")
  plt.ylabel("Acurácia (%)")
  plt.title("Evolução da Acurácia")
  plt.legend()
  plt.grid(True)

  filename='./historico_'+str(args.model_name)+'_'+str(epocas)+'epocas'+'.png'
  plt.savefig(filename, dpi=150)
  plt.show()


  GRAPH_DIR='./'

  plt.figure(figsize=(8,5))
  plt.plot(epochs, history["train_loss"], label="Train Loss")
  plt.plot(epochs, history["val_loss"],   label="Val Loss")
  plt.title("Curva de Loss - Melhor Configuração")
  plt.xlabel("Época"); plt.ylabel("Loss"); plt.grid(True); plt.legend()
  plt.tight_layout()
  filename=GRAPH_DIR+'/experimento_CNN_AlexNet_curvas_loss.png'
  plt.savefig(filename, dpi=150)
  plt.show()

  plt.figure(figsize=(8,5))
  plt.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train Acc")
  plt.plot(epochs, [a * 100 for a in history["val_acc"]],   label="Val Acc")
  plt.title("Curva de Acurácia - Melhor Configuração")
  plt.xlabel("Época"); plt.ylabel("Acurácia"); plt.grid(True); plt.legend()
  plt.tight_layout()
  filename=GRAPH_DIR+'/experimento_CNN_AlexNet_curvas_acuracia.png'
  plt.savefig(filename, dpi=150)
  plt.show()













# Função para visualizar os dados de validação por batch (número de imagens = 6)
def visualize_model(model, num_images=6):
  # Salva se o modelo estava em treino ou eval
  was_training = model.training
  model.eval()

  fig = plt.figure(figsize=(10,10))
  images_so_far = 0

  # Sorteia índices aleatórios do dataset de validação
  test_dataset_vis = dataloaders['test'].dataset
  indices = random.sample(range(len(test_dataset_vis)), num_images)

  with torch.no_grad():
    for idx in indices:
      # Pega uma imagem e label aleatórios
      img, label = test_dataset_vis[idx]

      # Prepara entrada para o modelo (adiciona dimensão do batch)
      inputs = img.unsqueeze(0).to(device)
      label = torch.tensor(label).to(device)

      # Passa pelo modelo
      outputs = model(inputs)
      _, pred = torch.max(outputs, 1)

      # Plota imagem com predição e rótulo real
      images_so_far += 1
      ax = plt.subplot(num_images//2, 2, images_so_far)
      ax.axis("off")
      ax.set_title(f"pred: {class_names[pred.item()]}\ntrue: {class_names[label.item()]}")
      imshow(img)

      if images_so_far == num_images:
        break

  plt.tight_layout()
  filename='./predict_'+str(epocas)+'epocas'+'.png'
  plt.savefig(filename, dpi=150)
  plt.show()
  # Res

def eval_test_set(model):
  # Salva se o modelo estava em treino ou eval
  was_training = model.training
  model.eval()

  # Sorteia índices aleatórios do dataset de validação
  test_dataset_eval = dataloaders['test'].dataset
  indices = random.sample(range(len(test_dataset_eval)), test_size)

  print(f" len indices {len(indices)}")
  print(f"indices {indices}")


  recall_metric = Recall(task="multiclass", num_classes=args.num_classes, average='macro')

  f1_metric = MulticlassF1Score(num_classes=args.num_classes)
  #f1_metric = MultilabelF1Score(num_labels=args.num_classes)

  with torch.no_grad():
    total_loss = 0
    correct_predictions = 0
    for idx in indices:
      # Pega uma imagem e label aleatórios
      img, label = test_dataset_eval[idx]

      # Prepara entrada para o modelo (adiciona dimensão do batch)
      inputs = img.unsqueeze(0).to(device)
      label = torch.tensor(label).to(device)

      # Passa pelo modelo
      outputs = model(inputs)

      #criterion = nn.CrossEntropyLoss(reduction='mean')
      #loss = criterion(outputs, label)
      #total_loss += loss.item()

      _, pred = torch.max(outputs, 1)

      correct_predictions += (pred == label).sum().item()
      recall_metric.update(pred, label)
      f1_metric.update(pred, label)

      #print(f"types: {type(label)} === {type(pred)}  ")
      if pred == label:
        print(f"\033[0;32mpred: {class_names[pred.item()]}\ntrue: {class_names[label.item()]}\033[0m")
      else:
        print(f"\033[0;31mpred: {class_names[pred.item()]}\ntrue: {class_names[label.item()]}\033[0m")
      print("")


    #avg_loss = total_loss / len(test_loader)
    avg_loss = 0.11111
    accuracy = correct_predictions / len(test_dataset)
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")

    recall_score = recall_metric.compute()
    print(f"Recall (macro average): {recall_score}")

    f1_score = f1_metric.compute()
    print(f"F1-score: {f1_score}")



def eval_test_set2(model):
  model.eval() # Set model to evaluation mode

  all_preds = []
  all_labels = []

  with torch.no_grad(): # Disable gradient calculation
    for inputs, labels in test_loader:
      outputs = model(inputs)
      probabilities = torch.nn.functional.softmax(outputs, dim=1)
      predicted_labels = torch.argmax(probabilities, dim=1)

      all_preds.extend(predicted_labels.cpu().numpy())
      all_labels.extend(labels.cpu().numpy())

  # Calculate metrics
  accuracy = accuracy_score(all_labels, all_preds)
  precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
  conf_matrix = confusion_matrix(all_labels, all_preds)

  print(f"Accuracy: {accuracy:.4f}")
  print(f"Precision (weighted): {precision:.4f}")
  print(f"Recall (weighted): {recall:.4f}")
  print(f"F1-Score (weighted): {f1_score:.4f}")
  print("Confusion Matrix:\n", conf_matrix)







def predict1(model, image_to_pred):
  print("Pedicting 1 single image")
  # Salva se o modelo estava em treino ou eval
  was_training = model.training
  model.eval()

  imagem = image_to_pred.to(device)

  inputs = imagem.unsqueeze(0).to(device)
  outputs = model(inputs)
  _, pred = torch.max(outputs, 1)

  print(f"\033[0;35mPredição: \033[1;36m{class_names[pred.item()]}\033[0m")

  # Convert the tensor to channel-last format
  image_np = image_to_pred.permute(1, 2, 0).numpy()

  # Display the image
  plt.imshow(image_np)
  #plt.colorbar()
  texto = 'Placa detectada: '+class_names[pred.item()]
  plt.title(texto)
  #plt.axis('off')  # Turn off axis labels
  plt.show()


#===========================
if __name__ == "__main__":
  # -= init =-
  banner(f"Program -> {os.path.basename(__file__)}")

  args = parse_args()
  #===========================

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
  dataset = datasets.ImageFolder(root=args.imagefolder_dir, transform=data_transforms['train'])

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

  class_names = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons"
  }

  # Get a batch of training data
  inputs, classes = next(iter(dataloaders['train']))

  # Make a grid from batch
  out = torchvision.utils.make_grid(inputs)

  # Plota o batch de imagens (correspondendo a imagem e o rótulo)
  #imshow2(out, title=[class_names[x] for x in classes])

  #model = AlexNet(num_classes=args.num_classes)
  print(f"Carregando o modelo salvo no disco...")
  nomedomodelo = args.model_out_dir+args.model_name+'_clean.pth'
  net = torch.load(nomedomodelo, weights_only=False)
  summary(net)

  print("Ambiente de execução:")
  # Utiliza GPU ou CPU, verificar no colab (Ambiente de execução -> Alterar o tipo de ambiente de execução)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f"Pytorch version: {torch.__version__}")
  print(f"Device used: {device}")
  print(f"Current GPU device name: {torch.cuda.get_device_name(0)}") # Assuming at least one GPU
  print("")

  # testando as saidas da camadas conv2d
  # Use esta opção para clacular corretamente as dimensões

  # Insira a dimensão de entrada (111x111) e o tamanho do filtro (3x3)
  # Gera como resultado o tamanho do feature map
  print("tamanho do feature map:")
  print(out_conv2d(227, 3))
  print("")

  # Acesso aos pesos da camada especificada pelo nome (primeira camada convolucional)
  print("pesos da primeira camada convolucional:")
  #print(net.conv1.weight)
  print("")

  # Acesso ao bias da camada especificada pelo nome (primeira camada convolucional)
  print("bias da primeira camada convolucional:")
  #print(net.conv1.bias)
  print("")

  # Definindo os parâmetros importantes do treinamento
  # Função de custo e função de otimização dos parâmetros
  #criterion = nn.CrossEntropyLoss() # define o critério do erro (função de perda eh entropia)
  criterion = nn.CrossEntropyLoss(reduction='mean')
  optimizer = optim.SGD(net.parameters(), lr=0.01) # define a taxa de aprendizado e o otimizador SGD (Stochastic gradient descent)
  #optimizer = optim.Adam(net.parameters(), lr=0.001)

  # Colocar a rede na GPU
  net.to(device)

  epocas = args.max_epochs
  epocas = 40

  treinamento = 1
  if treinamento:
    # Treinamento
    print("          Treinando modelo...")
    trained_model, history = train_model(
      model=net,
      criterion=criterion,
      optimizer=optimizer,
      dataloaders=dataloaders,
      device=device,
      num_epochs=epocas
    )

    # Plotagem
    plot_training(history)

    # Salvando modelo treinado
    print("Salvando o modelo treinado no disco...")
    nomedomodelo = args.model_out_dir+args.model_name+'_'+str(epocas)+'epocas'+'.pth'
    #torch.save(net.state_dict(), nomedomodelo)
    torch.save(net, nomedomodelo)


  avaliacao = 1
  if avaliacao:
    print(f"Carregando o modelo salvo no disco...")
    nomedomodelo = args.model_out_dir+args.model_name+'_'+str(epocas)+'epocas'+'.pth'
    net = torch.load(nomedomodelo, weights_only=False)
    net.to(device)

    # Visualizar a predição de alguns dados de teste (6 imagens)
    print("          Visualizando predição...")
    visualize_model(net, num_images=6)

    print(f"\033[0;31mFim da rede otimizada\033[0m")
    print("")

    # Lendo nova imagem e fazendo a predição:
    nova_imagem = args.root_dir+'stop.jpg'
    new_img = Image.open(nova_imagem)
    new_img = np.array(new_img, dtype=np.uint8)
    new_img = resize(new_img,
                  output_shape=(227, 227),
                  mode='reflect', anti_aliasing=True)
    new_img = (new_img * 255).astype(np.uint8)

    to_tensor = transforms.ToTensor()
    new_img_tensor = to_tensor(new_img)

    #predict1(trained_model, new_img_tensor)

    # Cálculo das Métricas:
    eval_test_set2(net)


  print(f"\033[0;31mFim do programa\033[0m")

  #===========================
  sys.exit(0)

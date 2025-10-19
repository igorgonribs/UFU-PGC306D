#!/usr/bin/env python3

"""PGC306D - Tóp. Esp. em Inteligência Artificial 2 - Análise de Imagens e Vídeo
Grupo:
Augusto Carvalho
Igor Gonçalves
João Eloy
"""

"""Aplicando tarefas de pre-processamento
  * Balanceamento de classes
  * Recorte da regiao de interesse
  * Redimencionamento
"""

# Importação das bibliotecas
import os
import sys
import glob
import yaml
import shutil
import datetime
import argparse

# Own modules (local sources)
sys.path.append("../../")
sys.path.append("./")
from lib.config import *
from lib.decor import *

#---
import urllib3
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from imageio import imread
from PIL import Image
from skimage.transform import resize
from skimage.util import crop
import cv2

# -= init =-
banner(f"Program -> {os.path.basename(__file__)}")

# baixando base de dados de placas de transito
print("baixando base de dados de placas de transito... ", end='', flush=True)
if not os.path.exists("GTSRB_Final_Training_Images.zip"):
  url = ("https://sid.erda.dk/public/archives/"
  + "daaeac0d7ce1152aea9b61d9f1e19370/"
  + "GTSRB_Final_Training_Images.zip")
  filename = "./GTSRB_Final_Training_Images.zip"
  http = urllib3.PoolManager()
  with open(filename, 'wb') as out:
    r = http.request('GET', url, preload_content=False)
    shutil.copyfileobj(r, out)
print("pronto")


# Lendo os arquivos descritores das classes
print("lendo o arquivo compactado e extraindo informações de classes...")
archive = zipfile.ZipFile('GTSRB_Final_Training_Images.zip', 'r')
class_describe_files = [file for file in archive.namelist() if '.csv' in file]

# Gera os dataframes para o balanceamento
all_infos = pd.DataFrame()
all_infos_balance = pd.DataFrame()
# Passa por todas as classes
for class_describe_file in class_describe_files:
  with archive.open(class_describe_file) as class_file:
    newdf = pd.read_csv(class_file, sep=';', engine='python')
    #print(f"Tamanho da classe {class_describe_file.split('/')[-1].split('.')[0]}: {len(newdf):>4} ")
    print(f"Tamanho da classe {class_describe_file}: {len(newdf):>4} ")

    all_infos = pd.concat([all_infos, newdf])

    # Caso de classes demasiadamente grandes
    if len(newdf) > 500:
      # Mistura duas vezes para garantir aleatoriedade (random_state para reprodutibilidade)
      newdf_shuffled = newdf.sample(frac=1, random_state=42).reset_index(drop=True)
      newdf_shuffled2 = newdf_shuffled.sample(frac=1, random_state=89).reset_index(drop=True)

      # Sorteia um novo tamanho para as classes grandes
      random_number = random.randint(380, 620)
      fixed_number = 500
      newdf = newdf_shuffled2.iloc[:random_number]
      all_infos_balance = pd.concat([all_infos_balance, newdf])
    else:
      all_infos_balance = pd.concat([all_infos_balance, newdf])




# Gera um gráfico para comparação do balanceada de classes
fig, axs = plt.subplots(1, 2, figsize=(14, 6), dpi=150) # 1 row, 2 columns

axs[0].hist(all_infos['ClassId'], bins=43, rwidth=0.8, color='red', edgecolor='black')
axs[0].set_title('Distribuição original: '+str(len(all_infos))+' samples')
axs[0].set_xlabel('Classes')
axs[0].set_ylabel('Frequencia')
axs[0].set_ylim(0, 2500)
axs[0].grid(axis='y', linestyle='--', alpha=0.7)

axs[1].hist(all_infos_balance['ClassId'], bins=43, rwidth=0.8, color='blue', edgecolor='black')
axs[1].set_title('Distribuição Balanceada: '+str(len(all_infos_balance))+' samples')
axs[1].set_xlabel('Classes')
axs[1].set_ylabel('Frequencia')
axs[1].set_ylim(0, 2500)
axs[1].grid(axis='y', linestyle='--', alpha=0.7)

plt.suptitle("Comparação Balanceamento de classes")

plt.tight_layout()
filenamefig='./Histograma_classes_comp.png'
plt.savefig(filenamefig, dpi=150)
plt.show()


#print(all_infos_balance)


file_paths = [file for file in archive.namelist() if '.ppm' in file]
file_list = all_infos_balance['Filename'].tolist()

#print(file_paths)

olddir = 'GTSRB/Final_Training/Images'
newdir = './GTSRB_novo/Final_Training/Images'
os.makedirs(newdir, exist_ok=True)

# Gerando novos arquivos descritotes de imagens
for class_describe_file in class_describe_files:
  classe_int = int(class_describe_file.split('/')[-1].split('-')[-1].split('.')[0])
  classe_str = class_describe_file.split('/')[-1].split('-')[-1].split('.')[0]

  classe_newdir = newdir+'/'+classe_str
  os.makedirs(classe_newdir, exist_ok=True)

  new_df_classe = all_infos_balance[all_infos_balance['ClassId'] == classe_int].copy()
  new_df_classe = new_df_classe.sort_values(by='Filename')

  name_to_save = newdir+'/'+classe_str+'/'+'GT-'+classe_str+'.csv'
  new_df_classe.to_csv(name_to_save, sep=';', index=False)


# Lendo as imagens e aplicando tratamento
print("Lendo as imagens e aplicando tratamento...")
for index, row in all_infos_balance.iterrows():
  #print(row['Filename'])
  class_path = str(row['ClassId']).zfill(5)
  caminho = olddir+'/'+class_path+'/'
  filename = caminho+str(row['Filename'])
  #print(f"processing filename: {filename}")

  with archive.open(filename) as img_file:
    img = Image.open(img_file).convert('RGB')

  img = np.array(img, dtype=np.uint8)

  filtered_multiple_conditions = all_infos_balance[(all_infos_balance['ClassId'] == int(class_path)) & (all_infos_balance['Filename'] == filename.split('/')[-1])]

  #print(filtered_multiple_conditions)
  RoiX1 = int(filtered_multiple_conditions['Roi.X1'].iloc[0])
  RoiY1 = int(filtered_multiple_conditions['Roi.Y1'].iloc[0])
  RoiX2 = int(filtered_multiple_conditions['Roi.X2'].iloc[0])
  RoiY2 = int(filtered_multiple_conditions['Roi.Y2'].iloc[0])
  #print (f"  Region of interess: x1={RoiX1} y1={RoiY1} x2={RoiX2} y2={RoiY2}")

  cropped_image = img[RoiY1:RoiY2, RoiX1:RoiX2]

  IMG_SIZE = 28
  img2 = resize(cropped_image,
               output_shape=(IMG_SIZE, IMG_SIZE),
               anti_aliasing=True)
  img2 = (img2 * 255).astype(np.uint8)
  #print(img2.shape)

  caminho_novo = newdir+'/'+class_path+'/'
  os.makedirs(caminho_novo, exist_ok=True)

  new_img_file = caminho_novo+row['Filename']

  bgr_image = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
  cv2.imwrite(new_img_file, bgr_image)


# Gerando novo arquivo compactado
def zip_folder_with_root(folder_path, output_zip_path):
  """
  Zips a folder, including its root folder and all subfolders and files.
  Args:
    folder_path (str): The path to the folder to be zipped.
    output_zip_path (str): The desired path for the output .zip file.
  """
  parent_folder = os.path.dirname(folder_path)
  root_folder_name = os.path.basename(folder_path)


  with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(folder_path):
      # Calculate the relative path of the current root directory
      # with respect to the parent of the folder being zipped.
      # This ensures the root folder itself is included in the archive structure.
      relative_root = os.path.relpath(root, parent_folder)

      for file in files:
        file_path = os.path.join(root, file)
        # The arcname is the name of the file inside the zip archive.
        # It should maintain the directory structure relative to the root folder.
        arcname = os.path.join(relative_root, file)
        zipf.write(file_path, arcname)

      # To include empty subfolders, you can also add them explicitly.
      # However, zipfile.write will add empty directories automatically
      # if files within them are added. If a folder is entirely empty,
      # you might need to add it explicitly if desired.
      # For simplicity, this example focuses on files, as adding files
      # implicitly creates the necessary directory structure for them.

folder_to_zip = "./GTSRB_novo"
output_zip = "./GTSRB_novo_Final_Training_Images.zip"
print("Compactando arquivos...")
zip_folder_with_root('./GTSRB_novo', './GTSRB_novo_Final_Training_Images.zip')
print(f"Folder '{folder_to_zip}' successfully zipped to '{output_zip}'.")

archive.close()
print("pronto!!")

sys.exit(0)

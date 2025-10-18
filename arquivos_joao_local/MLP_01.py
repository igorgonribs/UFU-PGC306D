#!/usr/bin/env python
# coding: utf-8

# # **Bibliotecas**

# In[ ]:


#get_ipython().system('pip install optuna')


# In[1]:


import urllib.request
import os
import zipfile
import pandas as pd

import glob
from tqdm.auto import tqdm
tqdm.pandas()
from PIL import Image
from sklearn.model_selection import train_test_split
import shutil

import numpy as np
from skimage.transform import resize

import optuna


# # **Base de dados (placas de trânsito)**

# In[2]:


# URL para o conjunto de dados
url = ("https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip")

# Verifica se o arquivo já existe para evitar baixar novamente
if not os.path.exists("GTSRB_Final_Training_Images.zip"):
    # Define o nome do arquivo
    filename = "./GTSRB_Final_Training_Images.zip"
    # Baixa o arquivo da URL e salva com o nome definido
    urllib.request.urlretrieve(url, filename)

#Extraindo arquivos

# Caminho do arquivo zip
zip_path = 'GTSRB_Final_Training_Images.zip'

# Pasta onde os arquivos serão extraídos
extract_to = 'GTSRB_Final_Training_Images/'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print(f"Arquivos extraídos para: {extract_to}")


# In[3]:


# Lista para armazenar os caminhos completos de todos os arquivos CSV encontrados
csv_files = []

# Percorrer a pasta principal e suas subpastas
for root, dirs, files in os.walk(extract_to):
    # Filtrar apenas os arquivos .csv em cada pasta
    for file in files:
        if file.endswith('.csv'):
            # Adicionar o caminho completo do arquivo à lista
            csv_files.append(os.path.join(root, file))

dfs = []

# Ler cada arquivo CSV e adicionar a um dataframe na lista
for arquivo in csv_files:
    df = pd.read_csv(arquivo, sep=";")  # ajuste o sep se necessário
    df['Path'] = '/'.join(arquivo.split('/')[:-1]) + '/' + df.Filename
    dfs.append(df)

# Concatenar todos os dataframes em um único dataframe final
df_final = pd.concat(dfs, ignore_index=True)

# Imprimir o formato do dataframe final e as primeiras linhas
print(df_final.shape)
df_final.head()


# In[4]:


classes = {
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



# In[5]:


df_final['ClassName'] = df_final['ClassId'].map(classes)


# In[6]:


img_size = 32
def load_img(img_file):
  try:
    img = Image.open('./'+img_file).convert('RGB')
    img = np.array(img, dtype=np.uint8)

    img = resize(
        img,
        output_shape=(img_size, img_size, 3),
        mode='reflect',
        anti_aliasing=True,
        preserve_range=False  # -> output in [0,1]
    )

    img = (img * 255).astype(np.uint8)
    return img
  except:
    return None


# In[7]:


df_final['Image'] = df_final.progress_apply(lambda x: load_img(x['Path']), axis=1)

df_final = df_final.dropna(subset=['Image']).reset_index(drop=True)


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(df_final.Image,
                                                    df_final.ClassName,
                                                    test_size=0.3,
                                                    random_state=42,
                                                    stratify=df_final.ClassName
                                                    )


# In[9]:


import torch
from torch.utils.data import TensorDataset, DataLoader

# Converte os dados para tensores PyTorch e normaliza para o intervalo [-1, 1]
tensor_x_treino = torch.from_numpy(np.array(list(X_train), dtype=np.uint8)).permute(0, 3, 1, 2).to(torch.float32) / 255.0
tensor_x_teste  = torch.from_numpy(np.array(list(X_test), dtype=np.uint8)).permute(0, 3, 1, 2).to(torch.float32) / 255.0

# Normalização para o intervalo [-1, 1]
tensor_x_treino = (tensor_x_treino - 0.5) / 0.5
tensor_x_teste  = (tensor_x_teste  - 0.5) / 0.5

# Converte os rótulos para tensores PyTorch (tipo long para classificação)
# Mapear os nomes das classes para IDs numéricos
class_to_id = {name: id for id, name in classes.items()}
tensor_y_treino = torch.from_numpy(np.array([class_to_id[label] for label in y_train])).to(torch.long)
tensor_y_teste = torch.from_numpy(np.array([class_to_id[label] for label in y_test])).to(torch.long)


# Cria os TensorDatasets
dataset_treino = TensorDataset(tensor_x_treino, tensor_y_treino)
dataset_teste = TensorDataset(tensor_x_teste, tensor_y_teste)

print("TensorDatasets criados com sucesso!")


# In[10]:


import matplotlib.pyplot as plt

image, label = dataset_treino[0] # Usar dataset_treino para pegar um exemplo

# Denormalize the image data from [-1, 1] to [0, 1] for plotting
image_denormalized = (image * 0.5) + 0.5

plt.imshow(image_denormalized.permute(1, 2, 0)) # Ajusta a ordem dos canais para (altura, largura, canais) para matplotlib
print(image.dtype)


# # **Modelo**

# In[11]:


# ===== Célula 1: Imports, configs e utilitários =====
import os, random, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, TensorDataset

from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score

# Reprodutibilidade
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED); random.seed(RANDOM_SEED); torch.manual_seed(RANDOM_SEED)

# Dispositivo
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)

# Saídas
OUT_DIR = "./resultados_mlp_optuna"
os.makedirs(OUT_DIR, exist_ok=True)
GRAPH_DIR = "./graficos"
os.makedirs(GRAPH_DIR, exist_ok=True)
METRICS_CSV = os.path.join(OUT_DIR, "todas_metricas_por_epoca.csv")
TRIALS_CSV  = os.path.join(OUT_DIR, "resumo_trials.csv")
BEST_MODEL  = os.path.join(OUT_DIR, "best_model_state_dict.pt")

# Número de classes e dimensão de entrada (imagens 3x32x32 do seu pipeline)
N_CLASSES = 43          # len(classes)
INPUT_DIM = 3 * img_size * img_size # Flatten (C*H*W)


# In[12]:


# ===== Célula 2: Criação do conjunto de validação =====
# Parte do pressuposto de que você já tem: dataset_treino e dataset_teste (TensorDataset)
# e DataLoaders trainloader / testloader criados nas células anteriores do seu notebook.

# Vamos criar train/val a partir do dataset_treino (80/20):
val_ratio = 0.2

# Cria um TensorDataset com dados de treino e rótulos
full_train_dataset = TensorDataset(tensor_x_treino, tensor_y_treino)

n_total = len(full_train_dataset)
n_val = int(n_total * val_ratio)
n_train = n_total - n_val

# Divide o dataset completo de treino em treino e validação
train_ds, val_ds = random_split(full_train_dataset, [n_train, n_val],
                                generator=torch.Generator().manual_seed(RANDOM_SEED))

# DataLoaders (mantendo batch e workers semelhantes aos seus)
BATCH_SIZE_DEFAULT = 64

pin = torch.cuda.is_available()

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE_DEFAULT, shuffle=True,
    num_workers=0, pin_memory=pin, persistent_workers=False
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE_DEFAULT, shuffle=False,
    num_workers=0, pin_memory=pin, persistent_workers=False
)
test_loader = DataLoader(
    dataset_teste, batch_size=BATCH_SIZE_DEFAULT, shuffle=False,
    num_workers=0, pin_memory=pin, persistent_workers=False
)

len(train_ds), len(val_ds), len(dataset_teste)

len(train_ds), len(val_ds), len(dataset_teste)


# In[13]:


# ===== Célula 3: Construtor do MLP (nn.Sequential) =====
def build_mlp_sequential(input_dim:int,
                         hidden_sizes:list,
                         output_dim:int,
                         activation:str="relu",
                         dropout_p:float=0.0) -> nn.Sequential:
    """
    Cria um MLP via nn.Sequential:
    - Começa com nn.Flatten() para achatar (N, C, H, W) -> (N, C*H*W)
    - hidden_sizes: lista com nº de neurônios por camada oculta (ex.: [256,128])
    - activation: 'relu' ou 'sigmoid'
    - dropout_p: 0.0..0.5
    - output_dim: número de classes (43)
    """
    acts = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid}
    layers = [nn.Flatten()]
    prev = input_dim
    for h in hidden_sizes:
        layers += [nn.Linear(prev, h), acts[activation]()]
        if dropout_p > 0:
            layers.append(nn.Dropout(dropout_p))
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers).to(DEVICE)


# In[14]:


# ===== Célula 4: Funções de treino/val/test =====
def train_one_epoch(model, loader, optimizer, criterion):
    """
    Treina 1 época:
    - Retorna (loss_médio, acc)
    """
    model.train()
    losses, preds_all, y_all = [], [], []
    pbar = tqdm(loader, desc="Treinando", leave=False)
    for xb, yb in pbar:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        preds = torch.argmax(logits, 1)
        preds_all.append(preds.detach().cpu().numpy())
        y_all.append(yb.detach().cpu().numpy())
        pbar.set_postfix(loss=np.mean(losses))

    preds_all = np.concatenate(preds_all); y_all = np.concatenate(y_all)
    acc = accuracy_score(y_all, preds_all)
    return float(np.mean(losses)), float(acc)


@torch.no_grad()
def evaluate(model, loader, criterion):
    """
    Avaliação:
    - Retorna (loss_médio, acc)
    """
    model.eval()
    losses, preds_all, y_all = [], [], []
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        loss = criterion(logits, yb)
        losses.append(loss.item())
        preds = torch.argmax(logits, 1)
        preds_all.append(preds.detach().cpu().numpy())
        y_all.append(yb.detach().cpu().numpy())

    preds_all = np.concatenate(preds_all); y_all = np.concatenate(y_all)
    acc = accuracy_score(y_all, preds_all)
    return float(np.mean(losses)), float(acc)


# In[15]:


# ===== Célula 5: Execução de um treino completo com logging =====
def run_training(params:dict,
                 train_loader, val_loader, test_loader,
                 input_dim:int, n_classes:int,
                 n_epochs:int, # Added n_epochs as a parameter
                 trial_id=None):
    """
    Executa treino completo conforme 'params' e registra histórico por época.
    params esperados:
      - hidden_sizes: list[int]
      - activation: 'relu'|'sigmoid'
      - dropout_p: float [0, 0.5]
      - lr: float
      - batch_size: int (não usado aqui pois DataLoader já está criado)
    """
    lr       = params.get("lr", 1e-3)
    # n_epochs = params.get("n_epochs", 12) # Removed n_epochs from params
    hidden   = params["hidden_sizes"]
    act      = params["activation"]
    drop     = params["dropout_p"]

    model = build_mlp_sequential(input_dim, hidden, n_classes, activation=act, dropout_p=drop)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    best_val_acc, best_state = -1.0, None

    for epoch in tqdm(range(1, n_epochs+1), desc=f"Trial {trial_id} | Épocas", leave=False):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        va_loss, va_acc = evaluate(model, val_loader, criterion)

        history.append({
            "trial_id": trial_id,
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": va_loss,
            "val_acc": va_acc,
            **params,
        })

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Avaliar no teste usando o melhor estado
    if best_state is not None:
        model.load_state_dict(best_state)
    te_loss, te_acc = evaluate(model, test_loader, criterion)

    return {
        "history": history,
        "best_val_acc": best_val_acc,
        "test_loss": te_loss,
        "test_acc": te_acc,
        "best_state": best_state,
    }


# In[16]:


def suggest_params_optuna(trial):
    """
    Espaço de busca:
      - nº de camadas: 1..4
      - neurônios: 32..512 (log)
      - ativação: relu|sigmoid
      - dropout: 0.0..0.5
      - lr: 1e-4..3e-2 (log)
      - batch_size: 64|128|256  (não usado aqui, mas salvo no log)
    """
    n_layers = trial.suggest_int("n_hidden_layers", 1, 4)
    hidden = [trial.suggest_int(f"n_units_l{i+1}", 32, 512, log=True) for i in range(n_layers)]
    return {
        "hidden_sizes": hidden,
        "activation": trial.suggest_categorical("activation", ["relu", "sigmoid"]),
        "dropout_p": trial.suggest_float("dropout_p", 0.0, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 3e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
    }

def random_params_fallback():
    """Fallback simples caso Optuna não esteja disponível."""
    n_layers = random.randint(1, 4)
    hidden = [int(2 ** random.randint(5, 9)) for _ in range(n_layers)]  # 32..512
    return {
        "hidden_sizes": hidden,
        "activation": random.choice(["relu", "sigmoid"]),
        "dropout_p": random.uniform(0.0, 0.5),
        "lr": 10 ** random.uniform(-4, -1.5),
        "batch_size": random.choice([64, 128, 256]),
    }


# In[18]:


# ===== Célula 7: Execução da busca, CSVs e melhor modelo =====
def run_hparam_search(n_trials=5, n_epochs=50, output_prefix=""):
    """
    Executa n_trials de busca (Optuna ou fallback), salva CSVs e re-treina o melhor.
    Adiciona um prefixo aos arquivos de saída se especificado.
    Retorna o DataFrame de trials e o DataFrame de histórico do melhor trial,
    além dos melhores parâmetros encontrados.
    """
    all_hist = []
    trials_rows = []
    best_params = None # Initialize best_params

    # Define os caminhos dos arquivos de saída com o prefixo
    metrics_csv_path = os.path.join(OUT_DIR, f"{output_prefix}todas_metricas_por_epoca.csv")
    trials_csv_path  = os.path.join(OUT_DIR, f"{output_prefix}resumo_trials.csv")
    best_model_path  = os.path.join(OUT_DIR, f"{output_prefix}best_model_state_dict.pt")

    if optuna is not None:
        print("Optuna encontrado. Iniciando estudo...")
        def objective(trial):
            params = suggest_params_optuna(trial)
            res = run_training(params, train_loader, val_loader, test_loader,
                               INPUT_DIM, N_CLASSES, n_epochs=n_epochs, trial_id=trial.number)
            all_hist.extend(res["history"])

            # Get metrics from the last epoch for this trial
            last_epoch_metrics = res["history"][-1] if res["history"] else {}

            trials_rows.append({
                "trial_id": trial.number, **params,
                "best_val_acc": res["best_val_acc"],
                "test_acc": res["test_acc"],
                "test_loss": res["test_loss"],
                "train_acc": last_epoch_metrics.get("train_acc"),
                "val_acc": last_epoch_metrics.get("val_acc"),
                "train_loss": last_epoch_metrics.get("train_loss"),
                "val_loss": last_epoch_metrics.get("val_loss"),
            })
            print(f"Trial {trial.number} - Test Acc: {res['test_acc']:.4f}, Test Loss: {res['test_loss']:.4f}")
            return res["best_val_acc"]

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        best_trial = study.best_trial

        # Reconstruir params do melhor trial
        hidden = [best_trial.params[k] for k in sorted(best_trial.params) if k.startswith("n_units_l")]
        best_params = {
            "hidden_sizes": hidden,
            "activation": best_trial.params["activation"],
            "dropout_p": best_trial.params["dropout_p"],
            "lr": best_trial.params["lr"],
            "batch_size": best_trial.params["batch_size"],
            "n_epochs": n_epochs, # Use the fixed n_epochs
        }
        print(f"Melhor trial #{best_trial.number} | best_val_acc={study.best_value:.4f}")
    else:
        print("Optuna NÃO encontrado. Usando fallback aleatório.")
        best_val, best_params = -1.0, None
        for t in tqdm(range(n_trials), desc="Trials (fallback)"):
            params = random_params_fallback()
            res = run_training(params, train_loader, val_loader, test_loader,
                               INPUT_DIM, N_CLASSES, n_epochs=n_epochs, trial_id=t)
            all_hist.extend(res["history"])

            # Get metrics from the last epoch for this trial
            last_epoch_metrics = res["history"][-1] if res["history"] else {}

            trials_rows.append({
                "trial_id": t, **params,
                "best_val_acc": res["best_val_acc"],
                "test_acc": res["test_acc"],
                "test_loss": res["test_loss"],
                "train_acc": last_epoch_metrics.get("train_acc"),
                "val_acc": last_epoch_metrics.get("val_acc"),
                "train_loss": last_epoch_metrics.get("train_loss"),
                "val_loss": last_epoch_metrics.get("val_loss"),
            })
            print(f"Trial {t} - Test Acc: {res['test_acc']:.4f}, Test Loss: {res['test_loss']:.4f}")
            if res["best_val_acc"] > best_val:
                best_val, best_params = res["best_val_acc"], params

    # Salvar CSVs
    pd.DataFrame(all_hist).to_csv(metrics_csv_path, index=False)
    trials_df = pd.DataFrame(trials_rows).sort_values("best_val_acc", ascending=False)

    # Reorder columns
    metric_cols = ["train_acc", "val_acc", "test_acc", "train_loss", "val_loss", "test_loss"]
    other_cols = [col for col in trials_df.columns if col not in metric_cols]
    trials_df = trials_df[other_cols[:6] + metric_cols + other_cols[6:]] # Adjust indices based on your dataframe structure

    trials_df.to_csv(trials_csv_path, index=False)

    # Re-treinar melhor configuração (para salvar e plotar)
    print("Treinando melhor configuração para salvar modelo e curvas...")
    best_run = run_training(best_params, train_loader, val_loader, test_loader,
                            INPUT_DIM, N_CLASSES, n_epochs=n_epochs, trial_id=9999)
    torch.save(best_run["best_state"], best_model_path)
    print(f"Melhor configuração - Test Acc: {best_run['test_acc']:.4f}, Test Loss: {best_run['test_loss']:.4f}")


    return trials_df, pd.DataFrame(best_run["history"]), best_params # Return best_params

# Execute (ajuste n_trials conforme seu tempo)
trials_df, best_hist_df, best_params = run_hparam_search(n_trials=25, n_epochs=40, output_prefix="experimento_01_") # Capture best_params
trials_df.head()


# In[19]:

# ===== Célula 8: Visualizações e caminhos dos arquivos =====
plt.figure(figsize=(8,5))
plt.plot(best_hist_df["epoch"], best_hist_df["train_loss"], label="Train Loss")
plt.plot(best_hist_df["epoch"], best_hist_df["val_loss"],   label="Val Loss")
plt.title("Curva de Loss - Melhor Configuração")
plt.xlabel("Época"); plt.ylabel("Loss"); plt.grid(True); plt.legend()
plt.tight_layout()
filename=GRAPH_DIR+'/experimento_01_curvas_loss.png'
plt.savefig(filename, dpi=150)
plt.show()

plt.figure(figsize=(8,5))
plt.plot(best_hist_df["epoch"], best_hist_df["train_acc"], label="Train Acc")
plt.plot(best_hist_df["epoch"], best_hist_df["val_acc"],   label="Val Acc")
plt.title("Curva de Acurácia - Melhor Configuração")
plt.xlabel("Época"); plt.ylabel("Acurácia"); plt.grid(True); plt.legend()
plt.tight_layout()
filename=GRAPH_DIR+'/experimento_01_curvas_acuracia.png'
plt.savefig(filename, dpi=150)
plt.show()

# Atualiza a impressão dos caminhos dos arquivos para refletir o prefixo
# Para fazer isso de forma mais flexível, vamos redefinir os caminhos aqui
output_prefix = "experimento_01_" # Certifique-se que este prefixo corresponde ao usado em run_hparam_search
OUT_DIR = "./resultados_mlp_optuna" # Mantenha a pasta base

metrics_csv_path = os.path.join(OUT_DIR, f"{output_prefix}todas_metricas_por_epoca.csv")
trials_csv_path  = os.path.join(OUT_DIR, f"{output_prefix}resumo_trials.csv")
best_model_path  = os.path.join(OUT_DIR, f"{output_prefix}best_model_state_dict.pt")


print("Arquivos salvos em:", os.path.abspath(OUT_DIR))
print(" - Métricas por época:", metrics_csv_path)
print(" - Resumo dos trials: ", trials_csv_path)
print(" - Melhor modelo:     ", best_model_path)


# In[ ]:





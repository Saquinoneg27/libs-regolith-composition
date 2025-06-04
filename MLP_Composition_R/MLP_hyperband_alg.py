# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:23:28 2024

@author: alejo
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import os
import pandas as pd
import numpy as np
import optuna
from optuna.trial import TrialState

# Directory where the files are saved
directory = r'C:\Users\alejo\OneDrive\Desktop\ingeniería física\8vo Brno University\Spectra\Mars\Moon samples'
ordered_files = [
    'LHS-1E.txt', 'LMS-1.txt', 'Al2O3-BL.txt','SiO2-CL.txt', 'SiO2-BL.txt',
    'TiO2-BL.txt','MgO-BL.txt', 'Al2O3-AL.txt',
    'LHS-1.txt', 'Dusty.txt', 'LHS-1D.txt', 
    'K2O-AL.txt', 'Na2O-AL.txt', 'MgO-AL.txt', 'SiO2-DL.txt', 
    'SiO2-AL.txt', 'TiO2-AL.txt',  'MgO-CL.txt', 'TiO2-CL.txt', 'TiO2-DL.txt'
]

pure_samples = []

spectra_data = []
labels = []

for file_name in ordered_files:
    file_path = os.path.join(directory, file_name)
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = [list(map(float, line.strip().split())) for line in lines]
    df = pd.DataFrame(data)
    df.columns = ['Wavelength'] + [f'Measurement_{i}' for i in range(1, df.shape[1])]
    
    wavelengths_np = df['Wavelength'].to_numpy()
    intensities = df.drop('Wavelength', axis=1).to_numpy()
    
    spectrum_data = {
        'file': file_name,
        'wavelengths': wavelengths_np,
        'intensities': intensities
    }
    
    spectra_data.append(spectrum_data)
    labels.append(file_name)

# Extract relevant information from the composition data

# Load the Excel data
file_path = r'C:\Users\alejo\OneDrive\Desktop\ingeniería física\8vo Brno University\Codes\MLP_Composition_R\Composition_percentages.xlsx'
composition_data = pd.read_excel(file_path)

# Extract relevant information from the composition data
composition_dict = composition_data.set_index('Unnamed: 0').T.to_dict('list')
print(composition_data.head())


# Prepare the dataset
X = []
y = []

for spectrum in spectra_data:
    file_name = spectrum['file']
    intensities = spectrum['intensities']

    if file_name in composition_dict:
            composition = composition_dict[file_name]
            for i in range(intensities.shape[1]):
                X.append(intensities[:, i])
                y.append([composition[0], composition[2], composition[3], composition[5]])
    else:
        print(f"File {file_name} not found in composition dictionary.")

X = np.array(X)
y = np.array(y)
##########################################
# Normalizar nuevamente
X_norm = X / np.max(X, axis=1, keepdims=True)

# Normalize the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_norm)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')



# Definir el modelo MLP
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        layers = []
        in_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size
        layers.append(nn.Linear(in_size, output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# Función de entrenamiento
def objective(trial):
    # Definir hiperparámetros
    hidden_sizes = trial.suggest_categorical("hidden_sizes", [[256, 128], [512, 256], [1024, 512]])
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    num_epochs = 50

    kf = KFold(n_splits=3)
    fold = 1
    val_losses = []

    for train_index, val_index in kf.split(X_scaled):
        X_train, X_val = X_scaled[train_index], X_scaled[val_index]
        y_train, y_val = y[train_index], y[val_index]

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

        model = MLP(input_size=X_train.shape[1],
                    hidden_sizes=hidden_sizes,
                    output_size=4).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            epoch_loss /= len(train_loader)
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
            
            val_losses.append(val_loss)
            if epoch_loss < 0.001:
                break
        fold += 1

    return np.mean(val_losses)

# Ejecutar la búsqueda de hiperparámetros con Optuna
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

# Obtener los mejores resultados
best_trial = study.best_trial

print("Best trial:")
print(f"  Value: {best_trial.value}")
print("  Params: ")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 09:13:12 2024

@author: alejo
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# Directory where the files are saved
directory = r'C:\Users\alejo\OneDrive\Desktop\ingeniería física\8vo Brno University\Sergio\Mars\Moon samples'
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

# Iterate through each file in the ordered list
for file_name in ordered_files:
    file_path = os.path.join(directory, file_name)
    
    # Open and read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Convert lines to a list of floats and create a DataFrame
    data = [list(map(float, line.strip().split())) for line in lines]
    df = pd.DataFrame(data)
    
    # Set column names: 'Wavelength' and 'Measurement_i' for each measurement
    df.columns = ['Wavelength'] + [f'Measurement_{i}' for i in range(1, df.shape[1])]
    
    # Convert wavelengths to numpy arrays and extract intensities
    wavelengths_np = df['Wavelength'].to_numpy()
    intensities = df.drop('Wavelength', axis=1).to_numpy()
    
    # Store the data in a dictionary
    spectrum_data = {
        'file': file_name,
        'wavelengths': wavelengths_np,
        'intensities': intensities
    }
    
    # Append processed data to the lists
    spectra_data.append(spectrum_data)
    labels.append(file_name)  # File name represents the class label

# Load the Excel file containing the composition data
file_path = r'C:\Users\alejo\OneDrive\Desktop\ingeniería física\8vo Brno University\Codes\CNN_Composition\Composition_percentages.xlsx'
composition_data = pd.read_excel(file_path)

# Create a dictionary from the composition data for quick lookup
composition_dict = composition_data.set_index('Unnamed: 0').T.to_dict('list')
print(composition_data.head())

# Prepare the dataset
X = []
y = []

# Match each spectrum with its corresponding composition
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

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Normalize the spectra
X_norm = X / np.max(X, axis=1, keepdims=True)

# Normalize input features using Standard Scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_norm)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# K-Fold Cross-Validation
kf = KFold(n_splits=2)
fold = 1

# Learning rate scheduler function
def scheduler(optimizer, epoch, lr_decay=0.8, lr_step=10):
    if epoch % lr_step == 0 and epoch != 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay

# Function to calculate Mean Absolute Error (MAE)
def mean_absolute_error(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))

# List to store MAE scores for each fold
mae_scores = []

for train_index, val_index in kf.split(X_scaled):
    print(f'Fold {fold}')
    
    # Split data into training and validation sets
    X_train, X_val = X_scaled[train_index], X_scaled[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # Convert to PyTorch tensors and move to GPU if available
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    # Define the MLP model
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

    # Model parameters
    input_size = X_train.shape[1]
    hidden_sizes = [512, 256]
    output_size = 4  # Number of outputs

    # Build the model and move it to the GPU if available
    model = MLP(input_size, hidden_sizes, output_size).to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0011212623520098514)

    # Implementing Early Stopping
    class EarlyStopping:
        def __init__(self, patience=5, min_delta=0):
            self.patience = patience
            self.min_delta = min_delta
            self.best_loss = None
            self.counter = 0
            self.early_stop = False

        def __call__(self, val_loss):
            if self.best_loss is None:
                self.best_loss = val_loss
            elif self.best_loss - val_loss > self.min_delta:
                self.best_loss = val_loss
                self.counter = 0
            elif self.best_loss - val_loss < self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

    early_stopping = EarlyStopping(patience=10, min_delta=0.01)

    train_losses = []
    val_losses = []

    num_epochs = 80
    batch_size = 8
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        scheduler(optimizer, epoch, lr_decay=0.8)  # Using lr_decay of 0.8
        epoch_loss = 0.0
        epoch_mae = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_mae += mean_absolute_error(batch_y, outputs).item()
        
        epoch_loss /= len(train_loader)
        epoch_mae /= len(train_loader)
        train_losses.append(epoch_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
            val_mae = mean_absolute_error(y_val_tensor, val_outputs).item()
        val_losses.append(val_loss)
        
        model.train()
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train MAE: {epoch_mae:.4f}, Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}')
        
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Store final validation MAE for this fold
    mae_scores.append(val_mae)

    fold += 1

    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Training and Validation Loss (Fold {fold-1})')
    plt.show()

# Calculate the mean and standard deviation of MAE across all folds
mae_mean = np.mean(mae_scores)
mae_std = np.std(mae_scores)

print(f'Average MAE across all folds: {mae_mean:.4f}')
print(f'Standard deviation of MAE across all folds: {mae_std:.4f}')

from torchsummary import summary

# Print the model summary
summary(model, input_size=(input_size,))

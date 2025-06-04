import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import silhouette_score


def find_best_random_state(data, num_clusters, n_trials=100):
    """
  This function finds the best random state for K-Means clustering using silhouette score.

  Args:
      data (numpy.ndarray): Data matrix used for clustering.
      num_clusters (int): Number of clusters to use in K-Means clustering.
      n_trials (int, optional): Number of random state trials to evaluate. Defaults to 100.

  Returns:
      tuple: Tuple containing the best random state (integer seed) and the corresponding silhouette score (float).
    """
    best_random_state = None
    best_silhouette_score = -1

    for random_state in range(n_trials):
        kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=random_state)
        clusters = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, clusters)
        
        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_random_state = random_state

    return best_random_state, best_silhouette_score

# Directorio de los archivos y lista de archivos ordenados
directory = r'c:\Users\alejo\OneDrive\Desktop\ingeniería física\8vo Brno University\Sergio\Mars\Moon samples'
ordered_files = [
    'LHS-1E.txt', 'LHS-1.txt', 'Dusty.txt', 'LHS-1D.txt', 'LMS-1.txt', 'Al2O3-BL.txt',
    'K2O-AL.txt', 'Na2O-AL.txt', 'MgO-AL.txt', 'SiO2-DL.txt', 'SiO2-CL.txt', 'SiO2-BL.txt',
    'SiO2-AL.txt', 'TiO2-AL.txt', 'TiO2-BL.txt', 'MgO-CL.txt', 'TiO2-CL.txt', 'TiO2-DL.txt',
    'MgO-BL.txt', 'Al2O3-AL.txt'
]

spectra_data = []
labels = []

for file_name in ordered_files:
    # Construir la ruta completa del archivo desde el directorio y el nombre del archivo
    file_path = os.path.join(directory, file_name)
    
    # Leer el contenido del archivo línea por línea
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Convertir cada línea (cadena) en una lista de flotantes
    data = [list(map(float, line.strip().split())) for line in lines]
    df = pd.DataFrame(data)
    df.columns = ['Wavelength'] + [f'Measurement_{i}' for i in range(1, df.shape[1])]

    for column in df.columns[1:]:
        spectrum_data = {
            'wavelengths': df['Wavelength'].to_numpy(),
            'intensities': df[column].to_numpy()
        }
        spectra_data.append(spectrum_data)
        labels.append(file_name)  # Asumiendo que el nombre del archivo representa la etiqueta de clase

# Convertir listas a arrays de numpy para la entrada del modelo
X = np.array([spectrum['intensities'] for spectrum in spectra_data])

# Normalizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar PCA con 3 componentes
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Método del codo para determinar el número óptimo de clusters
SS = []
CONT = []
max_clusters = min(20, len(spectra_data))

for i in range(1, max_clusters):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_pca)
    SS.append(kmeans.inertia_)
    CONT.append(i)

# Graficar el codo
plt.figure(figsize=(8, 6))
plt.plot(CONT, SS, 'bx-')
plt.xlabel('Número de Clusters')
plt.ylabel('Inercia')
plt.title('Método del Codo para el número óptimo de clusters')
plt.show()

# Determinar el número óptimo de clusters (asumiendo que el número óptimo es 5 para este ejemplo)
optimal_clusters = 5  # Este número debe seleccionarse basado en el gráfico del codo

# Aplicar KMeans con el número óptimo de clusters
best_random_state, best_silhouette_score = find_best_random_state(X_pca, optimal_clusters)

print(f"Best random state: {best_random_state} with a silhouette score of {best_silhouette_score:.4f}")

kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=best_random_state)
y_kmeans = kmeans.fit_predict(X_pca)

# Graficar los clusters en 2D para PC2 vs PC1 y PC3 vs PC1
plt.figure(figsize=(14, 6))

# PC2 vs PC1
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='viridis', edgecolor='k', s=50)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Clusters: PC2 vs PC1')
plt.colorbar(label='Cluster')

# PC3 vs PC1
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 2], c=y_kmeans, cmap='viridis', edgecolor='k', s=50)
plt.xlabel('PC1')
plt.ylabel('PC3')
plt.title('Clusters: PC3 vs PC1')
plt.colorbar(label='Cluster')

plt.tight_layout()
plt.show()

# Preparar los datos para el CNN
X = X / np.max(X, axis=1, keepdims=True)  # Normalizar nuevamente
y_categorical = to_categorical(y_kmeans)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Reajustar para la entrada del CNN
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Verificar formas después de reajustar
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# Definir el modelo
model = Sequential()
model.add(Conv1D(filters=8, kernel_size=8, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=16, kernel_size=4, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=y_train.shape[1], activation='softmax'))  # Asegurarse de que coincida con el número de clases

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.2, verbose=1)

# Evaluar el modelo
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test accuracy: {accuracy * 100:.2f}%')

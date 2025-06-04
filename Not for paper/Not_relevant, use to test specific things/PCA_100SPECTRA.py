import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, medfilt
from scipy.interpolate import InterpolatedUnivariateSpline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from sklearn.metrics import silhouette_score
from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.layouts import gridplot
import os
import seaborn as sns
from scipy.stats import ttest_rel
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D

def find_best_random_state(data, num_clusters, n_trials=100):
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

lower_bound = {"CaO": 422, "MgO": 279.43, "Al2O3": 308.65, "SiO2": 286.5, "TiO2": 334.5, "Fe2O3": 381.95, "TiO2": 334.4, "K2O": 769}
upper_bound = {"CaO": 423.5, "MgO": 279.66, "Al2O3": 309.9, "SiO2": 290, "TiO2": 335.36, "Fe2O3": 382.15, "TiO2": 335.4, "K2O": 771}

file_path_conc = r'C:\Users\alejo\OneDrive\Desktop\ingeniería física\8vo Brno University\Sergio\Concentrations.xlsx'
df_concentrations = pd.read_excel(file_path_conc)

atm = int(input("Choose the kind of samples: 1: Moon samples, 2: Mars samples :"))
aux = 2

if atm == 1:   
    directory = r'c:\Users\alejo\OneDrive\Desktop\ingeniería física\8vo Brno University\Sergio\Mars\Moon samples'
    ordered_files = [
        'LHS-1.txt',
        'Dusty.txt',
        'LHS-1D.txt',
        'K2O-AL.txt',
        'Na2O-AL.txt',
        'MgO-AL.txt',
        'SiO2-DL.txt',
        'SiO2-AL.txt',
        'TiO2-AL.txt',
        'MgO-CL.txt',
        'TiO2-CL.txt',
        'TiO2-DL.txt'
    ]

if atm == 2:   
    directory = r'c:\Users\alejo\OneDrive\Desktop\ingeniería física\8vo Brno University\Sergio\Mars\Mars samples'
    ordered_files = [
       'MGS-1.txt',
       'Na2O-BM.txt',
       'Al2O3-BM.txt',
       'K2O-CM.txt',
       'K2O-BM.txt',
        'K2O-DM.txt',
       'MgO-AM.txt',
        'MgO-BM.txt',
       'SiO2-AM.txt',
        'SiO2-CM.txt',
        'Al2O3-BM.txt',
        'CaO-AM.txt',
        'K2O-AM.txt'
    ]

spectra_data = []
references = {}
samples = {}
wavelengths_common = None

for file_name in ordered_files:
    file_path = os.path.join(directory, file_name)
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = [list(map(float, line.strip().split())) for line in lines]
    df = pd.DataFrame(data)
    df.columns = ['Wavelength'] + [f'Measurement_{i}' for i in range(1, df.shape[1])]
    normalization_factors = df.iloc[:, 1:].sum(axis=0)
    df.iloc[:, 1:] = df.iloc[:, 1:].div(normalization_factors, axis=1)

    for i in range(1, df.shape[1]):
        spectrum_data = {
            'file': file_name,
            'wavelengths': df['Wavelength'].to_numpy(),
            'intensities': df.iloc[:, i].to_numpy()
        }
        if 'Reference' in file_name:
            if file_name not in references:
                references[file_name] = []
            references[file_name].append(spectrum_data)
        else:
            if file_name not in samples:
                samples[file_name] = []
            samples[file_name].append(spectrum_data)
        spectra_data.append(spectrum_data)

# Data from spectra to PCA matrix
num_clusters = 5
intensity_matrix = np.array([spectrum['intensities'] for spectrum in spectra_data])
pca = PCA(n_components=3)
principal_components = pca.fit_transform(intensity_matrix)
explained_variance = pca.explained_variance_ratio_

# KMeans clustering
best_random_state, best_silhouette_score = find_best_random_state(principal_components, num_clusters)
kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=best_random_state)
clusters = kmeans.fit_predict(principal_components)

# Find the majority cluster for each sample and filter out outliers
filtered_spectra_data = []
filtered_principal_components = []
filtered_clusters = []

for file_name in ordered_files:
    sample_spectra = [s for s in spectra_data if s['file'] == file_name]
    sample_indices = [i for i, s in enumerate(spectra_data) if s['file'] == file_name]
    sample_clusters = clusters[sample_indices]
    
    # Find the majority cluster
    cluster_counts = Counter(sample_clusters)
    majority_cluster = cluster_counts.most_common(1)[0][0]
    
    # Filter spectra to keep only those in the majority cluster
    for i, spectrum in zip(sample_indices, sample_spectra):
        if clusters[i] == majority_cluster:
            filtered_spectra_data.append(spectrum)
            filtered_principal_components.append(principal_components[i])
            filtered_clusters.append(clusters[i])

filtered_principal_components = np.array(filtered_principal_components)
filtered_clusters = np.array(filtered_clusters)


# Crear un directorio para guardar los espectros filtrados
filtered_directory = os.path.join(directory, "filtered_spectra")
os.makedirs(filtered_directory, exist_ok=True)

# Guardar los espectros filtrados en nuevos archivos
for file_name in ordered_files:
    sample_spectra = [s for s in filtered_spectra_data if s['file'] == file_name]
    if sample_spectra:
        wavelengths = sample_spectra[0]['wavelengths']
        intensities = [s['intensities'] for s in sample_spectra]
        df = pd.DataFrame(intensities).transpose()
        df.insert(0, 'Wavelength', wavelengths)
        filtered_file_path = os.path.join(filtered_directory, f"{file_name}")
        df.to_csv(filtered_file_path, index=False, sep=' ', header=False)



# Plotting the results
combinations = [(0, 1), (0, 2), (1, 2)]
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for (i, j), ax in zip(combinations, axes):
    for cluster in range(num_clusters):
        cluster_indices = np.where(filtered_clusters == cluster)
        ax.scatter(filtered_principal_components[cluster_indices, i], filtered_principal_components[cluster_indices, j], 
                   label=f'Cluster {cluster + 1}')
    ax.set_xlabel(f'Principal Component {i+1}')
    ax.set_ylabel(f'Principal Component {j+1}')
    ax.set_title(f'PC{i+1} vs PC{j+1}')
    ax.legend()

plt.tight_layout()
plt.show()

# Plotting the 3D results
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')

for cluster in range(num_clusters):
    cluster_indices = np.where(filtered_clusters == cluster)
    ax.scatter(filtered_principal_components[cluster_indices, 0], filtered_principal_components[cluster_indices, 1], filtered_principal_components[cluster_indices, 2], 
               label=f'Cluster {cluster + 1}')


ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('3D PCA of Spectra with Cluster Highlighting')
ax.legend()
plt.show()
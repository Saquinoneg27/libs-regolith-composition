import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import InterpolatedUnivariateSpline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os



def process_spectrum(x, y, lower_bound, upper_bound):
    mask = (x >= lower_bound) & (x <= upper_bound)
    x_roi = x[mask]
    y_roi = y[mask]
    peaks, properties = find_peaks(y_roi)
    if len(peaks) == 0:
        return x_roi, y_roi, None, None, None
    main_peak = peaks[np.argmax(y_roi[peaks])]
    
    # Calculate the total area under the peak
    area_under_peak = np.trapz(y_roi, x_roi)
    
    return x_roi, y_roi, main_peak, None, area_under_peak


lower_bound = {"CaO":391.95, "MgO": 279.43, "Al2O3": 308.65, "SiO2": 286.5, "Na2O":588.5, "TiO2": 334.5}
upper_bound = {"CaO":395.66, "MgO": 279.66, "Al2O3": 309.9, "SiO2": 290, "Na2O":589.5, "TiO2":335.36}


# Cargar los datos desde el archivo de Excel
file_path_conc = r'c:\Users\alejo\OneDrive\Desktop\ingeniería física\8vo Brno University\Codes\concentraciones.xlsx' 

# Leer el archivo de Excel
df_concentrations = pd.read_excel(file_path_conc)


aux= int(input("Do you want to do a reference or sample analysis; 1(Ref), 2(Sample): "))


# Directorio de archivos
directory = r'c:\Users\alejo\OneDrive\Desktop\ingeniería física\8vo Brno University\Sergio\Mars\Moon samples'

if aux ==1:
   
    # Listar archivos
    ordered_files = [f for f in os.listdir(directory) if f.startswith('Reference_') and f.endswith('.txt')]

if aux == 2:   
    ordered_files = [
    'Reference_00.txt', 'LHS-1E.txt',
    'Reference_01.txt', 'LHS-1.txt',
    'Reference_02.txt', 'Dusty.txt',
    'Reference_03.txt', 'LHS-1D.txt',
    'Reference_04.txt', 'LMS-1.txt',
    'Reference_05.txt', 'Al2O3-BL.txt',
    'Reference_06.txt', 'K2O-AL.txt',
    'Reference_07.txt', 'Na2O-AL.txt',
    'Reference_08.txt', 'MgO-AL.txt',
    #'Reference_09.txt', 'SiO2-DL.txt',
    'Reference_10.txt', 'SiO2-CL.txt',
    'Reference_11.txt', 'SiO2-BL.txt',
    'Reference_12.txt', 'SiO2-AL.txt',
    'Reference_13.txt', 'TiO2-AL.txt',
    'Reference_14.txt', 'TiO2-BL.txt',
    'Reference_15.txt', 'MgO-CL.txt',
    'Reference_16.txt', 'TiO2-CL.txt',
    'Reference_17.txt', 'TiO2-DL.txt',
    'Reference_18.txt', 'MgO-BL.txt',
    'Reference_19.txt', 'Al2O3-AL.txt'
    ]
    compound = input("Enter the compound to analyze (e.g., SiO2): ")




#compound = input("Enter the compound to analyze (e.g., SiO2): ")
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
    average_spectrum = df.iloc[:, 1:].mean(axis=1)

    peaks, properties = find_peaks(average_spectrum, prominence=0.5, width=3)
    blank_indices = [i for i in range(len(average_spectrum)) if i not in peaks]
    blank_values = [average_spectrum[i] for i in blank_indices]
    x_bi = np.mean(blank_values)
    s_bi = np.std(blank_values)
    k = 3
    lod = x_bi + k * s_bi

    corrected_intensity = np.where(average_spectrum >= lod, average_spectrum, 0)
    corrected_intensity /= np.sum(corrected_intensity)

    df['Corrected_Intensity'] = corrected_intensity
    wavelengths_np = df['Wavelength'].to_numpy()
    intensities_np = df['Corrected_Intensity'].to_numpy()
    
    if wavelengths_common is None:
        wavelengths_common = wavelengths_np
    else:
        intensities_np = np.interp(wavelengths_common, wavelengths_np, intensities_np)

    spectrum_data = {
        'file': file_name,
        'wavelengths': wavelengths_common,
        'intensities': intensities_np
    }
    
    if 'Reference' in file_name:
        references[file_name] = spectrum_data
    else:
        samples[file_name] = spectrum_data
    
    spectra_data.append(spectrum_data)


# Filtrar solo las muestras de spectra_data
sample_spectra = [spectrum for spectrum in spectra_data if 'Reference' not in spectrum['file']]

# Convertir datos de espectros a matriz para PCA solo con muestras
intensity_matrix = np.array([spectrum['intensities'] for spectrum in sample_spectra])
pca = PCA(n_components=3)
principal_components = pca.fit_transform(intensity_matrix)
explained_variance = pca.explained_variance_ratio_

fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')
colormap = plt.colormaps['tab20']
colors = colormap(np.linspace(0, 1, len(sample_spectra)))

for i, spectrum in enumerate(sample_spectra):
    ax.scatter(principal_components[i, 0], principal_components[i, 1], principal_components[i, 2], 
               color=colors[i], label=spectrum['file'])

ax.set_xlabel(f'Principal Component 1 ({explained_variance[0]*100:.2f}%)')
ax.set_ylabel(f'Principal Component 2 ({explained_variance[1]*100:.2f}%)')
ax.set_zlabel(f'Principal Component 3 ({explained_variance[2]*100:.2f}%)')
ax.set_title('3D PCA of Spectra')
ax.legend()
#plt.show()

loadings = pca.components_

plt.figure(figsize=(21, 6))

plt.subplot(1, 3, 1)
plt.plot(sample_spectra[0]['wavelengths'], loadings[0], label='PC1')
plt.xlabel('Wavelength')
plt.ylabel('Weight')
plt.title('Weight of Principal Component 1')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(sample_spectra[0]['wavelengths'], loadings[1], label='PC2', color='orange')
plt.xlabel('Wavelength')
plt.ylabel('Weight')
plt.title('Weight of Principal Component 2')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(sample_spectra[0]['wavelengths'], loadings[2], label='PC3', color='green')
plt.xlabel('Wavelength')
plt.ylabel('Weight')
plt.title('Weight of Principal Component 3')
plt.legend()

plt.tight_layout()
#plt.show()

scores = pca.transform(intensity_matrix)

SS = []
CONT = []

# Asegurarse de que el número de clusters no sea mayor que el número de muestras
max_clusters = min(20, len(sample_spectra))

for i in range(1, max_clusters):
    kmeans = KMeans(n_clusters=i, init='k-means++')
    kmeans.fit(intensity_matrix)
    SS.append(kmeans.inertia_)
    CONT.append(i)

plt.plot(CONT, SS, color='red', label='Prediction')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
#plt.show()

num_clusters = 4  #  ajustar el número de clusters según sea necesario
kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=40)
clusters = kmeans.fit_predict(principal_components)


fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')

colors = colormap(np.linspace(0, 1, num_clusters))

for cluster in range(num_clusters):
    cluster_indices = np.where(clusters == cluster)
    ax.scatter(principal_components[cluster_indices, 0], principal_components[cluster_indices, 1], principal_components[cluster_indices, 2], 
               label=f'Cluster {cluster + 1}', color=colors[cluster])

ax.set_xlabel('Principal Component 1 (PC1)')
ax.set_ylabel('Principal Component 2 (PC2)')
ax.set_zlabel('Principal Component 3 (PC3)')
ax.set_title('PCA of Spectra Data with Cluster Highlighting')
ax.legend()
#plt.show()



if aux == 1:
    plt.figure(figsize=(14, 8))
    for spectrum in spectra_data:
        plt.plot(spectrum['wavelengths'], spectrum['intensities'], label=spectrum['file'])
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    plt.title('Comparative Spectrum Data')
    plt.legend()
    #plt.show()

if aux == 2:
    Sum_fwhm_vect = []
    concentrations = []
    cluster_assignments = []
    area_ratios = []

    for i, (sample_file, sample_data) in enumerate(samples.items()):
        x, y, main_peak, _, area_under_peak = process_spectrum(sample_data['wavelengths'], sample_data['intensities'], lower_bound[compound], upper_bound[compound])
        reference_file = ordered_files[i * 2]  # Corresponding reference file
        reference_data = references[reference_file]
        _, _, _, _, ref_area_under_peak = process_spectrum(reference_data['wavelengths'], reference_data['intensities'], lower_bound[compound], upper_bound[compound])

        cluster = clusters[i]
        print(f"Sample File: {sample_file}")
        print(f"Cluster: {cluster + 1}")
        print(f"Main peak at: {x[main_peak] if main_peak is not None else 'N/A'}")
        if area_under_peak and ref_area_under_peak:
            area_ratio = area_under_peak /1 
            print(f"Total Area under Peak: {area_under_peak}")
            print(f"Area Ratio: {area_ratio}")
            area_ratios.append(area_ratio)
            Sum_fwhm_vect.append(area_under_peak)
            cluster_assignments.append(cluster)

            sample_name = os.path.splitext(sample_data['file'])[0]
            concentration = df_concentrations.loc[df_concentrations['Sample'] == compound, sample_name].values[0]
            concentrations.append(concentration)
        else:
            print("Area under peak could not be determined.")
            concentrations.append(None)
            Sum_fwhm_vect.append(None)
            area_ratios.append(None)
        print()

    filtered_area_ratios = [area_ratios[i] for i in range(len(area_ratios)) if area_ratios[i] is not None]
    filtered_concentrations = [concentrations[i] for i in range(len(concentrations)) if area_ratios[i] is not None]
    filtered_clusters = [cluster_assignments[i] for i in range(len(cluster_assignments)) if area_ratios[i] is not None]

    fig, axes = plt.subplots(num_clusters, 1, figsize=(10, 15))

    for cluster in range(num_clusters):
        cluster_indices = [i for i in range(len(filtered_clusters)) if filtered_clusters[i] == cluster]
        cluster_concentrations = [filtered_concentrations[i] for i in cluster_indices]
        cluster_area_ratios = [filtered_area_ratios[i] for i in cluster_indices]

        A = np.vstack([cluster_concentrations, np.ones(len(cluster_concentrations))]).T
        pendiente, intercepto = np.linalg.lstsq(A, cluster_area_ratios, rcond=None)[0]

        intensidades_pred = pendiente * np.array(cluster_concentrations) + intercepto

        ss_tot = np.sum((np.array(cluster_area_ratios) - np.mean(cluster_area_ratios)) ** 2)
        ss_res = np.sum((np.array(cluster_area_ratios) - intensidades_pred) ** 2)
        r_cuadrado = 1 - (ss_res / ss_tot)

        ax = axes[cluster]
        ax.scatter(cluster_concentrations, cluster_area_ratios, label='Experimental Data')
        ax.plot(cluster_concentrations, intensidades_pred, color='red', label='Prediction')
        ax.set_xlabel('Concentration (Mass fractions in %)')
        ax.set_ylabel('Area Ratio (sample/reference)')
        ax.set_title(f'Calibration curve for cluster {cluster + 1}')
        ax.legend()
        ax.grid()

        ax.text(0.05, 0.95, f'Intensidad = {pendiente:.2f} * Concentración + {intercepto:.2f}\\n$R^2$ = {r_cuadrado:.2f}', 
                transform=ax.transAxes, verticalalignment='top')

    plt.tight_layout()
    plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import InterpolatedUnivariateSpline
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
import os

def find_fwhm(x, y, peak_index):
    peak_x = x[peak_index]
    peak_y = y[peak_index]
    half_max = peak_y / 2

    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices] - half_max

    x_sorted, unique_indices = np.unique(x_sorted, return_index=True)
    y_sorted = y_sorted[unique_indices]

    spline = InterpolatedUnivariateSpline(x_sorted, y_sorted)
    roots = spline.roots()

    if len(roots) >= 2:
        left_roots = roots[roots < peak_x]
        right_roots = roots[roots > peak_x]

        if len(left_roots) > 0 and len(right_roots) > 0:
            left_nearest = left_roots[np.argmin(np.abs(left_roots - peak_x))]
            right_nearest = right_roots[np.argmin(np.abs(right_roots - peak_x))]
            fwhm_x = left_nearest, right_nearest
            return fwhm_x
        else:
            return None
    else:
        return None

def process_spectrum(x, y, lower_bound, upper_bound):
    mask = (x >= lower_bound) & (x <= upper_bound)
    x_roi = x[mask]
    y_roi = y[mask]
    peaks, properties = find_peaks(y_roi, prominence=0.1, width=5)
    if len(peaks) == 0:
        return x_roi, y_roi, None, None, None
    main_peak = peaks[np.argmax(y_roi[peaks])]
    fwhm_x = find_fwhm(x_roi, y_roi, main_peak)
    if fwhm_x is not None:
        sum_in_fwhm = np.sum(y_roi[(x_roi >= fwhm_x[0]) & (x_roi <= fwhm_x[1])])
        return x_roi, y_roi, main_peak, fwhm_x, sum_in_fwhm
    else:
        return x_roi, y_roi, main_peak, None, None

# Cargar los datos desde el archivo de Excel
file_path_conc = r'c:\Users\alejo\OneDrive\Desktop\ingeniería física\8vo Brno University\Codes\concentraciones.xlsx' 

# Leer el archivo de Excel
df_concentrations = pd.read_excel(file_path_conc)

aux = int(input("Do you want to do a reference or sample analysis; 1(Ref), 2(Sample): "))

# Directorio de archivos
directory = r'c:\Users\alejo\OneDrive\Desktop\ingeniería física\8vo Brno University\Sergio\Earth\Moon samples'

if aux == 1:
    files = [f for f in os.listdir(directory) if f.startswith('Reference_') and f.endswith('.txt')]

if aux == 2:
    files = [f for f in os.listdir(directory) if not f.startswith('Reference_') and f.endswith('.txt')]
    compound = input("Enter the compound to analyze (e.g., SiO2): ")

spectra_data = []
wavelengths_common = None

for file_name in files:
    file_path = os.path.join(directory, file_name)
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    data = [list(map(float, line.strip().split())) for line in lines]

    df = pd.DataFrame(data)
    df.columns = ['Wavelength'] + [f'Measurement_{i}' for i in range(1, df.shape[1])]
    average_spectrum = df.iloc[:, 1:].mean(axis=1)
    
    # Antes del filtrado
    plt.figure(figsize=(14, 8))

    plt.subplot(2, 1, 1)
    plt.plot(df['Wavelength'], average_spectrum, label='Espectro Promedio (Antes del Filtrado)')
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    plt.title('Espectro Promedio (Antes del Filtrado)')
    plt.legend()

    # Corrección de fondo utilizando el filtro de Savitzky-Golay
    baseline_corrected_spectrum = average_spectrum - savgol_filter(average_spectrum, window_length=11, polyorder=2)

    plt.subplot(2, 1, 2)
    plt.plot(df['Wavelength'], baseline_corrected_spectrum, label='Espectro Promedio (Después del Filtrado)')
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    plt.title('Espectro Promedio (Después del Filtrado)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Detección de picos y filtrado
    peaks, properties = find_peaks(baseline_corrected_spectrum, prominence=0.1, width=5)

    # Identificar las regiones del espectro donde no hay picos presentes
    blank_indices = [i for i in range(len(baseline_corrected_spectrum)) if i not in peaks]
    blank_values = [baseline_corrected_spectrum[i] for i in blank_indices]

    # Calcular la media de las mediciones en blanco (x_bi)
    x_bi = np.mean(blank_values)

    # Calcular la desviación estándar de las mediciones en blanco (s_bi)
    s_bi = np.std(blank_values)

    # Establecer el factor de confianza k (típicamente 3)
    k = 3

    # Calcular el LOD usando la fórmula: LOD = x_bi + k * s_bi
    lod = x_bi + k * s_bi

    # Aplicar el umbral LOD
    corrected_intensity = np.where(baseline_corrected_spectrum >= lod, baseline_corrected_spectrum, 0)

    # Normalización interna al área
    corrected_intensity /= np.sum(corrected_intensity)

    # Añadir la intensidad corregida al DataFrame
    df['Corrected_Intensity'] = corrected_intensity

    # Visualizar el espectro después de la normalización
    plt.figure(figsize=(14, 8))
    plt.plot(df['Wavelength'], corrected_intensity, label='Espectro Normalizado (Después del Filtrado)')
    plt.xlabel('Wavelength')
    plt.ylabel('Corrected Intensity')
    plt.title('Espectro Normalizado (Después del Filtrado)')
    plt.legend()
    plt.show()


    # Convertir DataFrame a matrices NumPy para su procesamiento
    wavelengths_np = df['Wavelength'].to_numpy()
    intensities_np = df['Corrected_Intensity'].to_numpy()
    
    if wavelengths_common is None:
        wavelengths_common = wavelengths_np
    else:
        # Interpolar intensidades a la longitud de onda común
        intensities_np = np.interp(wavelengths_common, wavelengths_np, intensities_np)
    
    spectra_data.append({
        'file': file_name,
        'wavelengths': wavelengths_common,
        'intensities': intensities_np
    })

# Convertir datos de espectros a matriz para PCA
intensity_matrix = np.array([spectrum['intensities'] for spectrum in spectra_data])

# Realizar PCA con 3 componentes
pca = PCA(n_components=3)
principal_components = pca.fit_transform(intensity_matrix)

# Porcentaje de varianza explicada por cada componente
explained_variance = pca.explained_variance_ratio_

# Gráfico 3D de los primeros 3 componentes principales con más colores diferenciados
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')
colors = plt.cm.get_cmap('tab20', len(spectra_data))

for i, spectrum in enumerate(spectra_data):
    ax.scatter(principal_components[i, 0], principal_components[i, 1], principal_components[i, 2], 
               color=colors(i), label=spectrum['file'])

ax.set_xlabel(f'Principal Component 1 ({explained_variance[0]*100:.2f}%)')
ax.set_ylabel(f'Principal Component 2 ({explained_variance[1]*100:.2f}%)')
ax.set_zlabel(f'Principal Component 3 ({explained_variance[2]*100:.2f}%)')
ax.set_title('3D PCA of Spectra')
ax.legend()
plt.show()

# Obtener las cargas de las componentes principales
loadings = pca.components_

# Visualizar las cargas para PC1, PC2 y PC3
plt.figure(figsize=(21, 6))

# Cargas de PC1
plt.subplot(1, 3, 1)
plt.plot(spectra_data[0]['wavelengths'], loadings[0], label='PC1')
plt.xlabel('Wavelength')
plt.ylabel('Weight')
plt.title('Weight of Principal Component 1')
plt.legend()

# Cargas de PC2
plt.subplot(1, 3, 2)
plt.plot(spectra_data[0]['wavelengths'], loadings[1], label='PC2', color='orange')
plt.xlabel('Wavelength')
plt.ylabel('Weight')
plt.title('Weight of Principal Component 2')
plt.legend()

# Cargas de PC3
plt.subplot(1, 3, 3)
plt.plot(spectra_data[0]['wavelengths'], loadings[2], label='PC3', color='green')
plt.xlabel('Wavelength')
plt.ylabel('Weight')
plt.title('Weight of Principal Component 3')
plt.legend()

plt.tight_layout()
plt.show()

scores = pca.transform(intensity_matrix)

SS = []
CONT = []
# Usar KMeans para agrupar los datos
for i in range(1, 20):
    num_clusters = i  # Puedes ajustar el número de clusters según sea necesario
    kmeans = KMeans(n_clusters=i, init='k-means++')
    kmeans.fit(intensity_matrix)
    SS.append(kmeans.inertia_)
    CONT.append(i)

plt.plot(CONT, SS, color='red', label='Prediction')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# Usar KMeans para agrupar los datos
num_clusters = 4  # Puedes ajustar el número de clusters según sea necesario
kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(principal_components)

# Graficar los resultados de PCA con los clusters resaltados
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')

colors = plt.cm.get_cmap('tab20', num_clusters)

for cluster in range(num_clusters):
    cluster_indices = np.where(clusters == cluster)
    ax.scatter(principal_components[cluster_indices, 0], principal_components[cluster_indices, 1], principal_components[cluster_indices, 2], 
               label=f'Cluster {cluster + 1}', color=colors(cluster))

ax.set_xlabel('Principal Component 1 (PC1)')
ax.set_ylabel('Principal Component 2 (PC2)')
ax.set_zlabel('Principal Component 3 (PC3)')
ax.set_title('PCA of Spectra Data with Cluster Highlighting')
ax.legend()
plt.show()

# Gráficos comparativos de espectros
if aux == 1:
    plt.figure(figsize=(14, 8))
    for spectrum in spectra_data:
        plt.plot(spectrum['wavelengths'], spectrum['intensities'], label=spectrum['file'])
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    plt.title('Comparative Spectrum Data')
    plt.legend()
    plt.show()

# Calcular y mostrar métricas de pico principal y FWHM en un rango específico
lower_bound = {"CaO":391.95, "MgO": 279.43, "Al2O3": 308.65, "SiO2": 288.14}
upper_bound = {"CaO":395.66, "MgO": 279.66, "Al2O3": 309.9, "SiO2": 288.2}

if aux == 2:
    Sum_fwhm_vect = []
    concentrations = []

    for spectrum in spectra_data:
        x, y, main_peak, fwhm_x, sum_in_fwhm = process_spectrum(spectrum['wavelengths'], spectrum['intensities'], lower_bound[compound], upper_bound[compound])
        print(f"File: {spectrum['file']}")
        print(f"Main peak at: {x[main_peak] if main_peak is not None else 'N/A'}")
        if fwhm_x:
            print(f"FWHM range: {fwhm_x[0]} - {fwhm_x[1]}")
            print(f"Sum in FWHM: {sum_in_fwhm}")
            Sum_fwhm_vect.append(sum_in_fwhm)

            # Obtener la concentración correspondiente del archivo Excel
            sample_name = os.path.splitext(spectrum['file'])[0]
            concentration = df_concentrations.loc[df_concentrations['Sample'] == compound, sample_name].values[0]
            concentrations.append(concentration)
        else:
            print("FWHM could not be determined.")
            concentrations.append(None)
            Sum_fwhm_vect.append(None)
        print()

    # Filtrar None values
    # Filtrar None values
    filtered_sum_fwhm = [fwhm for fwhm in Sum_fwhm_vect if fwhm is not None]
    filtered_concentrations = [concentrations[i] for i in range(len(concentrations)) if Sum_fwhm_vect[i] is not None]

    # Comprobar si hay suficientes datos para ajustar el modelo
    if len(filtered_concentrations) < 2 or len(filtered_sum_fwhm) < 2:
        print("No hay suficientes datos para ajustar el modelo PLS.")
    else:
        # Preparar los datos para PLS
        X = np.array(filtered_concentrations).reshape(-1, 1)
        y = np.array(filtered_sum_fwhm)

        # Crear y ajustar el modelo PLS
        pls = PLSRegression(n_components=2)
        pls.fit(X, y)

        # Validación cruzada para evaluar la robustez del modelo
        scores = cross_val_score(pls, X, y, cv=5)

        # Imprimir los resultados de la validación cruzada
        print(f'Cross-validated scores: {scores}')
        print(f'Mean cross-validated score: {np.mean(scores)}')

        # Predicciones de intensidades a partir de las concentraciones
        intensidades_pred = pls.predict(X)

        # Graficar la curva de calibración
        plt.scatter(X, y, label='Experimental Data')
        plt.plot(X, intensidades_pred, color='red', label='Prediction')
        plt.xlabel('Concentration (Mass fractions in %)')
        plt.ylabel('Intensity (au)')
        plt.title(f'Calibration Curve for {compound} with PLS')
        plt.legend()
        plt.grid()
        plt.show()

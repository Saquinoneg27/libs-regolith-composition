import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, medfilt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

def load_and_process_spectrum(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = [list(map(float, line.strip().split())) for line in lines]
    df = pd.DataFrame(data)
    df.columns = ['Wavelength'] + [f'Measurement_{i}' for i in range(1, df.shape[1])]
    normalization_factors = df.iloc[:, 1:].sum(axis=0)
    df.iloc[:, 1:] = df.iloc[:, 1:].div(normalization_factors, axis=1)
    average_spectrum = df.iloc[:, 1:].mean(axis=1)
    
   

    max_intensity = np.max(average_spectrum)
    relative_prominence = 0.1 * max_intensity
    peaks, properties = find_peaks(average_spectrum, prominence=relative_prominence, width=0.2)

    mask = np.ones_like(average_spectrum, dtype=bool)
    mask[peaks] = False
    mask2 = ~mask
    background = medfilt(average_spectrum[mask], kernel_size=51)
    background2 = np.mean(background)

    corrected_spectrum = np.copy(average_spectrum)
    corrected_spectrum[mask] = average_spectrum[mask] - background
    corrected_spectrum[mask2] = average_spectrum[mask2] - background2

    corrected_spectrum[corrected_spectrum < 0] = 0

    lod = 0
    corrected_intensity = np.where(corrected_spectrum >= lod, corrected_spectrum, 0)
   

    df['Corrected_Intensity'] = corrected_intensity
    return df['Wavelength'].to_numpy(), df['Corrected_Intensity'].to_numpy()

# Rutas de los archivos
file_path1 = r'C:\Users\alejo\OneDrive\Desktop\ingeniería física\8vo Brno University\Sergio\Earth\Moon samples\LMS-1.txt'
file_path2 = r'C:\Users\alejo\OneDrive\Desktop\ingeniería física\8vo Brno University\Sergio\Earth\Moon samples\LHS-1E.txt'

# Cargar y procesar ambos espectros
wavelengths1, intensities1 = load_and_process_spectrum(file_path1)
wavelengths2, intensities2 = load_and_process_spectrum(file_path2)

# Secciones de 250 unidades
sections = np.arange(200, 1001, 50)
metrics = []

plt.close("all")
for i in range(len(sections) - 1):
    lower_bound = sections[i]
    upper_bound = sections[i + 1]

    # Filtrar los datos dentro de la sección
    mask1 = (wavelengths1 >= lower_bound) & (wavelengths1 < upper_bound)
    mask2 = (wavelengths2 >= lower_bound) & (wavelengths2 < upper_bound)

    if np.any(mask1) and np.any(mask2):
        x1, y1 = wavelengths1[mask1], intensities1[mask1]
        x2, y2 = wavelengths2[mask2], intensities2[mask2]

        # Asegurarse de que ambos espectros estén en el mismo dominio de longitud de onda
        common_wavelengths = np.intersect1d(x1, x2)
        y1_interp = np.interp(common_wavelengths, x1, y1)
        y2_interp = np.interp(common_wavelengths, x2, y2)

        # Diferencia media absoluta
        mad = np.mean(np.abs(y1_interp - y2_interp))

        # Coeficiente de correlación de Pearson
        corr, _ = pearsonr(y1_interp, y2_interp)

        # Área bajo la curva
        area1 = np.trapz(y1_interp, common_wavelengths)
        area2 = np.trapz(y2_interp, common_wavelengths)

        # Distancia Euclidiana
        euclidean_distance = np.linalg.norm(y1_interp - y2_interp)

        metrics.append({
            'Section': f'{lower_bound}-{upper_bound}',
            'MAD': mad,
            'Pearson Correlation': corr,
            'Area Sample 1': area1,
            'Area Sample 2': area2,
            'Euclidean Distance': euclidean_distance
        })

# Convertir las métricas a un DataFrame para facilitar la visualización
metrics_df = pd.DataFrame(metrics)

# Mostrar las métricas
print(metrics_df)

# Gráfica comparativa de las métricas (excluyendo MAD)
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(metrics_df['Section'], metrics_df['Euclidean Distance'], marker='o', label='Euclidean Distance')
plt.ylabel('Difference Metrics')
plt.legend()
plt.title('Comparison of Spectral Sections LMS-1D and LMS-1')

plt.subplot(2, 1, 2)
plt.plot(metrics_df['Section'], metrics_df['Pearson Correlation'], marker='o', label='Pearson Correlation')
plt.xlabel('Wavelength Section')
plt.ylabel('Correlation')
plt.legend()

plt.tight_layout()
plt.show()

# Nueva figura solo para MAD
plt.figure(figsize=(10, 6))
plt.plot(metrics_df['Section'], metrics_df['MAD'], marker='o', label='MAD', color='purple')
plt.xlabel('Wavelength Section')
plt.ylabel('MAD')
plt.title('Mean Absolute Difference (MAD) Across Sections')
plt.legend()
plt.show()

# Crear nueva figura para la visualización de los espectros
plt.figure(figsize=(12, 8))

# Subplot 1: Espectros sobrepuestos
plt.subplot(2, 1, 1)
plt.plot(wavelengths1, intensities1, label='LHS-1D', color='blue')
plt.plot(wavelengths2, intensities2, label='LHS-1', color='red')
plt.xlabel('Wavelength')
plt.ylabel('Intensity(a.u) ')
plt.title('LHS-1D vs LHS-1')
plt.legend()

# Subplot 2: Espectros con offset
offset = 0.001 # Define el valor del offset
plt.subplot(2, 1, 2)
plt.plot(wavelengths1, intensities1, label='LMS-1D', color='blue')
plt.plot(wavelengths2, intensities2 + offset, label=f'LMS-1(Offset by {offset})', color='red')
plt.xlabel('Wavelength')
plt.ylabel('Intensity (a.u) (with Offset)')
plt.title('Spectra with Offset')
plt.legend()

plt.tight_layout()
plt.show()

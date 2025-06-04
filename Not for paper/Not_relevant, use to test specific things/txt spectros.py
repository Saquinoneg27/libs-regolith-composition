import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, medfilt
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit

def gaussian(x, amp, cen, wid):
    """
  This function defines a Gaussian function for curve fitting.

  Args:
      x (numpy.ndarray): Independent variable (wavelengths).
      amp (float): Amplitude of the Gaussian peak.
      cen (float): Center position of the Gaussian peak.
      wid (float): Width of the Gaussian peak.

  Returns:
      numpy.ndarray: Array containing the Gaussian function values for the given parameters.
    """
    return amp * np.exp(-(x - cen)**2 / (2 * wid**2))



def process_spectrum(x, y, lower_bound, upper_bound):
    """
  This function processes a spectrum by finding peaks, fitting the main peak with a Gaussian, 
  and calculating the area under the peak within a specified wavelength range.

  Args:
      x (numpy.ndarray): Array containing independent variable (wavelengths).
      y (numpy.ndarray): Array containing the corresponding spectrum intensity values.
      lower_bound (float): Lower wavelength bound for the region of interest (ROI).
      upper_bound (float): Upper wavelength bound for the ROI.

  Returns:
      tuple: Tuple containing the following elements:
          - x_roi (numpy.ndarray): Wavelengths within the ROI.
          - y_roi (numpy.ndarray): Intensity values within the ROI.
          - main_peak_index (int, optional): Index of the main peak (if found), otherwise None.
          - peak_properties (dict, optional): Properties of the main peak (if found), otherwise None.
          - area_under_peak (float, optional): Total area under the main peak within the ROI (if found), otherwise None.
    """
    mask = (x >= lower_bound) & (x <= upper_bound)
    x_roi = x[mask]
    y_roi = y[mask]
    peaks, properties = find_peaks(y_roi)
    if len(peaks) == 0:
        return x_roi, y_roi, None, None, None
    main_peak = peaks[np.argmax(y_roi[peaks])]
    
    # Fit the main peak with a Gaussian
    try:
        popt, _ = curve_fit(gaussian, x_roi, y_roi, p0=[y_roi[main_peak], x_roi[main_peak], 1])
        y_fit = gaussian(x_roi, *popt)
    except RuntimeError:
        y_fit = y_roi  # In case the fitting fails, fallback to the original data

    # Calculate the total area under the peak
    area_under_peak = np.trapz(y_fit, x_roi)

    
    return x_roi, y_fit, main_peak, None, area_under_peak

# Cargar el archivo
file_path = r'C:\Users\alejo\OneDrive\Desktop\ingeniería física\8vo Brno University\Sergio\Vacuum\Moon samples\LHS-1.txt'
# Leer el contenido del archivo
with open(file_path, 'r') as file:
    lines = file.readlines()

# Procesar las líneas para crear un DataFrame
data = [list(map(float, line.strip().split())) for line in lines]

# Crear el DataFrame
df = pd.DataFrame(data)

# Nombrar las columnas
df.columns = ['Wavelength'] + [f'Measurement_{i}' for i in range(1, df.shape[1])]
# Normalizar cada espectro antes de calcular la media
normalization_factors = df.iloc[:, 1:].sum(axis=0)
df.iloc[:, 1:] = df.iloc[:, 1:].div(normalization_factors, axis=1)

# Calcular el espectro promedio
average_spectrum = df.iloc[:, 1:].mean(axis=1)

# Encontrar picos en el espectro promedio
max_intensity = np.max(average_spectrum)
relative_prominence = 0.1 * max_intensity
peaks, properties = find_peaks(average_spectrum, prominence=relative_prominence, width=0.2)
print(peaks)
# Crear una máscara que sea True en todas partes excepto en los picos
mask = np.ones_like(average_spectrum, dtype=bool)
mask[peaks] = False
mask2 = ~mask
# Suavizar el fondo utilizando un filtro de mediana solo en las regiones fuera de los picos
background = medfilt(average_spectrum[mask], kernel_size=51)
background2 = np.mean(background)

corrected_spectrum= np.copy(average_spectrum)
corrected_spectrum[mask] = average_spectrum[mask] - background
corrected_spectrum[mask2] = average_spectrum[mask2] - background2

#corrected_spectrum = average_spectrum - background
corrected_spectrum[corrected_spectrum < 0] = 0  # Set negative values to zero
# Calcular límites de detección (LOD)
#blank_indices = [i for i in range(len(corrected_spectrum)) if i not in peaks]
#blank_values = [corrected_spectrum[i] for i in blank_indices]
#x_bi = np.mean(blank_values)
#s_bi = np.std(blank_values)
#k = 3
#lod = x_bi + k * s_bi
lod=0

# Corregir la intensidad utilizando el límite de detección
corrected_intensity = np.where(corrected_spectrum >= lod, corrected_spectrum, 0)
#corrected_intensity /= np.sum(corrected_intensity)

df['Corrected_Intensity'] = corrected_intensity
wavelengths_np = df['Wavelength'].to_numpy()
intensities_np = df['Corrected_Intensity'].to_numpy()
    
# Definir el rango de interés
lower_bound =526.9
upper_bound =527.1

# Procesar el espectro en el rango de interés
x_roi_avg, y_roi_avg, main_peak, _, area_under_peak  = process_spectrum(wavelengths_np, intensities_np, lower_bound, upper_bound)

# Crear la figura con dos subplots
plt.figure(figsize=(14, 8))

# Datos originales
plt.subplot(2, 2, 1)
plt.plot(wavelengths_np, average_spectrum, label='Data')
plt.xlabel('Wavelength')
plt.ylabel('Intensity')
plt.title('Spectrum Data')
plt.legend()




# Nueva gráfica con el ajuste gaussiano y el sombreado del área bajo la curva
plt.subplot(2, 2, 4)
plt.plot(x_roi_avg, y_roi_avg, label='Gaussian Fit', color='orange')
plt.fill_between(x_roi_avg, y_roi_avg, color='orange', alpha=0.3, label=f'Area = {area_under_peak:.2e}')
plt.xlabel('Wavelength')
plt.ylabel('Intensity')
plt.title('Gaussian Fit with Area Under Curve')
plt.legend()

plt.tight_layout()
plt.show()

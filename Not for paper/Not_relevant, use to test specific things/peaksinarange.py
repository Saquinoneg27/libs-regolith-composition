import pandas as pd
import numpy as np
from scipy.signal import find_peaks

# Cargar el archivo
file_path = r'C:\Users\alejo\OneDrive\Desktop\ingeniería física\8vo Brno University\Sergio\Earth\Moon samples\LMS-1.txt'


# Leer el contenido del archivo
with open(file_path, 'r') as file:
    lines = file.readlines()

# Procesar las líneas para crear un DataFrame
data = [list(map(float, line.strip().split())) for line in lines]

# Crear el DataFrame
df = pd.DataFrame(data)

# Nombrar las columnas
df.columns = ['Wavelength'] + [f'Measurement_{i}' for i in range(1, df.shape[1])]

# Calcular el espectro promedio
average_spectrum = df.iloc[:, 1:].mean(axis=1)

# Filtrar los datos para el rango de interés (350 a 400 nm)
mask = (df['Wavelength'] >= 370) & (df['Wavelength'] <= 378)
filtered_wavelengths = df['Wavelength'][mask]
filtered_spectrum = average_spectrum[mask]

# Encontrar picos en el espectro promedio filtrado
peaks, properties = find_peaks(filtered_spectrum, prominence=10, width=3)

# Obtener las longitudes de onda de los picos encontrados
peak_wavelengths = filtered_wavelengths.iloc[peaks]

print("Longitudes de onda de los picos entre 350 y 400 nm:")
print(peak_wavelengths)




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, medfilt
import os

plt.close('all')

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
    'TiO2-DL.txt',
]

# Agregar un pequeño offset para poder diferenciar los espectros en la gráfica
offsets = {
    'Earth': 0,
    'Vacuum': 0.001,
    'Mars': 0.004
}

# Asumir que tienes tres condiciones atmosféricas con los nombres correspondientes
conditions = ['Earth', 'Vacuum', 'Mars']
condition_directories = {
    'Earth': r'c:\Users\alejo\OneDrive\Desktop\ingeniería física\8vo Brno University\Sergio\Earth\Moon samples',
    'Vacuum': r'c:\Users\alejo\OneDrive\Desktop\ingeniería física\8vo Brno University\Sergio\Vacuum\Moon samples',
    'Mars': r'c:\Users\alejo\OneDrive\Desktop\ingeniería física\8vo Brno University\Sergio\Mars\Moon samples'
}

# Crear un diccionario para almacenar los espectros procesados por condición
spectra_by_condition = {condition: [] for condition in conditions}

for condition in conditions:
    directory = condition_directories[condition]
    #compound = input("Enter the compound to analyze (e.g., SiO2): ")
    spectra_data = []
    references = {}
    samples = {}
    wavelengths_common = None
    sd_mix = []

    for file_name in ordered_files:
        # Construct the complete file path from directory and filename
        file_path = os.path.join(directory, file_name)
        # Read the file contents line by line
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Convert each line (string) to a list of floats
        data = [list(map(float, line.strip().split())) for line in lines]
        df = pd.DataFrame(data)
        df.columns = ['Wavelength'] + [f'Measurement_{i}' for i in range(1, df.shape[1])]
        # Normalize each measurement column by dividing by the sum of all measurements in that column
        normalization_factors = df.iloc[:, 1:].sum(axis=0)
        df.iloc[:, 1:] = df.iloc[:, 1:].div(normalization_factors, axis=1)
    
        average_spectrum = df.iloc[:, 1:].mean(axis=1)
        # Find peaks in the average spectrum based on prominence and width thresholds
        max_intensity = np.max(average_spectrum)
        relative_prominence = 0.1 * max_intensity
        peaks, properties = find_peaks(average_spectrum, prominence=relative_prominence, width=0.2)
        # Create a mask to identify non-peak regions (where True indicates non-peak)
        mask = np.ones_like(average_spectrum, dtype=bool)
        mask[peaks] = False
        mask2 = ~mask

        background = medfilt(average_spectrum[mask], kernel_size=51)
        background2 = np.mean(background)

        corrected_spectrum= np.copy(average_spectrum)
        corrected_spectrum[mask] = average_spectrum[mask] - background
        corrected_spectrum[mask2] = average_spectrum[mask2] - background2

        #corrected_spectrum = average_spectrum - background
        corrected_spectrum[corrected_spectrum < 0] = 0  # Set negative values to zero

        blank_indices = [i for i in range(len(corrected_spectrum)) if i not in peaks]
        blank_values = [corrected_spectrum[i] for i in blank_indices]
        x_bi = np.mean(blank_values)
        s_bi = np.std(blank_values)
        # Define limit of detection (LOD) based on a factor (k) times the blank standard deviation
        k = 3
        lod = x_bi + k * s_bi

        corrected_intensity = np.where(corrected_spectrum >= lod, corrected_spectrum, 0)
        df['Corrected_Intensity'] = corrected_intensity
        wavelengths_np = df['Wavelength'].to_numpy()
        intensities_np = df['Corrected_Intensity'].to_numpy()
        
        if wavelengths_common is None:
            wavelengths_common = wavelengths_np
        else:
            intensities_np = np.interp(wavelengths_common, wavelengths_np, intensities_np)

        spectra_by_condition[condition].append({
            'file': file_name,
            'wavelengths': wavelengths_common,
            'intensities': intensities_np + offsets[condition]  # Aplicar offset para diferenciarlos
        })

# Crear gráficos individuales para cada archivo
for file_name in ordered_files:
    plt.figure(figsize=(12, 6))
    
    for condition in conditions:
        for spectrum in spectra_by_condition[condition]:
            if spectrum['file'] == file_name:
                plt.plot(spectrum['wavelengths'], spectrum['intensities'], label=f"{condition} - {file_name}")

    plt.xlabel('Wavelength')
    plt.ylabel('Corrected Intensity')
    plt.title(f'Spectra Comparison for {file_name}')
    plt.legend()
    plt.show()
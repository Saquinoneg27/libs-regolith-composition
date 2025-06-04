"""
===============================================================================
LIBS SPECTRAL ANALYSIS FOR QUANTITATIVE CHEMICAL COMPOSITION PREDICTION USING PLS
===============================================================================

PURPOSE:
This code processes Laser-Induced Breakdown Spectroscopy (LIBS) data to create
quantitative calibration models for predicting chemical compound concentrations
in planetary samples (Moon and Mars simulants).

WHAT IT DOES:
1. Reads raw LIBS spectral files (wavelength vs intensity data)
2. Processes and cleans the spectra (noise removal, background correction)
3. Extracts relevant wavelength regions for specific compounds
4. Matches spectral data with known concentrations from reference samples
5. Trains a machine learning model (PLS regression) to predict concentrations
6. Validates model performance using cross-validation

INPUT FILES REQUIRED:
- Spectral data files (.txt): Contains wavelength and intensity measurements
  Format: [Wavelength, Measurement_1, Measurement_2, ..., Measurement_n]
- Concentration file (.xlsx): Contains known compound concentrations for samples
  Format: Rows=compounds, Columns=sample names, Values=concentrations

SUPPORTED COMPOUNDS & WAVELENGTH RANGES:
- SiO2: 288.0-288.4 nm
- CaO: 373.6-373.9, 393.0-393.8, 422.55-422.85 nm  
- MgO: 285.0-285.45 nm
- TiO2: 444.2-446.2 nm
- Al2O3: 394.2-394.6 nm
- K2O: 769.5-770.25, 766.0-767.0 nm
- Fe2O3: 247.6-249.6 nm

PROCESSING STEPS:
1. Data Loading: Reads spectral files and concentration data
2. Quality Control: Removes measurements with high variability (outliers)
3. Spectrum Averaging: Creates representative spectrum per sample
4. Background Correction: Removes baseline using peak detection and median filtering
5. Wavelength Filtering: Extracts compound-specific spectral regions
6. Data Standardization: Normalizes features for machine learning
7. Model Training: Uses Partial Least Squares (PLS) regression
8. Optimization: Finds optimal number of PLS components via cross-validation
9. Validation: Evaluates model performance using R² and Mean Absolute Error

OUTPUT:
- Optimal number of PLS components for the model
- Cross-validated R² score (explained variance)
- Cross-validated Mean Absolute Error (MAE)
- Performance plot showing MAE across validation folds
- Trained calibration model ready for predicting unknown samples

USAGE:
0. Be sure that you have the paths to the files correctly set.
1. Run the script
2. Select sample type (1=Moon, 2=Mars, 3=All)
3. Enter target compound (e.g., SiO2, CaO, MgO)
4. The code will automatically process all data and display results

REQUIREMENTS:
- pandas, numpy, sklearn, matplotlib, scipy
- Proper file structure with spectral data and concentration files
- Consistent naming convention between spectral files and concentration data

AUTHORS: Sergio Quiñónez and Jakub Buday
CREATED: August 8, 2024
===============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os     
from scipy.signal import find_peaks, medfilt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error


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
    peaks, properties = find_peaks(y_roi, prominence=0.00005)
    if len(peaks) == 0:
        return x_roi, y_roi, None, None, None
    
    main_peak = peaks[np.argmax(y_roi[peaks])]
    
    # Fit the main peak with a Gaussian
    try:
        popt, _ = curve_fit(gaussian, x_roi, y_roi, p0=[y_roi[main_peak], x_roi[main_peak], 1])
        y_fit = gaussian(x_roi, *popt)
    except RuntimeError:
        y_fit = y_roi

    # Calculate the total area under the peak
    area_under_peak = np.trapz(y_fit, x_roi)

    return x_roi, y_fit, main_peak, None, area_under_peak

plt.close('all')
# Load concentration data
file_path_conc = r'C:\Users\alejo\OneDrive\Desktop\ingeniería física\8vo Brno University\Spectra\Concentrations.xlsx' 
df_concentrations = pd.read_excel(file_path_conc)

# Set parameters based on user input
aux = int(input("Choose the kind of samples: 1: Moon samples, 2: Mars samples , 3: All samples:"))
compound = input("Enter the compound to analyze (e.g., SiO2): ")

chosen_condition = "Earth"

directory_map = {
    1: r'c:\Users\alejo\OneDrive\Desktop\ingeniería física\8vo Brno University\Spectra\{chosen_condition}\Moon Samples',
    2: r'c:\Users\alejo\OneDrive\Desktop\ingeniería física\8vo Brno University\Spectra\{chosen_condition}\Mars samples',
    3: r'c:\Users\alejo\OneDrive\Desktop\ingeniería física\8vo Brno University\Spectra\{chosen_condition}\All_of_them'
}
ordered_files_map = {
    1: [ 'Dusty.txt',  'LHS-1D.txt', 'K2O-AL.txt',
         'Na2O-AL.txt',  'MgO-AL.txt',  'SiO2-DL.txt',
        'SiO2-AL.txt',  'TiO2-AL.txt',  'MgO-CL.txt',
        'TiO2-CL.txt', 'TiO2-DL.txt'],
    
    2: [ 'MGS-1.txt', 'Na2O-BM.txt',  'Al2O3-BM.txt',
         'K2O-CM.txt',  'K2O-BM.txt', 'K2O-DM.txt',
         'MgO-AM.txt',  'MgO-BM.txt',  'SiO2-AM.txt',
         'SiO2-CM.txt', 'Al2O3-BM.txt', 'CaO-AM.txt'],
    3: [ 'Dusty.txt',  'LHS-1D.txt', 'K2O-AL.txt',
         'Na2O-AL.txt',  'MgO-AL.txt',  'SiO2-DL.txt',
        'SiO2-AL.txt',  'TiO2-AL.txt',  'MgO-CL.txt',
        'TiO2-CL.txt', 'TiO2-DL.txt',

         'MGS-1.txt', 'Na2O-BM.txt',  'Al2O3-BM.txt',
        'K2O-CM.txt',  'K2O-BM.txt', 'K2O-DM.txt',
        'MgO-AM.txt',  'MgO-BM.txt',  'SiO2-AM.txt',
        'SiO2-CM.txt', 'Al2O3-BM.txt', 'CaO-AM.txt']
}

directory = directory_map.get(aux, "")
ordered_files = ordered_files_map.get(aux, [])

# Define wavelength bounds
lower_bound = {"CaO": [373.6, 393, 422.55], "MgO": [285], "Al2O3": [394.2], "SiO2": [288], "TiO2": [444.2], "Fe2O3": [247.6], "K2O": [769.5, 766]}
upper_bound = {"CaO": [373.9, 393.8, 422.85], "MgO": [285.45], "Al2O3": [394.6], "SiO2": [288.4], "TiO2": [446.2], "Fe2O3": [249.6], "K2O": [770.25, 767]}

# Initialize variables
spectra_data = []
references = {}
samples = {}
wavelengths_common = None
proportion_to_remove = 0.2

# Process each file
for file_name in ordered_files:
    file_path = os.path.join(directory, file_name)
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = [list(map(float, line.strip().split())) for line in lines]
    df = pd.DataFrame(data, columns=['Wavelength'] + [f'Measurement_{i}' for i in range(1, len(data[0]))])

    # Normalize each column
    normalization_factors = df.iloc[:, 1:].sum(axis=0)
    df.iloc[:, 1:] = df.iloc[:, 1:].div(normalization_factors, axis=1)

    # Remove columns with highest standard deviation
    std_devs = df.iloc[:, 1:].std(axis=0)
    num_to_remove = max(1, int(len(std_devs) * proportion_to_remove))
    top_std_columns = std_devs.nlargest(num_to_remove).index
    df.drop(columns=top_std_columns, inplace=True)

    average_spectrum = df.iloc[:, 1:].mean(axis=1)
    max_intensity = np.max(average_spectrum)
    relative_prominence = 0.01 * max_intensity
    peaks, _ = find_peaks(average_spectrum, prominence=relative_prominence, width=0.1)

    mask = np.ones_like(average_spectrum, dtype=bool)
    mask[peaks] = False
    background = medfilt(average_spectrum[mask], kernel_size=51)
    background_mean = np.mean(background)

    corrected_spectrum = np.copy(average_spectrum)
    corrected_spectrum[mask] -= background
    corrected_spectrum[~mask] -= background_mean
    corrected_spectrum[corrected_spectrum < 0] = 0

    blank_values = corrected_spectrum[mask]
    lod = np.mean(blank_values) + 3 * np.std(blank_values)
    lod=0
    corrected_intensity = np.where(corrected_spectrum >= lod, corrected_spectrum, 0)
    df['Corrected_Intensity'] = corrected_intensity

    wavelengths_np = df['Wavelength'].to_numpy()
    intensities_np = df['Corrected_Intensity'].to_numpy()

    if wavelengths_common is None:
        wavelengths_common = wavelengths_np
    else:
        intensities_np = np.interp(wavelengths_common, wavelengths_np, intensities_np)

    mask_range = (wavelengths_common >= lower_bound[compound][0]) & (wavelengths_common <= upper_bound[compound][0])
    wavelengths_filtered = wavelengths_common[mask_range]
    intensities_filtered = intensities_np[mask_range]

    spectrum_data = {
        'file': file_name,
        'wavelengths': wavelengths_filtered,
        'intensities': intensities_filtered
    }

    if 'Reference' in file_name:
        references[file_name] = spectrum_data
    else:
        samples[file_name] = spectrum_data

    spectra_data.append(spectrum_data)

# Match spectra data with concentration data
if df_concentrations.index.name != 'Sample':
    df_concentrations.set_index('Sample', inplace=True)

target_variable = []
ordered_spectra_data = []

for spectrum in spectra_data:
    file_name = spectrum['file'].split('/')[-1].replace('.txt', '')

    if file_name in df_concentrations.columns:
        try:
            target_variable.append(df_concentrations.loc[compound, file_name])
            ordered_spectra_data.append(spectrum)
        except KeyError as e:
            print(f"KeyError: {e} - Ensure the target element and sample names match correctly.")

target_variable = np.array(target_variable)
intensity_matrix = np.array([spectrum['intensities'] for spectrum in ordered_spectra_data])

# Standardize data
scaler = StandardScaler()
intensity_matrix_scaled = scaler.fit_transform(intensity_matrix)


# Inicializar variables para determinar el mejor número de componentes
kf = KFold(n_splits=5, shuffle=True, random_state=42)
component_range = range(1, min(intensity_matrix_scaled.shape[1], 10))
best_n_components = 0
best_r2_score = -np.inf
mae_scores_dict = {}

# Búsqueda del mejor número de componentes y cálculo del MAE en el mismo proceso
for n_components in component_range:
    pls = PLSRegression(n_components=n_components)
    
    fold_r2_scores = []
    fold_mae_scores = []
    
    for train_index, test_index in kf.split(intensity_matrix_scaled):
        X_train, X_test = intensity_matrix_scaled[train_index], intensity_matrix_scaled[test_index]
        y_train, y_test = target_variable[train_index], target_variable[test_index]
        
        pls.fit(X_train, y_train)
        y_pred = pls.predict(X_test)
        
        # Calcular R^2 para este pliegue
        fold_r2_scores.append(r2_score(y_test, y_pred))
        
        # Calcular el MAE para este pliegue
        mae_fold = mean_absolute_error(y_test, y_pred)
        fold_mae_scores.append(mae_fold)
    
    # Promedio del R^2 y MAE para este número de componentes
    mean_cv_r2 = np.mean(fold_r2_scores)
    mean_cv_mae = np.mean(fold_mae_scores)
    mae_scores_dict[n_components] = fold_mae_scores
    
    print(f'n_components={n_components}, Mean R^2={mean_cv_r2}, Mean MAE={mean_cv_mae}')
    
    if mean_cv_r2 > best_r2_score:
        best_r2_score = mean_cv_r2
        best_n_components = n_components

print(f'Best number of components: {best_n_components} with cross-validated R^2: {best_r2_score}')

# Obtención del MAE para el mejor número de componentes
best_n_components_mae_scores = mae_scores_dict[best_n_components]
cross_validated_mae = np.mean(best_n_components_mae_scores)
print(f'Cross-Validated MAE for best model: {cross_validated_mae}')

# Visualización del MAE a través de los pliegues para el mejor modelo
plt.plot(best_n_components_mae_scores, marker='o', linestyle='-', color='b')
plt.axhline(y=cross_validated_mae, color='r', linestyle='--', label=f'Mean MAE: {cross_validated_mae:.4f}')
plt.xlabel('Fold')
plt.ylabel('MAE (Mean Absolute Error)')
plt.title(f'MAE per Fold for Best Model (n_components={best_n_components})')
plt.legend()
plt.show()

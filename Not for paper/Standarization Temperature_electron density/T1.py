# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 09:36:05 2024

@author: alejo
"""
from scipy.stats import linregress
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import  KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import os     
from scipy.signal import find_peaks, medfilt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import CubicSpline
from sklearn.linear_model import RANSACRegressor, LinearRegression




def weighted_least_squares(E_values, log_term, weights):
    """
    Realiza un ajuste de mínimos cuadrados ponderados.

    Args:
        E_values (numpy array): Energías de nivel (eV).
        log_term (numpy array): Término logarítmico ln(Iλ / gA).
        weights (numpy array): Pesos correspondientes a cada punto de datos.

    Returns:
        slope, intercept: Pendiente e intercepto de la línea ajustada.
    """
    # Ajuste ponderado usando np.polyfit con weights
    slope, intercept = np.polyfit(E_values, log_term, 1, w=weights)
    return slope, intercept


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
    
    # Ensure the mask length matches y length
    if len(x) != len(y):
        print(f"Warning: Length of x ({len(x)}) does not match length of y ({len(y)}). Skipping this file.")
        return None, None, None, None, None, None
    
    
    x_roi = x[mask]
    y_roi = y[mask]
    
    if len(y_roi) == 0:
        return x_roi, y_roi, None, None, None, None

    peaks, properties = find_peaks(y_roi, prominence=0.00005)
    if len(peaks) == 0:
        return x_roi, y_roi, None, None, None, None
    
    main_peak = peaks[np.argmax(y_roi[peaks])]
    
    # Fit the main peak with a Gaussian
    try:
        popt, _ = curve_fit(gaussian, x_roi, y_roi, p0=[y_roi[main_peak], x_roi[main_peak], 1])
        y_fit = gaussian(x_roi, *popt)
    except RuntimeError:
        y_fit = y_roi  # In case the fitting fails, fallback to the original data

    # Calculate the total area under the peak
    area_under_peak = np.trapz(y_fit, x_roi)
    fwhm_x = find_fwhm(x_roi, y_roi, main_peak)
    
    
    return x_roi, y_fit, main_peak, None, area_under_peak, fwhm_x

def get_actual_temp(intensities_Ca, file_name, fwhm,errors):
    # Datos espectroscópicos del calcio
    wavelengths = np.array([393.366 , 396.8,317.933, 854.2, 866.2, 373.690 ])  # nm
    
    temperaturas_393_366 = np.array([11400, 11600, 12240, 13000, 13350, 16000, 17500, 19000, 25100, 28000, 29200, 30000])
    w_values_393_366 = np.array([0.039, 0.079, 0.0914, 0.235, 0.180, 0.16, 10.0, 0.172, 0.22, 0.25, 0.18, 0.24])
    # Crear la función de interpolación
    interpolacion_w_393_366 = CubicSpline(temperaturas_393_366, w_values_393_366)
    
    temperaturas_3968 = np.array([7450, 12240, 13000, 13350, 16000, 17500, 18560, 25100, 28000])
    w_values_3968 = np.array([0.210, 0.0846, 0.235, 0.161, 0.16, 10.3, 0.188, 0.20, 0.25])
    
    # Crear la función de interpolación para 3968.47 Å
    interpolacion_w_3968 = CubicSpline(temperaturas_3968, w_values_3968)
    
    # Para la línea 3736.20 Å
    temperaturas_3736 = np.array([7500, 10000, 13000, 25100])
    w_values_3736 = np.array([18.2, 0.69, 0.79, 0.30])
    
    # Crear la función de interpolación para 3736.20 Å
    interpolacion_w_3736 = CubicSpline(temperaturas_3736, w_values_3736)
    
    
    intensities_Ca = np.array(intensities_Ca)
    fwhm = np.array(fwhm)
    
    # Valores de degeneración g y energías de nivel E (en eV)
    g_values = np.array([2, 2,4, 6, 4,2])
    E_values = np.array([3.150984, 3.123349,7.049551, 3.150984, 3.123349,6.4678 ])  # Energías de nivel correspondientes en eV

    # Constante de probabilidad de transición A
    A_values = np.array([1.47e8, 1.4e8,3.6e8, 9.9e06, 1.06e07, 1.7e8])

    # Constante de Boltzmann en eV/K
    k_B = 8.617333262145e-5
    
    errors = np.array(errors)

    # Filtrar valores de intensities_Ca que son cero
    valid_indices = intensities_Ca > 0
    filtered_intensities_Ca = intensities_Ca[valid_indices]
    filtered_wavelengths = wavelengths[valid_indices]
    filtered_g_values = g_values[valid_indices]
    filtered_E_values = E_values[valid_indices]
    filtered_A_values = A_values[valid_indices]
    filtered_errors = errors[valid_indices]

    # Cálculo del término logarítmico
    log_term = np.log(filtered_intensities_Ca * filtered_wavelengths / (filtered_g_values * filtered_A_values))
    
    # Si tienes errores de medida, úsalo para calcular pesos

    weights = 1 / filtered_errors
    
    print(filtered_E_values,"wafag", log_term, "WAWWWW",weights )
    slope, intercept = weighted_least_squares(filtered_E_values, log_term, weights)

   
   # Ahora continúa con el cálculo de la temperatura usando el slope calculado

    # Gráfico de Boltzmann
    plt.figure(figsize=(8, 6))
    plt.plot(filtered_E_values, log_term, 'bo', label='Experimental data')
    #slope, intercept, r_value, p_value, std_err = linregress(filtered_E_values, log_term)
    plt.plot(filtered_E_values, slope * filtered_E_values + intercept, 'r-', label=f'Linear fit: T = {-1/(slope*k_B):.2f} K')
    plt.xlabel('Level energy E (eV)')
    plt.ylabel('ln(Iλ / gA)')
    plt.title('Boltzmann plot')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Temperatura calculada
    T = -1 / (slope * k_B)
    
    
    print(f'plasma temperature for {file_name}: {T:.2f} K')
    fwhm_values = [(end - start)*10 for start, end in fwhm_vect]
    w = np.array([interpolacion_w_393_366(T),interpolacion_w_3968(T),0.66,0.95,0.95,interpolacion_w_3736(T) ])
    
    ne = np.mean(fwhm_values/(2*w))
    
    return T ,ne

    
def standardize_spectrum(temperature, ne, I_obs, T_std, ne_std):
    """
    Estandariza el espectro ajustándolo a un estado de plasma estándar.

    Args:
        temperature (float): Temperatura del plasma medida.
        ne (float): Densidad de electrones medida.
        I_obs (numpy array): Intensidades observadas.
        T_std (float): Temperatura estándar de referencia.
        ne_std (float): Densidad de electrones estándar de referencia.

    Returns:
        numpy array: Intensidades estandarizadas.
    """
    # Ajuste basado en la razón de las temperaturas
    T_factor = (temperature / T_std)
    ne_factor = (ne / ne_std)

    # Ajuste de las intensidades
    I_std = I_obs * T_factor * ne_factor
    return I_std

    
    

# Load concentration data.
plt.close("all")
file_path_conc = r'C:\Users\alejo\OneDrive\Desktop\ingeniería física\8vo Brno University\Spectra\Concentrations.xlsx' 
df_concentrations = pd.read_excel(file_path_conc)

# Set parameters based on user input
aux = int(input("Choose the kind of samples: 1: Moon samples, 2: Mars samples , 3: All samples:"))
compound = input("Enter the compound to analyze (e.g., SiO2): ")

# Define the chosen condition
chosen_condition = "Mars"

# Create the directory map using the chosen condition
directory_map = {
    1: fr'c:\Users\alejo\OneDrive\Desktop\ingeniería física\8vo Brno University\Spectra\{chosen_condition}\Moon Samples',
    2: fr'c:\Users\alejo\OneDrive\Desktop\ingeniería física\8vo Brno University\Spectra\{chosen_condition}\Mars samples',
    3: fr'c:\Users\alejo\OneDrive\Desktop\ingeniería física\8vo Brno University\Spectra\{chosen_condition}\All_of_them'
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
lower_bound = {"CaO":  [ 393,396.6,317.8, 853.9,865.8,373.56],  "MgO": [279.4, 280, 285], "Al2O3": [394.2], "SiO2": [288], "TiO2": [444.2], "Na2O": [589.4], "K2O": [769.5]}
upper_bound = {"CaO": [ 393.8, 397.1,318.3,854.6, 866.6,373.97 ], "MgO": [279.7, 280.5, 285.45], "Al2O3": [394.6], "SiO2": [288.4], "TiO2": [446.2], "Na2O": [589.8], "K2O": [770.25]}


#lower_bound = {"CaO": [ 393,396.6,317.8, 853.9,865.8 ], "MgO": [279.4, 280, 285], "Al2O3": [309.15, 394.2,396], "SiO2": [288, 251.55], "TiO2": [444.2], "Na2O": [589.4, 588.7], "K2O": [769.5,766]}
#upper_bound = {"CaO": [ 393.8, 397.1,318.3,854.6, 866.6 ], "MgO": [279.7, 280.5, 285.45], "Al2O3": [309.5,394.6,396.35], "SiO2": [288.4,251.75], "TiO2": [446.2], "Na2O": [589.8, 589.3], "K2O": [770.25, 767]}
# Initialize variables

spectra_data = []
references = {}
samples = {}
wavelengths_common = None
proportion_to_remove = 0.1
plt.close("all")



plasma_parameters = {
    "Earth": {"T_std": 15900, "ne_std": 1.29e17},  # Standard values for Earth atmosphere
    "Vacuum": {"T_std": 15000, "ne_std": 0.8e17},  # Standard values for vacuum
    "Mars": {"T_std": 18900, "ne_std": 2.17e17}  # Standard values for Mars atmosphere

}

T_std = plasma_parameters[chosen_condition]["T_std"]
ne_std = plasma_parameters[chosen_condition]["ne_std"]



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
    
    area_Ca = []
    errors =[]
    
    if len(df['Wavelength'].to_numpy())!=  len(df.iloc[:, 1].to_numpy()):
        continue  # Omitir este archivo y continuar con el siguiente
    
    errors = []
    for i, (lower, upper) in enumerate(zip(lower_bound["CaO"], upper_bound["CaO"])):
        area_measurements = []
        for j in range(1, df.shape[1]):
            x_i, y_i, main_peak, _, area, _ = process_spectrum(df['Wavelength'].to_numpy(), df.iloc[:, j].to_numpy(), lower, upper)
            area_measurements.append(area if area is not None else 0)
    
        area_Ca.append(np.mean(area_measurements))
        errors.append(np.std(area_measurements) if len(area_measurements) > 1 else 0)
            
            
    
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
    intensities_Ca= []
    fwhm_vect = []

    if wavelengths_common is None:
        wavelengths_common = wavelengths_np
    else:
        intensities_np = np.interp(wavelengths_common, wavelengths_np, intensities_np)
        
    for lower, upper in zip(lower_bound["CaO"], upper_bound["CaO"]):   
        x, y, main_peak, _, area_under_peak, fwhm = process_spectrum(wavelengths_np,intensities_np, lower, upper)
   
        intensities_Ca.append(area_under_peak if area_under_peak is not None else 0)
        fwhm_vect.append (fwhm if fwhm is not None else (0,0))
    
   
    T,ne = get_actual_temp(intensities_Ca,file_name, fwhm, errors)
    
    if T <= 0:
        print(f"Omitiendo muestra {file_name} debido a una temperatura no válida: {T:.2f} K")
        continue  # Saltar al siguiente archivo si la temperatura es negativa
    
    I_std = standardize_spectrum(T, ne, intensities_np, T_std, ne_std)
    
    

    # Inicializa la máscara completa como un array de falsos
    mask_range = np.zeros_like(wavelengths_common, dtype=bool)
    
    
    for lower, upper in zip(lower_bound[compound], upper_bound[compound]):
        # Crea una máscara para cada rango y la combina con la máscara completa
        mask_range |= (wavelengths_common >= lower) & (wavelengths_common <= upper)
    
    
    
    # Aplica la máscara para filtrar las longitudes de onda e intensidades
    wavelengths_filtered = wavelengths_common[mask_range]
    intensities_filtered = intensities_np[mask_range]

    spectrum_data = {
        'file': file_name,
        'wavelengths': wavelengths_common,
        'intensities': I_std,
        "Temperature": T,
        "n_e" : ne
    }

    if 'Reference' in file_name:
        references[file_name] = spectrum_data
    else:
        samples[file_name] = spectrum_data

    spectra_data.append(spectrum_data)
    
    
# First, ensure that the 'Sample' column is set as the index if not already done
if df_concentrations.index.name != 'Sample':
    df_concentrations.set_index('Sample', inplace=True)

# Now, try to access the data for SiO2 and align with your spectra data

target_element = compound  # Specify the target element or compound

target_variable = []
ordered_spectra_data = []

for spectrum in spectra_data:
    # Extract the filename and remove the path and extension
    file_name = spectrum['file'].split('/')[-1].replace('.txt', '')

    if file_name in df_concentrations.columns:
        try:
            # Extract the concentration for the target element from the row and sample (file_name) from the column
            target_variable.append(df_concentrations.loc[target_element, file_name])
            
            # Append the spectra data in the same order
            ordered_spectra_data.append(spectrum)
        except KeyError as e:
            print(f"KeyError: {e} - Ensure the target element and sample names match correctly.")


# Convert target_variable to a numpy array for further processing
target_variable = np.array(target_variable)

# Prepare the intensity matrix for PCA
intensity_matrix = np.array([spectrum['intensities'] for spectrum in ordered_spectra_data])

# Estandarización
scaler_standard = StandardScaler()
intensity_matrix_standardized = scaler_standard.fit_transform(intensity_matrix)

# Normalización
scaler_minmax = MinMaxScaler()
intensity_matrix_normalized = scaler_minmax.fit_transform(intensity_matrix_standardized)

# Perform KFold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Range of components to test
component_range = range(1, min(intensity_matrix_normalized.shape[1], 10))  # Adjust the upper limit based on your data

best_n_components = 0
best_r2_score = -np.inf
best_mae_score = np.inf  # Initialize to a large value since lower MAE is better
best_alpha = None  # Variable to store the best alpha value

# Set up the parameter grid for alpha in Ridge regression
param_grid = {'alpha': np.logspace(-10, 10, 200)}

# Loop through different numbers of PCA components to find the optimal number
for n_components in component_range:
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(intensity_matrix_normalized)
    
    # Initialize GridSearchCV with Ridge regression to find the best alpha
    grid_search = GridSearchCV(Ridge(), param_grid, cv=kf, scoring='r2')
    grid_search.fit(principal_components, target_variable)
    
    # Get the best alpha value for this number of components
    alpha = grid_search.best_params_['alpha']
 
    
    # Get the R² score for the best alpha
    cv_r2 = cross_val_score(Ridge(alpha=alpha), principal_components, target_variable, cv=kf, scoring='r2')
    mean_cv_r2 = np.mean(cv_r2)

    # Perform cross-validation manually to capture MAE as well
    mae_scores = []
    for train_index, test_index in kf.split(principal_components):
        X_train, X_test = principal_components[train_index], principal_components[test_index]
        y_train, y_test = target_variable[train_index], target_variable[test_index]
        
        model = Ridge(alpha=alpha, positive=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae_scores.append(mean_absolute_error(y_test, y_pred))
    
    mean_cv_mae = np.mean(mae_scores)

    print(f'n_components={n_components}, Best Alpha={alpha}, Mean R^2={mean_cv_r2}, Mean MAE={mean_cv_mae}')

    if mean_cv_r2 > best_r2_score:
        best_r2_score = mean_cv_r2
        best_n_components = n_components
        best_mae_score = mean_cv_mae
        best_alpha = alpha  # Store the best alpha value

print(f'Best number of components: {best_n_components} with cross-validated R^2: {best_r2_score}, MAE: {best_mae_score}, and Alpha: {best_alpha}')


T_T = []
for i in range(1,len(spectra_data)):
    print(i, spectra_data[i]["Temperature"])
    T_T.append(spectra_data[i]["Temperature"])
    
print( "Mean temperature of the plasma: (K) ", np.mean(T_T))



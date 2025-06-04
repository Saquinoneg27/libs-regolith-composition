import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import os     
from scipy.signal import find_peaks, medfilt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.metrics import mean_squared_error


def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x - cen)**2 / (2 * wid**2))




# Load concentration data
file_path_conc = r'C:\Users\alejo\OneDrive\Desktop\ingeniería física\8vo Brno University\Spectra\Concentrations.xlsx' 
df_concentrations = pd.read_excel(file_path_conc)

# Set parameters based on user input
aux = int(input("Choose the kind of samples: 1: Moon samples, 2: Mars samples , 3: All samples:"))
compound = input("Enter the compound to analyze (e.g., SiO2): ")
# Define the chosen condition
chosen_condition = "Earth"

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
lower_bound = {"CaO":  [ 393,396.6,317.8, 853.9,865.8 ],  "MgO": [279.4, 280, 285], "Al2O3": [394.2], "SiO2": [288], "TiO2": [444.2], "Na2O": [589.4], "K2O": [769.5]}
upper_bound = {"CaO": [ 393.8, 397.1,318.3,854.6, 866.6 ], "MgO": [279.7, 280.5, 285.45], "Al2O3": [394.6], "SiO2": [288.4], "TiO2": [446.2], "Na2O": [589.8], "K2O": [770.25]}


#lower_bound = {"CaO": [ 393,396.6,317.8, 853.9,865.8 ], "MgO": [279.4, 280, 285], "Al2O3": [309.15, 394.2,396], "SiO2": [288, 251.55], "TiO2": [444.2], "Na2O": [589.4, 588.7], "K2O": [769.5,766]}
#upper_bound = {"CaO": [ 393.8, 397.1,318.3,854.6, 866.6 ], "MgO": [279.7, 280.5, 285.45], "Al2O3": [309.5,394.6,396.35], "SiO2": [288.4,251.75], "TiO2": [446.2], "Na2O": [589.8, 589.3], "K2O": [770.25, 767]}


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
    #lod = np.mean(blank_values) + 3 * np.std(blank_values)
    lod=0
    corrected_intensity = np.where(corrected_spectrum >= lod, corrected_spectrum, 0)
    df['Corrected_Intensity'] = corrected_intensity

    wavelengths_np = df['Wavelength'].to_numpy()
    intensities_np = df['Corrected_Intensity'].to_numpy()

    if wavelengths_common is None:
        wavelengths_common = wavelengths_np
    else:
        intensities_np = np.interp(wavelengths_common, wavelengths_np, intensities_np)

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
        'wavelengths': wavelengths_filtered,
        'intensities': intensities_filtered
    }

    if 'Reference' in file_name:
        references[file_name] = spectrum_data
    else:
        samples[file_name] = spectrum_data

    spectra_data.append(spectrum_data)

# First, ensure that the 'Sample' column is set as the index if not already done
if df_concentrations.index.name != 'Sample':
    df_concentrations.set_index('Sample', inplace=True)

# Now, try to access the data for COMPOUND and align with your spectra data

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
component_range = range(1, min(intensity_matrix_normalized.shape[1], 12))  # Adjust the upper limit based on your data

best_n_components = 0
best_r2_score = -np.inf
best_rmse_p_score = np.inf  # Initialize to a large value since lower RMSE is better
best_alpha = None  # Variable to store the best alpha value

# Set up the parameter grid for alpha in Ridge regression
param_grid = {'alpha': np.logspace(-3, 3, 10)}

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

    # Perform cross-validation manually to capture RMSE-P as well
    rmse_p_scores = []
    for train_index, test_index in kf.split(principal_components):
        X_train, X_test = principal_components[train_index], principal_components[test_index]
        y_train, y_test = target_variable[train_index], target_variable[test_index]
        
        model = Ridge(alpha=alpha, positive=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse_p_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
    
    mean_cv_rmse_p = np.mean(rmse_p_scores)

    print(f'n_components={n_components}, Best Alpha={alpha}, Mean R^2={mean_cv_r2}, Mean RMSE-P={mean_cv_rmse_p}')

    if mean_cv_r2 > best_r2_score:
        best_r2_score = mean_cv_r2
        best_n_components = n_components
        best_rmse_p_score = mean_cv_rmse_p
        best_alpha = alpha  # Store the best alpha value

print(f'Best number of components: {best_n_components} with cross-validated R^2: {best_r2_score}, RMSE-P: {best_rmse_p_score}, and Alpha: {best_alpha}')

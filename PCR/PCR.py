"""
===============================================================================
LIBS SPECTRAL ANALYSIS FOR QUANTITATIVE CHEMICAL COMPOSITION PREDICTION USING PCR
===============================================================================

PURPOSE:
This code processes Laser-Induced Breakdown Spectroscopy (LIBS) data  to create quantitative 
calibration models for predicting chemical compound concentrations in planetary 
samples (Moon and Mars simulants).

WHAT IT DOES:
1. Reads raw LIBS spectral files containing  wavelength vs intensity data
2. Processes and cleans the entire spectra (noise removal, background correction)
3. Uses Principal Component Analysis (PCA) to reduce dimensionality of full spectrum
4. Matches spectral data with known concentrations from reference samples
5. Trains a Ridge regression model on PCA components to predict concentrations
6. Optimizes both PCA components and Ridge alpha parameter via cross-validation
7. Validates model performance using R² and Mean Absolute Error (MAE)


INPUT FILES REQUIRED:
1. Spectral data files (.txt): Contains wavelength and intensity measurements
   Format: [Wavelength, Measurement_1, Measurement_2, ..., Measurement_n]
   Location: Modify the 'directory_map' paths below to point to your spectral files

2. Concentration file (.xlsx): Contains known compound concentrations for samples
   Format: Rows=compounds, Columns=sample names, Values=concentrations
   Location: Modify 'file_path_conc' variable below

CRITICAL PATHS TO MODIFY:
- file_path_conc: Path to your Excel concentration file
- directory_map: Dictionary containing paths to your spectral data folders
- ordered_files_map: Lists of spectral filenames in each category

SUPPORTED COMPOUNDS:
Any compound present in your concentration Excel file (e.g., SiO2, CaO, MgO, 
TiO2, Al2O3, K2O, Fe2O3, Na2O)

PROCESSING STEPS:
1. Data Loading: Reads all spectral files and concentration data
2. Quality Control: Removes measurements with high variability (top 20% outliers)
3. Spectrum Averaging: Creates representative spectrum per sample
4. Background Correction: Removes baseline using peak detection and median filtering
5. Data Standardization: Normalizes full spectrum features for machine learning
6. PCA Analysis: Reduces dimensionality while preserving spectral variance
7. Model Training: Uses Ridge regression with cross-validated hyperparameters
8. Optimization: Finds optimal PCA components and Ridge alpha via grid search
9. Validation: Evaluates model performance using cross-validation

USER INTERACTIONS:
During execution, you will be prompted to:
1. Choose sample type: 1=Moon samples, 2=Mars samples, 3=All samples
2. Enter target compound name (must match exactly with Excel file row names)

OUTPUT:
- Optimal number of PCA components
- Best Ridge regression alpha parameter
- Cross-validated R² score (explained variance)
- Cross-validated Mean Absolute Error (MAE)
- Performance comparison across different PCA component numbers
- Trained calibration model ready for predicting unknown samples


AUTHORS: Sergio Quiñónez and Jakub Buday

===============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os     
from scipy.signal import find_peaks, medfilt
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error


plt.close('all')

# Load the concentration file
file_path_conc = r'C:\Users\alejo\OneDrive\Desktop\ingeniería física\8vo Brno University\Spectra\Concentrations.xlsx' 

# Read the Excel file
df_concentrations = pd.read_excel(file_path_conc)


aux = int(input("Choose the kind of samples: 1: Moon samples, 2: Mars samples , 3: All samples:"))

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
    
   2: ['Reference_21.txt', 'MGS-1.txt', 'Reference_22.txt', 'Na2O-BM.txt', 'Reference_23.txt', 'Al2O3-BM.txt',
        'Reference_26.txt', 'K2O-CM.txt', 'Reference_27.txt', 'K2O-BM.txt', 'Reference_28.txt', 'K2O-DM.txt',
        'Reference_29.txt', 'MgO-AM.txt', 'Reference_30.txt', 'MgO-BM.txt', 'Reference_32.txt', 'SiO2-AM.txt',
        'Reference_34.txt', 'SiO2-CM.txt', 'Reference_35.txt', 'Al2O3-BM.txt', 'Reference_36.txt', 'CaO-AM.txt'],
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


compound = input("Enter the compound to analyze (e.g., SiO2): ")

#compound = input("Enter the compound to analyze (e.g., SiO2): ")
spectra_data = []
references = {}
samples = {}
wavelengths_common = None
sd_mix = []
proportion_to_remove = 0.2


for file_name in ordered_files:
    # Build the full path to the file
    file_path = os.path.join(directory, file_name)
    # Read the content of the file line by line
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Convert each line (string) into a list of floats
    data = [list(map(float, line.strip().split())) for line in lines]
    df = pd.DataFrame(data)
    df.columns = ['Wavelength'] + [f'Measurement_{i}' for i in range(1, df.shape[1])]
    

    
    # Normalize each measurement column by dividing by the sum of all measurements in that column
    normalization_factors = df.iloc[:, 1:].sum(axis=0)
    df.iloc[:, 1:] = df.iloc[:, 1:].div(normalization_factors, axis=1)
    
    # Calculate the standard deviation of each measurement column
    std_devs = df.iloc[:, 1:].std(axis=0)
    
    # Determine the number of spectra to remove (5% superior)
    num_to_remove = max(1, int(len(std_devs) * proportion_to_remove))
    
    # Identify the columns with the highest standard deviations
    top_std_columns = std_devs.nlargest(num_to_remove).index
    
    # Eliminate those columns from the DataFrame
    df.drop(columns=top_std_columns, inplace=True)
    
    average_spectrum = df.iloc[:, 1:].mean(axis=1)
    
    # Find peaks in the average spectrum based on prominence and width thresholds
    max_intensity = np.max(average_spectrum)
    relative_prominence = 0.01 * max_intensity
    peaks, properties = find_peaks(average_spectrum, prominence=relative_prominence, width=0.1)
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
    #lod = x_bi + k * s_bi
    lod= 0  # This is a placeholder for the LOD calculation, uncomment the line above to use the LOD calculation
    corrected_intensity = np.where(corrected_spectrum >= lod, corrected_spectrum, 0)
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

# Standardize the data
scaler = StandardScaler()
intensity_matrix_scaled = scaler.fit_transform(intensity_matrix)


# Perform KFold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Range of components to test
component_range = range(1, min(intensity_matrix_scaled.shape[1], 10))  # Adjust the upper limit based on your data

best_n_components = 0
best_r2_score = -np.inf
best_mae_score = np.inf  # Initialize to a large value since lower MAE is better
best_alpha = None  # Variable to store the best alpha value

# Set up the parameter grid for alpha in Ridge regression
param_grid = {'alpha': np.logspace(-3, 3, 10)}

# Loop through different numbers of PCA components to find the optimal number
for n_components in component_range:
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(intensity_matrix_scaled)
    
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
        
        model = Ridge(alpha=alpha)
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
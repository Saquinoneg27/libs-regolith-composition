import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, medfilt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from sklearn.metrics import silhouette_score
from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.layouts import gridplot
import os     
import openpyxl
from openpyxl.chart import ScatterChart, Reference, Series

def load_theoretical_spectra(directory):
    """
  This function loads theoretical spectra data from a specified directory.

  Args:
      directory (str): Path to the directory containing theoretical spectra files.

  Returns:
      dict: Dictionary containing theoretical spectra data.
          Keys are element names (e.g., "Fe") and values are dictionaries containing 
          data for each state (e.g., "Oxidation State +2").
          Inner dictionaries map wavelengths to intensities for each state.
    """
    theoretical_spectra = {}
    for file_name in os.listdir(directory):
        if file_name.endswith(".txt"):
            element = file_name.split('.')[0]
            file_path = os.path.join(directory, file_name)
            df = pd.read_csv(file_path, delimiter=',')
            for column in df.columns[1:]:  # Skip the first column "Wavelength (nm)"
                if 'Sum' not in column:  # Ignore sum columns
                    if element not in theoretical_spectra:
                        theoretical_spectra[element] = {}
                    theoretical_spectra[element][column] = df[['Wavelength (nm)', column]].dropna()
    return theoretical_spectra

def find_theoretical_peaks(theoretical_spectra, prominence_factor=0.1):
    """
  This function identifies peaks in the theoretical spectra data.

  Args:
      theoretical_spectra (dict): Dictionary containing theoretical spectra data 
          (same format as output from load_theoretical_spectra).
      prominence_factor (float, optional): Factor to adjust peak prominence threshold. Defaults to 0.3.

  Returns:
      dict: Dictionary containing peak locations for each element and state combination.
          Keys are tuples (element, state) and values are arrays containing peak wavelengths.
    """
    peaks_dict = {}
    for element, states in theoretical_spectra.items():
        for state, data in states.items():
            wavelengths = data['Wavelength (nm)'].values
            intensities = data[state].values
            normalized_intensities = intensities / np.max(intensities)  # Normalize intensities
            prominence = prominence_factor * np.max(normalized_intensities)  # Adjust prominence
            peaks, _ = find_peaks(normalized_intensities, prominence=prominence)
            peaks_dict[(element, state)] = wavelengths[peaks]
    return peaks_dict

def find_best_random_state(data, num_clusters, n_trials=100):
    """
  This function finds the best random state for K-Means clustering using silhouette score.

  Args:
      data (numpy.ndarray): Data matrix used for clustering.
      num_clusters (int): Number of clusters to use in K-Means clustering.
      n_trials (int, optional): Number of random state trials to evaluate. Defaults to 100.

  Returns:
      tuple: Tuple containing the best random state (integer seed) and the corresponding silhouette score (float).
    """
    best_random_state = None
    best_silhouette_score = -1

    for random_state in range(n_trials):
        kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=random_state)
        clusters = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, clusters)
        
        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_random_state = random_state

    return best_random_state, best_silhouette_score


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



def process_spectrum(x, y, lower_bound, upper_bound ):
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
        y_fit = y_roi  # In case the fitting fails, fallback to the original data

    # Calculate the total area under the peak
    area_under_peak = np.trapz(y_fit, x_roi)

    
    return x_roi, y_fit, main_peak, None, area_under_peak


lower_bound = {"CaO":373.6, "MgO": 279.43, "Al2O3": 394.2, "SiO2": 288,  "TiO2":444.2, "Fe2O3": 247.6, "K2O": 769}
upper_bound = {"CaO":373.9, "MgO": 279.66, "Al2O3": 394.6, "SiO2":288.4, "TiO2":446.2, "Fe2O3": 249.6, "K2O": 771}

plt.close('all')

# Cargar los datos desde el archivo de Excel
file_path_conc = r'C:\Users\alejo\OneDrive\Desktop\ingeniería física\8vo Brno University\Sergio\Concentrations.xlsx' 

# Leer el archivo de Excel
df_concentrations = pd.read_excel(file_path_conc)


atm = int(input("Choose the kind of samples: 1: Moon samples, 2: Mars samples :"))
aux = 2
num_clusters = 4
# Directorio de archivos

if aux ==1:
   
    # Listar archivos
    directory = r'c:\Users\alejo\OneDrive\Desktop\ingeniería física\8vo Brno University\Sergio\Mars\Moon samples'
    ordered_files = [f for f in os.listdir(directory) if f.startswith('Reference_') and f.endswith('.txt')]

if aux == 2 and atm==1 :   
    directory = r'c:\Users\alejo\OneDrive\Desktop\ingeniería física\8vo Brno University\Sergio\Earth\Moon Samples'
    ordered_files = [
    #'Reference_00.txt', 'LHS-1E.txt',
    #'Reference_01.txt', 'LHS-1.txt',
    'Reference_02.txt', 'Dusty.txt',
    'Reference_03.txt', 'LHS-1D.txt',
    #'Reference_04.txt', 'LMS-1.txt',
    #'Reference_05.txt', 'Al2O3-BL.txt',
    'Reference_06.txt', 'K2O-AL.txt',
    'Reference_07.txt', 'Na2O-AL.txt',
    'Reference_08.txt', 'MgO-AL.txt',
    'Reference_09.txt', 'SiO2-DL.txt',
    #'Reference_10.txt', 'SiO2-CL.txt',
    #'Reference_11.txt', 'SiO2-BL.txt',
    'Reference_12.txt', 'SiO2-AL.txt',
    'Reference_13.txt', 'TiO2-AL.txt',
    #'Reference_14.txt', 'TiO2-BL.txt',
    'Reference_15.txt', 'MgO-CL.txt',
    'Reference_16.txt', 'TiO2-CL.txt',
    'Reference_17.txt', 'TiO2-DL.txt',
    #'Reference_18.txt', 'MgO-BL.txt',
    #'Reference_19.txt', 'Al2O3-AL.txt
    
    #compound = input("Enter the compound to analyze (e.g., SiO2): ")
    
    # 'Reference_00.txt', 'MGS-1S.txt',
    # 'Reference_20.txt', 'JEZ-1.txt',
    # 'Reference_21.txt', 'MGS-1.txt',
    # 'Reference_22.txt', 'Na2O-BM.txt',
    # 'Reference_23.txt', 'Al2O3-BM.txt',
    # 'Reference_24.txt', 'SiO2-EM.txt',
    # 'Reference_25.txt', 'SiO2-DM.txt',
    # 'Reference_26.txt', 'K2O-CM.txt',
    # 'Reference_27.txt', 'K2O-BM.txt',
    # 'Reference_28.txt', 'K2O-DM.txt',
    # 'Reference_29.txt', 'MgO-AM.txt',
    # 'Reference_30.txt', 'MgO-BM.txt',
    # 'Reference_31.txt', 'Na2O-AM.txt',
    # 'Reference_32.txt', 'SiO2-AM.txt',
    # 'Reference_33.txt', 'SiO2-BM.txt',
    # 'Reference_34.txt', 'SiO2-CM.txt',
    # 'Reference_35.txt', 'Al2O3-BM.txt',
    # 'Reference_36.txt', 'CaO-AM.txt',
    # 'Reference_37.txt', 'CaO-BM.txt',
    # 'Reference_38.txt', 'K2O-AM.txt',
    ]

    

if aux == 2 and atm==2 :   
    directory = r'c:\Users\alejo\OneDrive\Desktop\ingeniería física\8vo Brno University\Sergio\Earth\Mars samples'
    ordered_files = [
    #'Reference_00.txt', 'MGS-1S.txt',
    #'Reference_20.txt', 'JEZ-1.txt',
    'Reference_21.txt', 'MGS-1.txt',
    'Reference_22.txt', 'Na2O-BM.txt',
    'Reference_23.txt', 'Al2O3-BM.txt',
    #'Reference_24.txt', 'SiO2-EM.txt',
    #'Reference_25.txt', 'SiO2-DM.txt',
    'Reference_26.txt', 'K2O-CM.txt',
    'Reference_27.txt', 'K2O-BM.txt',
    'Reference_28.txt', 'K2O-DM.txt',
    'Reference_29.txt', 'MgO-AM.txt',
    'Reference_30.txt', 'MgO-BM.txt',
    #'Reference_31.txt', 'Na2O-AM.txt',
    'Reference_32.txt', 'SiO2-AM.txt',
    #'Reference_33.txt', 'SiO2-BM.txt',
    'Reference_34.txt', 'SiO2-CM.txt',
    'Reference_35.txt', 'Al2O3-BM.txt',
    'Reference_36.txt', 'CaO-AM.txt',
    #'Reference_37.txt', 'CaO-BM.txt',
    'Reference_38.txt', 'K2O-AM.txt',
    
    ]

    



if aux == 2:
    # Prompt the user to enter the compound to analyze
    compound = input("Enter the compound to analyze (e.g., SiO2): ")

spectra_data = []  # List to store the spectra data
references = {}    # Dictionary to store reference data
samples = {}       # Dictionary to store sample data
wavelengths_common = None  # Variable to store common wavelengths across files
sd_mix = []  # List to store the standard deviation of the areas
proportion_to_remove = 0.2  # Proportion of spectra to remove based on standard deviation

# Iterate over each file in the ordered list
for file_name in ordered_files:
    # Construct the full file path
    file_path = os.path.join(directory, file_name)
    
    # Read the file line by line
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
    
    # Determine the number of spectra to remove (top 20%)
    num_to_remove = max(1, int(len(std_devs) * proportion_to_remove))
    
    # Identify the columns with the highest standard deviations
    top_std_columns = std_devs.nlargest(num_to_remove).index
    
    # Remove these columns from the DataFrame
    df.drop(columns=top_std_columns, inplace=True)
    
    area_ = []
    area_Si = []
    # Calculate the area under the curve for the specified compound and SiO2
    for i in range(1, df.shape[1]):
        x_i, y_i, main_peak, _, areaSi = process_spectrum(df['Wavelength'].to_numpy(), df.iloc[:, i].to_numpy(), lower_bound["SiO2"], upper_bound["SiO2"])
        x_i, y_i, main_peak, _, area = process_spectrum(df['Wavelength'].to_numpy(), df.iloc[:, i].to_numpy(), lower_bound[compound], upper_bound[compound])
        area_.append(area)
        area_Si.append(areaSi)
    
    # Calculate and store the standard deviation of the areas for this file
    # Replace None values with 0 before calculating standard deviation
    cleaned_area_ = [x if x is not None else 0 for x in area_]
    sd_value = np.std(cleaned_area_)  # Standard deviation
    sd_mix.append(sd_value)  # Append result to sd_mix
    
    # Calculate the average spectrum from the remaining data
    average_spectrum = df.iloc[:, 1:].mean(axis=1)
    
    # Identify peaks in the average spectrum based on prominence and width thresholds
    max_intensity = np.max(average_spectrum)
    relative_prominence = 0.01 * max_intensity
    peaks, properties = find_peaks(average_spectrum, prominence=relative_prominence, width=0.1)
    
    # Create a mask to identify non-peak regions (where True indicates non-peak)
    mask = np.ones_like(average_spectrum, dtype=bool)
    mask[peaks] = False
    mask2 = ~mask

    # Calculate the background noise
    background = medfilt(average_spectrum[mask], kernel_size=51)
    background2 = np.mean(background)

    # Correct the spectrum by subtracting the background
    corrected_spectrum = np.copy(average_spectrum)
    corrected_spectrum[mask] = average_spectrum[mask] - background
    corrected_spectrum[mask2] = average_spectrum[mask2] - background2
    corrected_spectrum[corrected_spectrum < 0] = 0  # Set negative values to zero

    # Calculate the mean and standard deviation of the non-peak regions
    blank_indices = [i for i in range(len(corrected_spectrum)) if i not in peaks]
    blank_values = [corrected_spectrum[i] for i in blank_indices]
    x_bi = np.mean(blank_values)
    s_bi = np.std(blank_values)
    
    # Define the limit of detection (LOD) based on a factor (k) times the blank standard deviation
    k = 3
    lod = x_bi + k * s_bi
    lod = 0  # Reset LOD to 0 if needed
    corrected_intensity = np.where(corrected_spectrum >= lod, corrected_spectrum, 0)
    df['Corrected_Intensity'] = corrected_intensity
    wavelengths_np = df['Wavelength'].to_numpy()
    intensities_np = df['Corrected_Intensity'].to_numpy()
    
    # Align wavelengths if necessary
    if wavelengths_common is None:
        wavelengths_common = wavelengths_np
    else:
        intensities_np = np.interp(wavelengths_common, wavelengths_np, intensities_np)

    # Store the spectrum data
    spectrum_data = {
        'file': file_name,
        'wavelengths': wavelengths_common,
        'intensities': intensities_np
    }
    
    # Separate references and samples
    if 'Reference' in file_name:
        references[file_name] = spectrum_data
    else:
        samples[file_name] = spectrum_data
    
    spectra_data.append(spectrum_data)

# Filter only sample spectra from the data
sample_spectra = [spectrum for spectrum in spectra_data if 'Reference' not in spectrum['file']]

# Convert data from spectra to a PCA matrix
intensity_matrix = np.array([spectrum['intensities'] for spectrum in sample_spectra])
pca = PCA(n_components=3)
principal_components = pca.fit_transform(intensity_matrix)
explained_variance = pca.explained_variance_ratio_


# fig = plt.figure(figsize=(14, 8))
# ax = fig.add_subplot(111, projection='3d')
# colormap = plt.colormaps['tab20']
# colors = colormap(np.linspace(0, 1, len(sample_spectra)))

# for i, spectrum in enumerate(sample_spectra):
#     ax.scatter(principal_components[i, 0], principal_components[i, 1], principal_components[i, 2], 
#                 color=colors[i], label=spectrum['file'])

# ax.set_xlabel(f'Principal Component 1 ({explained_variance[0]*100:.2f}%)')
# ax.set_ylabel(f'Principal Component 2 ({explained_variance[1]*100:.2f}%)')
# ax.set_zlabel(f'Principal Component 3 ({explained_variance[2]*100:.2f}%)')
# ax.set_title('3D PCA of Spectra')
# ax.legend()
# plt.show()



# loadings = pca.components_


# output_file("spectra.html")

# # Load theorethical spectra 
# theoretical_directory = r'C:\Users\alejo\OneDrive\Desktop\ingeniería física\8vo Brno University\Sergio\Elements'
# theoretical_spectra = load_theoretical_spectra(theoretical_directory)
# theoretical_peaks = find_theoretical_peaks(theoretical_spectra)

# figures = []
# figures = []
# for i in range(3):
#     # Create a new plot figure with title, axis labels, and desired dimensions
#     p = figure(title=f'Weight of Principal Component {i+1}', x_axis_label='Wavelength', y_axis_label='Weight', width=700, height=400)
#     # Plot the loadings (weight) for each variable (wavelength) for the current component
#     p.line(sample_spectra[0]['wavelengths'], loadings[i], legend_label=f'PC{i+1}')

#     # find positive and negative peaks of the loads
#     # - prominence: minimum threshold for peak prominence
#     # - height: minimum threshold for peak height
#     positive_peaks, _ = find_peaks(loadings[i], prominence=0.025, height= 0.02)
#     negative_peaks, _ = find_peaks(-loadings[i], prominence=0.025, height= 0.02)
    
#     # List to store information about potential peaks
#     peaks_info = []
#     for peak in np.concatenate((positive_peaks, negative_peaks)):
#         wavelength = sample_spectra[0]['wavelengths'][peak]
#         for (element, state), peaks in theoretical_peaks.items():
#             # Check if any theoretical peak is within a tolerance (0.1) of the current peak's wavelength
#             if np.any(np.abs(peaks - wavelength) <= 0.1):
#                 peaks_info.append((wavelength, loadings[i][peak], f'{element} {state} {wavelength:.1f}'))
    
#      # Create a data source for efficient plotting of potential peaks
#     source = ColumnDataSource(data={
#         'x': [p[0] for p in peaks_info],# Extract wavelengths from peaks_info
#         'y': [p[1] for p in peaks_info],# Extract loading values from peaks_info
#         'desc': [p[2] for p in peaks_info] 
#     })

#     # Plot potential peaks as red circles with size 4
#     p.scatter('x', 'y', source=source, size=4, color='red', legend_label='Peaks')
#     # Add hover tool to display peak description on hover
#     hover = HoverTool(tooltips=[("Desc", "@desc")])
#     p.add_tools(hover)
#     # Append the current plot to a list of plots
#     figures.append(p)

# grid = gridplot(figures, ncols=3)
# show(grid)




best_random_state, best_silhouette_score = find_best_random_state(principal_components, num_clusters)

print(f"Best random state: {best_random_state} with a silhouette score of {best_silhouette_score:.4f}")

kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=best_random_state)
clusters = kmeans.fit_predict(principal_components)


# fig = plt.figure(figsize=(14, 8))
# ax = fig.add_subplot(111, projection='3d')

# colors = colormap(np.linspace(0, 1, num_clusters))

# for cluster in range(num_clusters):
#     cluster_indices = np.where(clusters == cluster)
#     ax.scatter(principal_components[cluster_indices, 0], principal_components[cluster_indices, 1], principal_components[cluster_indices, 2], 
#                label=f'Cluster {cluster + 1}', color=colors[cluster])

# ax.set_xlabel('Principal Component 1 (PC1)')
# ax.set_ylabel('Principal Component 2 (PC2)')
# ax.set_zlabel('Principal Component 3 (PC3)')
# ax.set_title('PCA of Spectra Data with Cluster Highlighting')
# ax.legend()
#plt.show()

# 2D Plots for clusters
#fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# PC2 vs PC1
#for cluster in range(num_clusters):
    #cluster_indices = np.where(clusters == cluster)
    #axes[0].scatter(principal_components[cluster_indices, 0], principal_components[cluster_indices, 1], label=f'Cluster {cluster + 1}', color=colors[cluster])

#axes[0].set_xlabel(f'Principal Component 1 ({explained_variance[0]*100:.2f}%)')
#xes[0].set_ylabel(f'Principal Component 2 ({explained_variance[1]*100:.2f}%)')
#axes[0].set_title('PC2 vs PC1')
#axes[0].legend()

# PC3 vs PC1
#for cluster in range(num_clusters):
 #   cluster_indices = np.where(clusters == cluster)
  #  axes[1].scatter(principal_components[cluster_indices, 0], principal_components[cluster_indices, 2], label=f'Cluster {cluster + 1}', color=colors[cluster])

#axes[1].set_xlabel(f'Principal Component 1 ({explained_variance[0]*100:.2f}%)')
#axes[1].set_ylabel(f'Principal Component 3 ({explained_variance[2]*100:.2f}%)')
#axes[1].set_title('PC3 vs PC1')
#axes[1].legend()

#plt.tight_layout()
#plt.show()

#Compare Reference spectra
# Not significant differences should be seen 
#if aux == 1:
   #plt.figure(figsize=(14, 8))
    #for spectrum in spectra_data:
     #   plt.plot(spectrum['wavelengths'], spectrum['intensities'], label=spectrum['file'])
    #plt.xlabel('Wavelength')
    #plt.ylabel('Intensity')
    #plt.title('Comparative Spectrum Data')
    #plt.legend()
    #plt.show()




# If any sample is to be excluded from the graphs but not from the clusters 
excluded_samples = ['']  

# This section analyzes data for samples excluding outliers

if aux == 2:
    Sum_fwhm_vect = []
    concentrations = []
    cluster_assignments = []
    area_ratios = []
    error_bars = []
    #reference_area =[]
    #error_bars_ref =[]
    #REFFF =[]
    # Loop through each sample (excluding excluded ones)

    for (i, (sample_file, sample_data)), (j, (reference_file, reference_data)) in zip(enumerate(samples.items()), enumerate(references.items())):        
        # Omit excluded samples
        if sample_file in excluded_samples:
            continue
        # Extract data from sample and corresponding reference spectra
        x, y, main_peak, _, area_under_peak = process_spectrum(sample_data['wavelengths'], sample_data['intensities'], lower_bound[compound], upper_bound[compound])
        _, _, main_peak_R, _, ref_area_under_peak = process_spectrum(reference_data['wavelengths'], reference_data['intensities'], 393, 394)
        
        # Get cluster assignment and print informative messages
        cluster = clusters[i]
        print(f"Sample File: {sample_file}")
        print(f"Cluster: {cluster + 1}")
        print(f"Main peak at: {x[main_peak] if main_peak is not None else 'N/A'}")
        #print(f"Main REFERENCE peak at: {x[main_peak_R] if main_peak_R is not None else 'N/A'}")
        
        # Calculate and print area ratio (if possible)
        if area_under_peak :
            area_ratio = area_under_peak / 1
            print(f" Area under Peak: {area_under_peak}")
            area_ratios.append(area_ratio)
            Sum_fwhm_vect.append(area_under_peak)
            cluster_assignments.append(cluster)
            #reference_area.append(ref_area_under_peak)
            error_bars.append(sd_mix[ordered_files.index(sample_file)])  # Agregar la desviación estándar correspondiente
            #error_bars_ref.append(sd_mix[ordered_files.index(reference_file)])  # Agregar la desviación estándar correspondiente
            reference_numbers = int(reference_file.split('_')[1].split('.')[0])
           # REFFF.append(reference_numbers)
        
            sample_name = os.path.splitext(sample_data['file'])[0]
            concentration = df_concentrations.loc[df_concentrations['Sample'] == compound, sample_name].values[0]
            concentrations.append(concentration)
        else:
            print("Area under peak could not be determined: Sample file: {sample_file}")
            concentrations.append(None)
            Sum_fwhm_vect.append(None)
            area_ratios.append(None)
            error_bars.append(None)
            #error_bars_ref.append(None)  # Asegúrate de mantener la consistencia en la longitud de las listas
            #reference_area.append(None)
        print()
    
    # Filter data for valid area ratios (excluding None values)
    filtered_area_ratios = [area_ratios[i] for i in range(len(area_ratios)) if area_ratios[i] is not None]
    filtered_concentrations = [concentrations[i] for i in range(len(concentrations)) if area_ratios[i] is not None]
    filtered_clusters = [cluster_assignments[i] for i in range(len(cluster_assignments)) if area_ratios[i] is not None]
    filtered_error_bars = [error_bars[i] for i in range(len(error_bars)) if area_ratios[i] is not None]

    # Create calibration curves for each cluster (if enough data points)
    fig, axes = plt.subplots(num_clusters, 1, figsize=(10, 15))

    for cluster in range(num_clusters):
        # Extract data for the current cluster
        cluster_indices = [i for i in range(len(cluster_assignments)) if cluster_assignments[i] == cluster]
        cluster_concentrations = [filtered_concentrations[i] for i in cluster_indices]
        cluster_area_ratios = [filtered_area_ratios[i] for i in cluster_indices]
        cluster_error_bars = [filtered_error_bars[i] for i in cluster_indices]

        if len(cluster_concentrations) < 2 or len(cluster_area_ratios) < 2:
            print(f"Not enough data points for cluster {cluster + 1}")
            continue

        # Perform linear regression and calculate R-squared
        A = np.vstack([cluster_concentrations, np.ones(len(cluster_concentrations))]).T

        mask = np.isfinite(A).all(axis=1) & np.isfinite(cluster_area_ratios)
        A = A[mask]
        cluster_concentrations = np.array(cluster_concentrations)[mask]
        cluster_area_ratios = np.array(cluster_area_ratios)[mask]
        cluster_error_bars = np.array(cluster_error_bars)[mask]

        if len(cluster_area_ratios) < 2:
            print(f"Not enough valid data points for cluster {cluster + 1} after filtering")
            continue

        # Perform linear regression for the current cluster
        pendiente, intercepto = np.linalg.lstsq(A, cluster_area_ratios, rcond=None)[0]
        intensidades_pred = pendiente * np.array(cluster_concentrations) + intercepto
        # Calculate R-squared

        ss_tot = np.sum((np.array(cluster_area_ratios) - np.mean(cluster_area_ratios)) ** 2)
        ss_res = np.sum((np.array(cluster_area_ratios) - intensidades_pred) ** 2)
        r_cuadrado = 1 - (ss_res / ss_tot)

         # Create and customize the plot for the current cluster
        ax = axes[cluster]
        ax.errorbar(cluster_concentrations, cluster_area_ratios, yerr=cluster_error_bars, fmt='o', label='Experimental Data')
        ax.plot(cluster_concentrations, intensidades_pred, color='red', label='Prediction')
        ax.set_xlabel('Concentration (Mass fractions in %)')
        ax.set_ylabel('Area Ratio (sample/reference)')
        ax.set_title(f'Calibration curve for cluster {cluster + 1}')
        ax.legend()
        ax.grid()
        ax.text(0.05, 0.95, f'Intensidad = {pendiente:.2f} * Concentración + {intercepto:.2f}\n$R^2$ = {r_cuadrado:.2f}', 
                transform=ax.transAxes, verticalalignment='top')

    plt.tight_layout()
    plt.show()

    # Create a new figure for all data points together
    fig_all, ax_all = plt.subplots(figsize=(10, 6))
    ax_all.errorbar(filtered_concentrations, filtered_area_ratios, yerr=filtered_error_bars, fmt='o', label='All Data')

    # Perform linear regression for all data points together
    A_all = np.vstack([filtered_concentrations, np.ones(len(filtered_concentrations))]).T
    mask_all = np.isfinite(A_all).all(axis=1) & np.isfinite(filtered_area_ratios)
    A_all = A_all[mask_all]
    filtered_concentrations = np.array(filtered_concentrations)[mask_all]
    filtered_area_ratios = np.array(filtered_area_ratios)[mask_all]
    filtered_error_bars = np.array(filtered_error_bars)[mask_all]

    if len(filtered_area_ratios) >= 2:
        pendiente_all, intercepto_all = np.linalg.lstsq(A_all, filtered_area_ratios, rcond=None)[0]
        intensidades_pred_all = pendiente_all * np.array(filtered_concentrations) + intercepto_all

        ss_tot_all = np.sum((np.array(filtered_area_ratios) - np.mean(filtered_area_ratios)) ** 2)
        ss_res_all = np.sum((np.array(filtered_area_ratios) - intensidades_pred_all) ** 2)
        r_cuadrado_all = 1 - (ss_res_all / ss_tot_all)

        ax_all.plot(filtered_concentrations, intensidades_pred_all, color='red', label='Prediction')
        ax_all.set_xlabel('Concentration (Mass fractions in %)')
        ax_all.set_ylabel('Area Ratio (sample/1)')
        ax_all.set_title(f'Calibration curve for all data')
        ax_all.legend()
        ax_all.grid()
        ax_all.text(0.05, 0.95, f'Intensidad = {pendiente_all:.2f} * Concentración + {intercepto_all:.2f}\n$R^2$ = {r_cuadrado_all:.2f}', 
                    transform=ax_all.transAxes, verticalalignment='top')

    plt.tight_layout()
    plt.show()

    # Create a new figure for all data points together
   
    
    #fig_all, ax_all = plt.subplots(figsize=(10, 6))
    #experiment = list(range(1, len(reference_area) + 1))
    #ax_all.errorbar(REFFF, reference_area, yerr=error_bars_ref, fmt='o', label='Reference data')
    #ax_all.set_xlabel('Experiment number')
    #ax_all.set_ylabel('Area Ratio (Reference/1)')
    #ax_all.set_title(f'Calibration curve for all data')
    #ax_all.legend()
    #ax_all.grid()

    #plt.tight_layout()
    #plt.show()

    # Realizar la prueba t para muestras pareadas
    #t_statistic, p_value = ttest_rel(reference_area[:-1], reference_area[1:])

    #print(f'T-statistic: {t_statistic}')
    #print(f'P-value: {p_value}')

    # Interpretación
   # alpha = 0.05
    #if p_value < alpha:
       # print("Hay una diferencia significativa entre los conjuntos de datos")
    #else:
        #print("No hay una diferencia significativa entre los conjuntos de datos")



    # Create a DataFrame with the data
    data = {
        'Concentration (Mass fractions in %)': filtered_concentrations,
        'Intensity (a.u)': filtered_area_ratios,
        "Errors": filtered_error_bars
    }
    df = pd.DataFrame(data)
    
    # Excel file path
    file_path = r'c:\Users\alejo\OneDrive\Desktop\ingeniería física\8vo Brno University\calibration_curve_data.xlsx'
    
    # Create an Excel file and write the DataFrame
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    
    # Upload Excel file to add the chart
    wb = openpyxl.load_workbook(file_path)
    ws = wb['Sheet1']
    
    # Create a scatter plot
    chart = ScatterChart()
    chart.title = f"{compound}-MS-EA"
    chart.x_axis.title = "Concentration (Mass fractions in %)"
    chart.y_axis.title = "I (a.u)"
    #  Select data for the graph
    xvalues = Reference(ws, min_col=1, min_row=2, max_row=len(df) + 1)
    yvalues = Reference(ws, min_col=2, min_row=2, max_row=len(df) + 1)
    
    # Create the series and add it to the chart
    series = Series(yvalues, xvalues, title="Data")
    series.marker.symbol = "circle"  # Asegurarse de que se muestren los puntos
    series.graphicalProperties.line.noFill = True  # Quitar las líneas
    
    chart.series.append(series)
    
    # Add the chart to the spreadsheet
    ws.add_chart(chart, "E5")
    
    # Save the Excel file with the chart
    wb.save(file_path)
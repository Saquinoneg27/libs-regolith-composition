"""
LIBS Spectra Pipeline Lunar / Martian Regolith Simulants CNN architecture
=======================================================================

Purpose
-------
End-to-end processing and analysis of LIBS (Laser-Induced Breakdown
Spectroscopy) data from lunar and Martian regolith simulants.  
Steps covered:
    • baseline correction and LOD filtering  
    • wavelength-wise normalisation  
    • dimensionality reduction with PCA  
    • unsupervised clustering (K-Means)  
    • supervised classification with a 1-D CNN  
    • feature importance via PLS-DA VIP scores and SHAP values  
    • interactive and static visualisation of results

Required Inputs
---------------
1. **Spectra folders** (choose *Moon Samples*, *Mars samples* or *All_of_them*):
       Spectra/
         └── <Planet>/
             ├── Moon Samples/        # optional
             ├── Mars samples/        # optional
             └── All_of_them/         # optional

   Each *.txt* file must be space-separated:  
   `wavelength  intensity₁  intensity₂ … intensity₁₀₀`.

2. **Theoretical line library** in `Spectra/Elements/`  
   CSV files named `Fe.txt`, `Si.txt`, … with columns  
   `Wavelength (nm), Oxidation State +2, …`.  
   Columns named “Sum” are ignored.

3. Python packages: `numpy, pandas, scikit-learn, tensorflow-keras,
   matplotlib, seaborn, scipy, bokeh, shap`.

Execution
---------
Run the script and pick the data set when prompted:  
`1 → Moon`, `2 → Mars`, or `3 → All`.  
Adjust `chosen_condition`, directory paths, or `num_columns` if your
layout differs.

Outputs
-------
* **spectra.html** – interactive Bokeh grid with theoretical found peak lines  
* Matplotlib figures: elbow diagram, pairplot by cluster, confusion
  matrices, VIP curve, prototype spectrum for class 1  
* Per-fold metrics printed to console (accuracy, precision, recall, loss)  
* SHAP force-plot HTML files (`force_plot_class_*.html`)  
* Predicted class labels for provided test samples

All tensors fed to the CNN are normalised and padded/truncated to the
same wavelength grid; modify the interpolation/padding logic if your
spectra lengths vary significantly.

Author: Sergio Quiñonez and Jakub Buday
"""

import os
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
from scipy.signal import find_peaks, medfilt
from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.layouts import gridplot
import shap
from scipy.spatial.distance import cdist

from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.cross_decomposition import PLSRegression

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
            for column in df.columns[1:]:  # Saltar la primera columna "Wavelength (nm)"
                if 'Sum' not in column:  # Ignorar las columnas de sumas
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
            normalized_intensities = intensities / np.max(intensities)  # Normalizar las intensidades
            prominence = prominence_factor * np.max(normalized_intensities)  # Ajustar prominencia
            peaks, _ = find_peaks(normalized_intensities, prominence=prominence)
            peaks_dict[(element, state)] = wavelengths[peaks]
    return peaks_dict

# Set parameters based on user input
aux = int(input("Choose the kind of samples: 1: Moon samples, 2: Mars samples , 3: All samples:"))
chosen_condition = "Mars"

directory_map = {
    1: rf'c:\Users\alejo\OneDrive\Desktop\ingeniería física\8vo Brno University\Spectra\{chosen_condition}\Moon Samples',
    2: rf'c:\Users\alejo\OneDrive\Desktop\ingeniería física\8vo Brno University\Spectra\{chosen_condition}\Mars samples',
    3: rf'c:\Users\alejo\OneDrive\Desktop\ingeniería física\8vo Brno University\Spectra\{chosen_condition}\All_of_them'
}

ordered_files_map = {
    1: [ 'LHS-1.txt', 'Dusty.txt', 'LHS-1D.txt', 
         'K2O-AL.txt', 'Na2O-AL.txt', 'MgO-AL.txt', 
         'SiO2-DL.txt', 'SiO2-AL.txt', 'TiO2-AL.txt', 
         'MgO-CL.txt', 'TiO2-CL.txt', 'TiO2-DL.txt'],
    
    2: [ 'MGS-1.txt', 'Na2O-BM.txt',  'Al2O3-BM.txt',
         'K2O-CM.txt',  'K2O-BM.txt', 'K2O-DM.txt',
         'MgO-AM.txt',  'MgO-BM.txt',  'SiO2-AM.txt',
         'SiO2-CM.txt', 'Al2O3-BM.txt', 'CaO-AM.txt'],
    3: [ 'LHS-1.txt', 'Dusty.txt', 'LHS-1D.txt', 
         'K2O-AL.txt', 'Na2O-AL.txt', 'MgO-AL.txt', 
         'SiO2-DL.txt', 'SiO2-AL.txt', 'TiO2-AL.txt', 
         'MgO-CL.txt', 'TiO2-CL.txt', 'TiO2-DL.txt',

         'MGS-1.txt', 'Na2O-BM.txt',  'Al2O3-BM.txt',
        'K2O-CM.txt',  'K2O-BM.txt', 'K2O-DM.txt',
        'MgO-AM.txt',  'MgO-BM.txt',  'SiO2-AM.txt',
        'SiO2-CM.txt', 'Al2O3-BM.txt', 'CaO-AM.txt']
}

directory = directory_map.get(aux, "")
ordered_files = ordered_files_map.get(aux, [])

spectra_data = []
spectra_data_wild = []
labels = []

# Dictionary to save normalization factors 
normalization_factors_dict = {}

# Iterate through each file in the ordered list
for file_name in ordered_files:
    file_path = os.path.join(directory, file_name)
    
    # Open and read each file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Convert lines to a list of floats and create a DataFrame
    data = [list(map(float, line.strip().split())) for line in lines]
    df = pd.DataFrame(data)
    
    # Assign column names: 'Wavelength' and 'Measurement_i' for each measurement
    df.columns = ['Wavelength'] + [f'Measurement_{i}' for i in range(1, df.shape[1])]
    
    # Make a copy of the DataFrame for unnormalized data
    wild_data = np.copy(df)    

    # Calculate normalization factors as the sum of each measurement column
    normalization_factors = df.iloc[:, 1:].sum(axis=0)
    normalization_factors_dict[file_name] = normalization_factors
    
    # Normalize each measurement column by its corresponding normalization factor
    df.iloc[:, 1:] = df.iloc[:, 1:].div(normalization_factors, axis=1)

    # Prepare to store corrected intensities for each measurement column
    num_columns = 100  # We have 100 measurement columns

    corrected_intensities = []
    wild = []
    
    # Loop over each measurement column
    for j in range(1, num_columns + 1):
        column_data = df.iloc[:, j]
        
        # Find peaks with a prominence of 10% of the maximum intensity
        max_intensity = np.max(column_data)
        relative_prominence = 0.1 * max_intensity
        peaks, properties = find_peaks(column_data, prominence=relative_prominence, width=0.2)
        
        # Create two masks: one for background and one for peaks
        mask = np.ones_like(column_data, dtype=bool)
        mask[peaks] = False
        mask2 = ~mask
        
        # Calculate background and adjust the spectrum
        background = medfilt(column_data[mask], kernel_size=51)
        background2 = np.mean(background)
        corrected_spectrum = np.copy(column_data)
        corrected_spectrum[mask] = column_data[mask] - background
        corrected_spectrum[mask2] = column_data[mask2] - background2
        
        # Calculate limits of detection (LOD)
        blank_mask = np.ones_like(corrected_spectrum, dtype=bool)
        blank_mask[peaks] = False
        x_bi = np.mean(corrected_spectrum[blank_mask])
        s_bi = np.std(corrected_spectrum[blank_mask])
        k = 3  # Detection limit constant, typically set to 3
        lod = x_bi + k * s_bi
    
        # Set intensities below LOD to zero
        corrected_intensity = np.where(corrected_spectrum >= lod, corrected_spectrum, 0)
        corrected_intensities.append(corrected_intensity)
        wild.append(wild_data[:, j])

    # Convert wavelengths and intensities to numpy arrays
    wavelengths_np = df['Wavelength'].to_numpy()
    spectrum_data = {
        'file': file_name,
        'wavelengths': wavelengths_np,
        'intensities': corrected_intensities
    }
    spectrum_data_wild = {
        'file': file_name,
        'wavelengths': wavelengths_np,
        'intensities': wild
    }
    
    # Append processed data to corresponding lists
    spectra_data.append(spectrum_data)
    spectra_data_wild.append(spectrum_data_wild)
    labels.append(file_name)  # File name represents class label

# Convert lists of intensities to numpy arrays for model input
X = np.array([spectrum['intensities'] for spectrum in spectra_data])
X = X.reshape(-1, X.shape[2])  # Reshape to required dimensions

X_wild = np.array([spectrum['intensities'] for spectrum in spectra_data_wild])
X_wild = X_wild.reshape(-1, X_wild.shape[2])  # Reshape to required dimensions

sample_spectra = [spectrum for spectrum in spectra_data]

# Normalize the spectra
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA with 3 components to reduce dimensionality
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)
explained_variance = pca.explained_variance_ratio_

# Use the elbow method to determine the optimal number of clusters
SS = []  # Sum of squared distances
CONT = []
max_clusters = min(20, len(spectra_data))

for i in range(1, max_clusters):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_pca)
    SS.append(kmeans.inertia_)
    CONT.append(i)

# Plot elbow graph to find the optimal number of clusters
plt.figure(figsize=(8, 6))
plt.plot(CONT, SS, 'bx-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()

optimal_clusters = 4  # This number should be selected based on the elbow graph

# Apply KMeans clustering with the optimal number of clusters
best_random_state, best_silhouette_score = find_best_random_state(X_pca, optimal_clusters)

print(f"Best random state: {best_random_state} with a silhouette score of {best_silhouette_score:.4f}")


# Calculate the centroid of the PCA data
centroid = np.mean(X_pca, axis=0)

# Calculate the Euclidean distance from each point to the centroid.
distances = cdist(X_pca, [centroid], 'euclidean').flatten()

# Determine a threshold for identifying outliers, e.g., 95th percentile
threshold = np.percentile(distances, 95)

# Identify outliers
outliers = distances > threshold
# Filter outliers
X_pca = X_pca[~outliers]


kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=best_random_state)
y_kmeans = kmeans.fit_predict(X_pca)

# Customized colors for clusters
cluster_colors = ['red', 'green', 'blue', 'orange', 'purple']
color_labels = ['Cluster 1: Red', 'Cluster 2: Green', 'Cluster 3: Blue', 'Cluster 4: Orange', 'Cluster 5: Purple']

# Create a list of colors for each point based on its cluster label.
colors = [cluster_colors[label] for label in y_kmeans]

# Create a DataFrame for use with seaborn
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2', 'PC3'])
pca_df['Cluster'] = y_kmeans

# Plot pairplot
sns.pairplot(pca_df, hue='Cluster', palette=cluster_colors, markers=["o", "s", "D", "X"])
# add personalized legend
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Clase 0'),
           plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=10, label='Clase 1'),
           plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='green', markersize=10, label='Clase 2'),
           plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='purple', markersize=10, label='Clase 3')]

plt.legend(handles=handles, title='Clases', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


loadings = pca.components_
output_file("spectra.html")

# Load theorethical spectra 
theoretical_directory = r'C:\Users\alejo\OneDrive\Desktop\ingeniería física\8vo Brno University\Spectra\Elements'
theoretical_spectra = load_theoretical_spectra(theoretical_directory)
theoretical_peaks = find_theoretical_peaks(theoretical_spectra)
figures = []

for i in range(3):
    # Create a new plot figure with title, axis labels, and desired dimensions
    p = figure(title=f'Weight of Principal Component {i+1}', x_axis_label='Wavelength', y_axis_label='Weight', width=700, height=400)
    # Plot the loadings (weight) for each variable (wavelength) for the current component
    p.line(sample_spectra[0]['wavelengths'], loadings[i], legend_label=f'PC{i+1}')

    # find positive and negative peaks of the loads
    # - prominence: minimum threshold for peak prominence
    # - height: minimum threshold for peak height
    positive_peaks, _ = find_peaks(loadings[i], prominence=0.025, height= 0.02)
    negative_peaks, _ = find_peaks(-loadings[i], prominence=0.025, height= 0.02)
    
    # List to store information about potential peaks
    peaks_info = []
    for peak in np.concatenate((positive_peaks, negative_peaks)):
        wavelength = sample_spectra[0]['wavelengths'][peak]
        for (element, state), peaks in theoretical_peaks.items():
            # Check if any theoretical peak is within a tolerance (0.1) of the current peak's wavelength
            if np.any(np.abs(peaks - wavelength) <= 0.1):
                peaks_info.append((wavelength, loadings[i][peak], f'{element} {state} {wavelength:.1f}'))
    
     # Create a data source for efficient plotting of potential peaks
    source = ColumnDataSource(data={
        'x': [p[0] for p in peaks_info],# Extract wavelengths from peaks_info
        'y': [p[1] for p in peaks_info],# Extract loading values from peaks_info
        'desc': [p[2] for p in peaks_info] 
    })

    # Plot potential peaks as red circles with size 4
    p.scatter('x', 'y', source=source, size=4, color='red', legend_label='Peaks')
    # Add hover tool to display peak description on hover
    hover = HoverTool(tooltips=[("Desc", "@desc")])
    p.add_tools(hover)
    # Append the current plot to a list of plots
    figures.append(p)

grid = gridplot(figures, ncols=3)
show(grid)

########################################################################3
# Preparing data for the CNN


X_wild = X_wild[~outliers]
X_wild = X_wild / np.max(X_wild, axis=1, keepdims=True)  # Normalize again
y_categorical = to_categorical(y_kmeans)


kf= KFold(n_splits=3, shuffle=True, random_state=42)
fold_no =1
acc_per_fold = []
loss_per_fold =[]
vip_scores_folds = []
precision_per_fold = []
recall_per_fold = []
correctly_classified_class_1_spectra = []

def calculate_vip(model, X):
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_

    p, h = w.shape
    vips = np.zeros((p,))

    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)

    for i in range(p):
        weight = np.array([ (w[i, k] / np.linalg.norm(w[:, k])) ** 2 for k in range(h) ])
        vips[i] = np.sqrt(p * (s.T @ weight) / total_s)

    return vips

for train_index, val_index in kf.split(X_wild):
    print(f"Training on fold {fold_no}...")
    
    # Splitting data into training and test sets
    X_train,X_val = X_wild[train_index], X_wild[val_index]
    y_train,y_val = y_categorical[train_index], y_categorical[val_index]
    
    # Reshape for CNN input
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    
 
    #define model
    model = Sequential()
    model.add(Conv1D(filters=8, kernel_size=8, activation='relu', input_shape=(X_train.shape[1], 1)))# First convolutional layer
    model.add(MaxPooling1D(pool_size=2))  #Reduce dimensionality of ur features but actually increase dimensionality of ur filters!
    model.add(Conv1D(filters=16, kernel_size=4, activation='relu')) #Relu: Max(0,z)
    model.add(MaxPooling1D(pool_size=2)) # u take the maximum of the patch, downsample and preserve  spatial invariance
    model.add(Flatten()) #This layer transforms the 2D output of the convolutional layers  into a 1D vector.
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=y_train.shape[1], activation='softmax'))  # Be sure that it matches the number of classes
    
    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.0012), loss='categorical_crossentropy', metrics=['accuracy'])
    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    # train model
    model.fit(X_train, y_train, epochs=25, batch_size=13, validation_split=0.2, verbose=1)
    
    # Evaluate model
    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f'Fold {fold_no} - Loss: {loss}, Accuracy: {accuracy}')
    acc_per_fold.append(accuracy)
    loss_per_fold.append(loss)
    
    
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_val, axis=1)
    
    # Identify correctly classified samples for class 1
    correct_class_1_indices = (y_true == 1) & (y_pred_classes == 1)
    correctly_classified_class_1_spectra.append(X_val[correct_class_1_indices])
   
    
    # Calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    precision = precision_score(y_true, y_pred_classes, average='weighted')
    recall = recall_score(y_true, y_pred_classes, average='weighted')
    
    print(f'Fold {fold_no} - Precision: {precision}, Recall: {recall}')
    precision_per_fold.append(precision)
    recall_per_fold.append(recall)

    
    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for Fold {fold_no}')
    plt.show()
    
    # Reshape for PLS-DA input (to 2D)
    X_train_2d = X_train.reshape(X_train.shape[0], X_train.shape[1])
    
    # Standardize the features for PLS-DA
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_2d)
    
    # Train PLS-DA model
    plsda = PLSRegression(n_components=2)
    plsda.fit(X_train_scaled, y_kmeans[train_index])
    
    # Calculate VIP scores
    vip_scores = calculate_vip(plsda, X_train_scaled)
    vip_scores_folds.append(vip_scores)
    
    fold_no += 1

# Print cross-validation results
print('Cross-Validation results:')
print(f'Average Accuracy: {np.mean(acc_per_fold):.3f} ± {np.std(acc_per_fold):.3f}')
print(f'Average Loss: {np.mean(loss_per_fold):.3f} ± {np.std(loss_per_fold):.3f}')
print(f'Average Precision: {np.mean(precision_per_fold):.3f} ± {np.std(precision_per_fold):.3f}')
print(f'Average Recall: {np.mean(recall_per_fold):.3f} ± {np.std(recall_per_fold):.3f}')



# Background data for explainer, without reshaping to add a channel
background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]

# Create explainer instance with background data that does not include a channel dimension
explainer = shap.DeepExplainer(model, background)

# Select 10 test samples without adding a channel dimension
test_samples_1 = X_val[:1]

# Calculate SHAP values for test samples
shap_values = explainer.shap_values(test_samples_1)

test_samples = np.squeeze(test_samples_1, axis=-1)
shap_values = np.squeeze(shap_values, axis=2)


num_features = test_samples.shape[1]  # Total number of characteristics
wavelengths = np.linspace(200, 1000, num_features)  # Generate wavelengths from 200 nm to 1000 nm
feature_names = [f"{wavelength:.1f} nm" for wavelength in wavelengths]
num_samples = test_samples.shape[0]  # Total number of samples
num_classes = shap_values.shape[2]  # Total number of classes


# Average VIP scores across folds
avg_vip_scores = np.mean(vip_scores_folds, axis=0)
print('Average VIP Scores across folds:', avg_vip_scores)

# Plot VIP scores
plt.figure(figsize=(10, 4))
plt.plot(wavelengths, avg_vip_scores)
plt.title('PLS-DA Feature Importance')
plt.xlabel('Wavelength (nm)')
plt.ylabel('VIP Score')
plt.show()

# Combine correctly classified spectra for class 1 across all folds
correctly_classified_class_1_spectra = np.vstack(correctly_classified_class_1_spectra)

# Calculate the prototype spectrum
prototype_spectrum_class_1 = np.mean(correctly_classified_class_1_spectra, axis=0)

# Plot the prototype spectrum
plt.figure(figsize=(10, 4))
plt.plot(wavelengths, prototype_spectrum_class_1)
plt.title('Prototype Spectrum for Class 1')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity (a.u)')
plt.show()
########################################################################

# Select sample input spectra
sample_indices = [0, 1, 2,50,140,400]  # Adjust indices based on your dataset
sample_spectra = X_train[sample_indices]

# Ensure the sample spectra have the correct shape
sample_spectra = sample_spectra.reshape(sample_spectra.shape[0], sample_spectra.shape[1], 1)

# Plot the sample input spectra
plt.figure(figsize=(12, 6))
for i, spectrum in enumerate(sample_spectra):
    plt.plot(spectrum.squeeze(), label=f'Spectrum {i}')
plt.title('Sample Input Spectra')
plt.xlabel('Wavelength Index')
plt.ylabel('Intensity')
plt.legend()
plt.show()

print(X_train.shape)


model.summary()


############################################################################


# Verify dimensions before proceeding
assert shap_values.shape == (num_samples, num_features, num_classes), "Shape of shap_values is incorrect."
assert test_samples.shape == (num_samples, num_features), "Shape of test_samples is incorrect."
assert len(feature_names) == num_features, "Number of feature names does not match number of features."

## FORCE PLOT

for i in range(num_samples):  # For each test sample
    for c in range(num_classes):  # For each class
        shap_values_class = shap_values[i, :, c]
        base_value = np.mean(model.predict(background)[:, c])

        # Order shap values by their magnitude
        sorted_indices = np.argsort(np.abs(shap_values_class))[::-1]

        # Choose top 5 positive and negative characterisctics
        top_indices = sorted_indices[:10]

        # Create an array with empty etiquetes
        all_feature_names = np.array(feature_names)
        filtered_feature_names = np.array(["" for _ in range(len(all_feature_names))])
        
        # Assign names only to the top 10
        filtered_feature_names[top_indices] = all_feature_names[top_indices]

        # Force plot
        force_plot = shap.force_plot(
            base_value,
            shap_values_class,  # All SHAP valies
            test_samples[i],  # All data
            feature_names=filtered_feature_names,  # Use only filter names
            show=False  
        )

        # Save force plot as HTML
        shap.save_html(f"force_plot_class_{c+1}_sample_{i+1}.html", force_plot)

        # Print label for the class
        print(f"SHAP Values for Class {c+1} - Sample {i+1}")

        
## WATER PLOT



# wavelenghts that u want to show
specific_wavelengths = [279.6, 288.1, 309.2, 317.9, 393.3, 394.4, 396.2, 396.8, 422.7, 589.0, 589.6, 656.3, 769.9, 777.3, 844.6, 854.2, 866.2]
specific_wavelength_names = [f"{wavelength} nm" for wavelength in specific_wavelengths]

# OObtain indices from feature_names
specific_indices = [i for i, name in enumerate(feature_names) if float(name.split(' ')[0]) in specific_wavelengths]

for i in range(num_samples):  
    for c in range(num_classes):  
        shap_values_class = shap_values[i, :, c]
        base_value = np.mean(model.predict(background)[:, c])

        # Filter by wavelenght 
        shap_values_specific = shap_values_class[specific_indices]
        data_specific = test_samples[i][specific_indices]
        feature_names_specific = [feature_names[idx] for idx in specific_indices]

        # Delete duplicates
        unique_contributions = {}
        for idx, feature, contribution in zip(specific_indices, feature_names_specific, shap_values_specific):
            if feature not in unique_contributions:
                unique_contributions[feature] = (idx, contribution)

        # Order by contribution
        sorted_unique_contributions = sorted(unique_contributions.items(), key=lambda item: item[1][1], reverse=True)
        unique_indices = [item[1][0] for item in sorted_unique_contributions]

        # Create Shap explanation
        shap_exp = shap.Explanation(values=shap_values_class[unique_indices], 
                                    base_values=base_value, 
                                    data=test_samples[i][unique_indices], 
                                    feature_names=[feature_names[idx] for idx in unique_indices])

        # new graphic
        plt.figure()  
        shap.plots.waterfall(shap_exp, max_display=10)
        plt.title(f"SHAP Values for Class {c+1} - Sample {i+1}")
        plt.show()  



# Predictions for test samples
predictions = model.predict(test_samples_1)
predicted_classes = np.argmax(predictions, axis=1)

print("Predicted classes:", predicted_classes)








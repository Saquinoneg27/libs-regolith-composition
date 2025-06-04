import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import InterpolatedUnivariateSpline

# Function to calculate Full Width at Half Maximum (FWHM)
def find_fwhm(x, y, peak_index):
    peak_x = x[peak_index]
    peak_y = y[peak_index]
    half_max = peak_y / 2
    
    # Sort data by x to ensure they are in ascending order
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices] - half_max
    
    # Remove duplicate values in x_sorted
    x_sorted, unique_indices = np.unique(x_sorted, return_index=True)
    y_sorted = y_sorted[unique_indices]
    
    spline = InterpolatedUnivariateSpline(x_sorted, y_sorted)
    roots = spline.roots()

    if len(roots) >= 2:
        # Find the closest roots to the main peak on the left and right
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

# Function to process the spectrum
def process_spectrum(x, y, lower_bound, upper_bound):
    mask = (x >= lower_bound) & (x <= upper_bound)
    x_roi = x[mask]
    y_roi = y[mask]
    peaks, _ = find_peaks(y_roi)
    if len(peaks) == 0:
        print("No peaks found in the specified range.")
        return x_roi, y_roi, None, None, None
    main_peak = peaks[np.argmax(y_roi[peaks])]
    fwhm_x = find_fwhm(x_roi, y_roi, main_peak)
    if fwhm_x is not None:
        sum_in_fwhm = np.sum(y_roi[(x_roi >= fwhm_x[0]) & (x_roi <= fwhm_x[1])])
        return x_roi, y_roi, main_peak, fwhm_x, sum_in_fwhm
    else:
        print("FWHM could not be determined.")
        return x_roi, y_roi, main_peak, None, None

# Path to your .asc file
file_path = r'c:\Users\alejo\OneDrive\Desktop\ingeniería física\8vo Brno University\Cilecek\C10\40mJ_1usGD_Mars.asc'

# Initialize lists to store the separated values
wavelengths = []
intensities = []

# Open and read the file line by line
with open(file_path, 'r') as file:
    for line in file:
        # Split the line into parts using the tab separator
        parts = line.strip().split('\t')
        # Check if the line was split into two parts
        if len(parts) == 2:
            wavelength, intensity = parts
            wavelengths.append(float(wavelength))
            intensities.append(int(intensity))

# Create a DataFrame from the lists
df = pd.DataFrame({
    'Wavelength': wavelengths,
    'Intensity': intensities
})

# Assuming each spectrum is 2560 data points long, calculate the total number of spectra
total_spectra = len(df) // 2560

# Identify peaks to avoid in noise calculation
peaks, _ = find_peaks(df['Intensity'], height=np.mean(df['Intensity']))

# Calculate the noise level as the mean intensity of non-peak regions
noise_level = np.mean([df['Intensity'][i] for i in range(len(df['Intensity'])) if i not in peaks]) 

# Subtract the noise level from the intensities
df['Corrected_Intensity'] = df['Intensity'] - noise_level
df['Corrected_Intensity'] = df['Corrected_Intensity'].clip(lower=0)  # Ensure no negative values

# Convert DataFrame to NumPy arrays for processing
wavelengths_np = df['Wavelength'].to_numpy()
intensities_np = df['Corrected_Intensity'].to_numpy()

# Plotting the first 6 spectra
plt.figure(figsize=(14, 8))
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']  # Colors for different spectra

for i in range(6):  # Change 6 to the number of spectra you want to plot
    # Extract each spectrum
    spectrum = df.iloc[i * 2560:(i + 1) * 2560]
    # Plotting
    plt.plot(spectrum['Wavelength'], spectrum['Corrected_Intensity'], color=colors[i], label=f'Spectrum {i + 1}')

plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity (Counts)')
plt.title('First 6 Spectra: Intensity vs. Wavelength')
plt.legend()
plt.grid(True)
plt.show()

# Calculate the average of all spectra
average_spectrum_intensity = []

for i in range(2560):  # Assuming all spectra have the same length
    avg_intensity = df.iloc[i::2560, 2].mean()  # Taking the mean of corrected intensity for each wavelength across all spectra
    average_spectrum_intensity.append(avg_intensity)

# Plot the average spectrum in a new figure
plt.figure(figsize=(14, 8))
plt.plot(df['Wavelength'].iloc[0:2560], average_spectrum_intensity, color='black', label='Average Spectrum')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity (Counts)')
plt.title('Average Spectrum: Intensity vs. Wavelength')
plt.legend()
plt.grid(True)
plt.show()

# Convert the average to a NumPy array
average_spectrum_intensity = np.array(average_spectrum_intensity)
sum_average_spectrum_intensity = sum(average_spectrum_intensity)
average_spectrum_intensity = average_spectrum_intensity / sum_average_spectrum_intensity

# Process the average spectrum to find the peak and calculate the FWHM in the desired range
lower_bound = 340.25
upper_bound = 342

x_roi_avg, y_roi_avg, main_peak_avg, fwhm_x_avg, sum_in_fwhm_avg = process_spectrum(wavelengths_np[:2560], average_spectrum_intensity, lower_bound, upper_bound)

if fwhm_x_avg is not None:
    plt.figure(figsize=(14, 8))
    plt.plot(x_roi_avg, y_roi_avg, label='Average Spectrum')
    plt.axvline(x=x_roi_avg[main_peak_avg], color='r', linestyle='--', label='Main Peak')
    plt.fill_between(x_roi_avg, 0, y_roi_avg, where=(x_roi_avg >= fwhm_x_avg[0]) & (x_roi_avg <= fwhm_x_avg[1]), color='gray', alpha=0.5, label='FWHM')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity (Counts)')
    plt.title('Average Spectrum with Main Peak and FWHM')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print(f"Main peak at: {x_roi_avg[main_peak_avg]}")
    print(f"FWHM range: {fwhm_x_avg[0]} - {fwhm_x_avg[1]}")
    print(f"Sum in FWHM: {sum_in_fwhm_avg}")
else:
    print("No valid peak found within the specified range.")

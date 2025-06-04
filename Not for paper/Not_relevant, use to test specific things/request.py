import requests
from bs4 import BeautifulSoup

# Elements to search for
elements = ["Si", "Al", "Ca", "Fe", "K", "Mg", "Mn", "P", "Ti", "S", "Na"]

# Base URL for the NIST emission line form
form_url = "https://physics.nist.gov/PhysRefData/ASD/LIBS/libs-form.html"

# Function to get emission lines for an element
def get_emission_lines(element):
  """Fetches emission lines data for a given element from NIST website.

  Args:
      element: String representing the element symbol (e.g., "Si").

  Returns:
      list: A list of tuples containing (wavelength, intensity) for each emission line.
  """

  # Form data for the NIST website
  data = {
      'spectra': element,
      'perc': 100,  # Percentage of element in sample (assumed 100%)
      'low_w': 200,  # Lower wavelength limit (nm)
      'upp_w': 600,  # Upper wavelength limit (nm)
      'unit': 0,     # Wavelength unit (nm)
      'resolution': 1000,  # Spectral resolution
      'temp': 1,     # Electron temperature (assumed 1)
      'e_density': 1e17,  # Electron density (assumed 1e17 cm^-3)
      'format': 1,  # Output format (HTML)
      'libs': 'on',  # Include LIBS data
      'en_unit': '0',
      'output': '0'
  }

  # Send POST request with form data
  response = requests.post(form_url, data=data)

  # Raise an error for unsuccessful requests
  response.raise_for_status()

  # Parse the HTML response
  soup = BeautifulSoup(response.text, 'html.parser')

  # Extract emission lines data
  lines = []
  table = soup.find('pre')
  if table:
    rows = table.text.splitlines()[4:]  # Skip header lines
    for row in rows:
      columns = row.split()
      if len(columns) >= 2:
        wavelength = columns[0]
        intensity = columns[1]
        lines.append((wavelength, intensity))
  return lines

# Dictionary to store emission lines for each element
emission_lines_dict = {}

# Get emission lines for each element and store in the dictionary
for element in elements:
  print(f"Fetching data for element: {element}")
  emission_lines_dict[element] = get_emission_lines(element)

# Print the emission line data (optional)
for element, lines in emission_lines_dict.items():
  print(f"Element: {element}")
  for line in lines:
    print(f"Wavelength: {line[0]}, Intensity: {line[1]}")
  print("\n")

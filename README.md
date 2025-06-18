# LIBS Regolith Composition Analysis

This repository contains the code and data for analyzing Laser-Induced Breakdown Spectroscopy (LIBS) data from lunar and Martian regolith simulants. The work is a collaboration between researchers at Brno University of Technology and focuses on quantitative calibration models for predicting chemical compound concentrations in planetary samples.

## Project Structure

```
.
├── Spectra/
│   ├── Earth/
│   ├── Mars/
│   ├── Vacuum/
│   └── *.xlsx, *.txt files
├── Codes/
│   ├── CNN_Classification_R/
│   ├── MLP_Composition_R/
│   ├── PCR/
│   └── PLS/
```

## Data Provenance

The spectral data used in this project comes from LIBS measurements of various lunar and Martian regolith simulants. The measurements were conducted at Brno University of Technology using standardized LIBS protocols. The data includes:

- Raw LIBS spectral files (wavelength vs intensity)
- Reference concentration data for various compounds
- Theoretical line libraries for elemental analysis

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/libs-regolith-composition.git
cd libs-regolith-composition
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Each analysis module can be run independently:

```bash
# For CNN Classification
python Codes/CNN_Classification_R/CNN.py

# For MLP Composition Analysis
python Codes/MLP_Composition_R/MLP_hyperband_alg.py

# For Principal Component Regression
python Codes/PCR/PCR.py

# For Partial Least Squares Analysis
python Codes/PLS/PLS.py
```

## Citation

If you use this code or data in your research, once published a citation will be provided:



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## Acknowledgments

This work was conducted at Brno University of Technology. We thank all contributors and researchers involved in the data collection and analysis process. 

# TIW SST Forecasting

[![Tests](https://github.com/username/tiw-sst-forecasting/actions/workflows/tests.yml/badge.svg)](https://github.com/username/tiw-sst-forecasting/actions/workflows/tests.yml)
[![Documentation](https://github.com/username/tiw-sst-forecasting/actions/workflows/docs.yml/badge.svg)](https://github.com/username/tiw-sst-forecasting/actions/workflows/docs.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Machine learning models for forecasting Tropical Instability Wave (TIW) sea surface temperature patterns in the tropical Pacific Ocean.

## Overview

Tropical Instability Waves (TIWs) are large-scale oceanic phenomena that play a crucial role in Pacific climate variability. This project develops deep learning models to predict TIW-driven SST anomalies using high-resolution MITgcm simulation data.

### Algorithm

The forecasting approach uses a **sequence-to-sequence** architecture:

1. **Input**: 14 days of daily SST anomaly fields from a large Pacific region (10°N-10°S, 150°W-110°W)
2. **Output**: 5-day forecast of SST anomalies in the TIW-active region (5°N-3°S, 145°W-135°W)  
3. **Architecture**: Spatiotemporal neural networks (CNN-LSTM, ConvLSTM, or Transformer-based)
4. **Training**: Sliding window approach with 3-day stride generates ~116 sequences from 1-year MITgcm data

**Key Innovation**: Uses broader regional context to predict localized TIW evolution, capturing cross-scale interactions between large-scale ocean dynamics and mesoscale TIW features.

## Features

- 🌊 **MITgcm Data Processing**: Automated pipeline for THETA variable extraction and preprocessing
- 📐 **Configurable Regions**: Flexible input/output domain specification  
- 🕰️ **Temporal Windowing**: Sliding window sequence generation with configurable stride
- 🗜️ **HDF5 Storage**: Optimized data format with 30% compression and fast I/O
- 🧠 **ML-Ready Output**: PyTorch tensors with standardized anomaly computation
- 🔬 **Comprehensive Testing**: Unit tests covering all preprocessing functions
- 📚 **Documentation**: Sphinx-generated API docs with docstring integration

## Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/username/tiw-sst-forecasting.git
cd tiw-sst-forecasting

# Create conda environment
conda env create -f environment.yaml
conda activate tiw-sst-forecasting
```

### 2. Data Preprocessing

```python
from data.preprocessing import preprocess_tiw_data
from pathlib import Path

# Run preprocessing pipeline
results = preprocess_tiw_data(
    raw_data_path=Path("data/raw"),
    processed_data_path=Path("data/processed"), 
    input_netcdf_filename="TPOSE6_Daily_2012_surface.nc",
    input_region_bounds={
        'lat_min': -10.0, 'lat_max': 10.0,
        'lon_min': 210.0, 'lon_max': 250.0  # 150W-110W
    },
    output_region_bounds={
        'lat_min': -3.0, 'lat_max': 5.0,
        'lon_min': 215.0, 'lon_max': 225.0  # 145W-135W
    }
)

print(f"Generated {results['total_sequences']} training sequences")
```

### 3. Model Development

```python
from data.preprocessing import load_training_data_hdf5
from pathlib import Path

# Load preprocessed data (HDF5 format for optimal performance)
data = load_training_data_hdf5(Path("data/processed"))
input_sequences = data['input_sequences']   # Shape: (samples, 14, lat, lon)
output_sequences = data['output_sequences'] # Shape: (samples, 5, lat, lon)

# Use with PyTorch DataLoader for training
from torch.utils.data import DataLoader, TensorDataset
dataset = TensorDataset(input_sequences, output_sequences)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Implement your spatiotemporal architecture here
```

## Project Structure

```
tiw-sst-forecasting/
├── data/
│   ├── raw/                    # Raw MITgcm NetCDF files
│   ├── processed/              # Preprocessed PyTorch tensors
│   └── interim/                # Intermediate processing files
├── models/                     # ML model implementations
├── training/                   # Training scripts and utilities  
├── experiments/                # Experiment configurations
├── notebooks/                  # Jupyter notebooks for exploration
│   ├── 01_data_preprocessing.ipynb
│   └── 02_ml_prototyping.ipynb
├── docs/                       # Sphinx documentation
├── preprocessing.py            # Core preprocessing module
├── preprocessing_test.py       # Unit tests
├── environment.yaml            # Conda environment
└── README.md
```

## Testing

Run the full test suite:

```bash
pytest data/preprocessing_test.py -v
```

Tests cover:
- ✅ NetCDF data loading and validation
- ✅ Regional cropping with bounds checking  
- ✅ SST anomaly computation
- ✅ Training sequence generation
- ✅ PyTorch data serialization
- ✅ End-to-end preprocessing pipeline

## Documentation

View the full API documentation:

```bash
cd docs
make html
open build/html/index.rst  # macOS
```

Or visit the [online documentation](https://username.github.io/tiw-sst-forecasting/).

## Data Requirements

- **Format**: NetCDF files from MITgcm simulations
- **Variable**: `THETA` (potential temperature) with coordinates `(time, Z, YC, XC)`
- **Resolution**: Daily timesteps recommended for TIW dynamics
- **Domain**: Tropical Pacific region covering TIW-active areas
- **Output**: HDF5 format with gzip compression (~30% smaller than PyTorch .pt files)
- **Example**: TPOSE6_Daily_2012_surface.nc (366 timesteps, 84×240 spatial grid)

## Development

### Adding New Features

1. **Functions**: Add to `data/preprocessing.py` with comprehensive docstrings
2. **Tests**: Create corresponding tests in `data/preprocessing_test.py`  
3. **Documentation**: Docstrings automatically generate API docs
4. **CI/CD**: Tests run automatically on commits via GitHub Actions

### Code Quality

- **Type Hints**: All functions include comprehensive type annotations
- **Error Handling**: Robust validation with descriptive error messages
- **Documentation**: Google-style docstrings for Sphinx compatibility
- **Testing**: pytest with fixtures for comprehensive coverage

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new functionality
4. Ensure all tests pass (`pytest`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Citation

If you use this code in your research, please cite:

```bibtex
@software{tiw_sst_forecasting_2024,
  title={TIW SST Forecasting: Machine Learning for Tropical Instability Wave Prediction},
  author={ML Team},
  year={2024},
  url={https://github.com/username/tiw-sst-forecasting}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- MITgcm team for ocean simulation framework
- TPOSE project for tropical Pacific data
- PyTorch community for deep learning tools
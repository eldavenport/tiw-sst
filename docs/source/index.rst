TIW SST Forecasting Documentation
==================================

Welcome to the TIW SST Forecasting project documentation. This project develops machine learning models for predicting Tropical Instability Wave (TIW) sea surface temperature patterns in the tropical Pacific Ocean.

Overview
--------

Tropical Instability Waves are large-scale oceanic features that significantly influence Pacific climate variability. This project uses deep learning to forecast TIW-driven SST anomalies using historical MITgcm simulation data.

**Key Features:**

* Preprocessing pipeline for MITgcm NetCDF data
* Configurable regional and temporal windowing
* HDF5 data storage with optimal compression and performance
* PyTorch-compatible training data generation
* SST anomaly computation and sequence generation
* Comprehensive unit testing and documentation

Quick Start
-----------

1. **Environment Setup**

   .. code-block:: bash

      conda env create -f environment.yaml
      conda activate tiw-sst-forecasting

2. **Data Preprocessing**

   .. code-block:: python

      from data.preprocessing import preprocess_tiw_data
      from pathlib import Path

      results = preprocess_tiw_data(
          raw_data_path=Path("data/raw"),
          processed_data_path=Path("data/processed"),
          input_netcdf_filename="TPOSE6_Daily_2012_surface.nc",
          input_region_bounds={'lat_min': -10, 'lat_max': 10, 'lon_min': 210, 'lon_max': 250},
          output_region_bounds={'lat_min': -3, 'lat_max': 5, 'lon_min': 215, 'lon_max': 225}
      )

3. **Model Development**

   Load the optimized HDF5 training data for ML model training:

   .. code-block:: python

      from data.preprocessing import load_training_data_hdf5
      from pathlib import Path

      # Load HDF5 data (30% smaller, faster I/O)
      data = load_training_data_hdf5(Path("data/processed"))
      input_sequences = data['input_sequences']
      output_sequences = data['output_sequences']

API Reference
-------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Testing
-------

Run the test suite:

.. code-block:: bash

   pytest preprocessing_test.py -v


#!/usr/bin/env python3
"""
Unit tests for TIW SST preprocessing module.
"""

import pytest
import numpy as np
import xarray as xr
import pandas as pd
import torch
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from .preprocessing import (
    load_theta_data, crop_region, compute_sst_anomalies,
    generate_training_sequences, save_training_data_hdf5, 
    load_training_data_hdf5, save_pytorch_training_data,
    preprocess_tiw_data
)


@pytest.fixture
def sample_theta_data():
    """Create sample THETA data for testing."""
    # Create synthetic data with known properties
    time_steps = 30
    lat_points = 20
    lon_points = 30
    
    # Create coordinates
    time = pd.date_range('2012-01-01', periods=time_steps, freq='D')
    lat = np.linspace(-5, 5, lat_points)
    lon = np.linspace(215, 225, lon_points)
    
    # Create synthetic SST data with realistic temperature values
    np.random.seed(42)  # For reproducible tests
    base_temp = 25.0
    temp_data = base_temp + 2.0 * np.random.randn(time_steps, lat_points, lon_points)
    
    # Create xarray DataArray
    theta_data = xr.DataArray(
        temp_data,
        coords={'time': time, 'YC': lat, 'XC': lon},
        dims=['time', 'YC', 'XC'],
        name='THETA'
    )
    
    return theta_data


@pytest.fixture
def temp_netcdf_file(sample_theta_data):
    """Create temporary NetCDF file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp_file:
        # Create dataset with THETA variable
        dataset = xr.Dataset({'THETA': sample_theta_data})
        dataset.to_netcdf(tmp_file.name)
        tmp_file_path = Path(tmp_file.name)
    
    yield tmp_file_path
    
    # Cleanup
    if tmp_file_path.exists():
        tmp_file_path.unlink()


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    raw_data_path = temp_dir / "raw"
    processed_data_path = temp_dir / "processed"
    
    raw_data_path.mkdir()
    processed_data_path.mkdir()
    
    yield raw_data_path, processed_data_path
    
    # Cleanup
    shutil.rmtree(temp_dir)


class TestLoadThetaData:
    """Test load_theta_data function."""
    
    def test_load_theta_data_success(self, temp_netcdf_file):
        """Test successful loading of THETA data."""
        theta_data = load_theta_data(temp_netcdf_file)
        
        assert isinstance(theta_data, xr.DataArray)
        assert theta_data.name == 'THETA'
        assert 'time' in theta_data.dims
        assert 'YC' in theta_data.dims
        assert 'XC' in theta_data.dims
    
    def test_load_theta_data_file_not_found(self):
        """Test error when NetCDF file doesn't exist."""
        nonexistent_path = Path("nonexistent_file.nc")
        
        with pytest.raises(FileNotFoundError):
            load_theta_data(nonexistent_path)
    
    def test_load_theta_data_missing_theta_variable(self, temp_dirs):
        """Test error when THETA variable is missing."""
        raw_data_path, _ = temp_dirs
        
        # Create NetCDF file without THETA variable
        dummy_data = xr.DataArray(
            np.random.randn(10, 5, 5),
            dims=['time', 'lat', 'lon'],
            name='OTHER_VAR'
        )
        dataset = xr.Dataset({'OTHER_VAR': dummy_data})
        test_file = raw_data_path / "no_theta.nc"
        dataset.to_netcdf(test_file)
        
        with pytest.raises(KeyError, match="THETA variable not found"):
            load_theta_data(test_file)


class TestCropRegion:
    """Test crop_region function."""
    
    def test_crop_region_success(self, sample_theta_data):
        """Test successful regional cropping."""
        lat_min, lat_max = -2.0, 2.0
        lon_min, lon_max = 217.0, 223.0
        
        cropped_data = crop_region(sample_theta_data, lat_min, lat_max, lon_min, lon_max)
        
        assert isinstance(cropped_data, xr.DataArray)
        assert cropped_data.shape[0] == sample_theta_data.shape[0]  # Same time dimension
        assert cropped_data.shape[1] < sample_theta_data.shape[1]   # Fewer lat points
        assert cropped_data.shape[2] < sample_theta_data.shape[2]   # Fewer lon points
        
        # Check that coordinates are within bounds
        assert cropped_data.YC.min() >= lat_min
        assert cropped_data.YC.max() <= lat_max
        assert cropped_data.XC.min() >= lon_min
        assert cropped_data.XC.max() <= lon_max
    
    def test_crop_region_no_data_in_bounds(self, sample_theta_data):
        """Test error when no data points fall within bounds."""
        # Use bounds that don't intersect with data
        lat_min, lat_max = 50.0, 60.0
        lon_min, lon_max = 300.0, 310.0
        
        with pytest.raises(ValueError, match="No data points found within bounds"):
            crop_region(sample_theta_data, lat_min, lat_max, lon_min, lon_max)


class TestComputeSSTAnomalies:
    """Test compute_sst_anomalies function."""
    
    def test_compute_sst_anomalies(self, sample_theta_data):
        """Test SST anomaly computation."""
        anomalies = compute_sst_anomalies(sample_theta_data)
        
        assert isinstance(anomalies, xr.DataArray)
        assert anomalies.shape == sample_theta_data.shape
        
        # Check that temporal mean is approximately zero
        temporal_mean = anomalies.mean(dim='time')
        assert np.allclose(temporal_mean.values, 0.0, atol=1e-10)


class TestGenerateTrainingSequences:
    """Test generate_training_sequences function."""
    
    def test_generate_training_sequences_success(self, sample_theta_data):
        """Test successful sequence generation."""
        input_anomalies = compute_sst_anomalies(sample_theta_data)
        output_anomalies = input_anomalies  # Use same data for simplicity
        
        input_length = 5
        output_length = 3
        stride = 2
        
        input_seq, output_seq, dates = generate_training_sequences(
            input_anomalies, output_anomalies, input_length, output_length, stride
        )
        
        expected_sequences = (len(sample_theta_data.time) - input_length - output_length) // stride + 1
        
        assert input_seq.shape[0] == expected_sequences
        assert output_seq.shape[0] == expected_sequences
        assert input_seq.shape[1] == input_length
        assert output_seq.shape[1] == output_length
        assert len(dates) == expected_sequences
        
        # Check DataFrame structure
        assert 'sequence_id' in dates.columns
        assert 'start_date' in dates.columns
        assert 'end_date' in dates.columns
    
    def test_generate_training_sequences_data_too_short(self, sample_theta_data):
        """Test error when data is too short for sequences."""
        input_anomalies = compute_sst_anomalies(sample_theta_data)
        output_anomalies = input_anomalies
        
        # Request sequences longer than available data
        input_length = 20
        output_length = 15
        stride = 1
        
        with pytest.raises(ValueError, match="Data too short"):
            generate_training_sequences(
                input_anomalies, output_anomalies, input_length, output_length, stride
            )


class TestSavePytorchTrainingData:
    """Test save_pytorch_training_data function."""
    
    def test_save_pytorch_training_data(self, temp_dirs):
        """Test saving training data."""
        _, processed_data_path = temp_dirs
        
        # Create sample data
        input_sequences = np.random.randn(10, 5, 8, 12)
        output_sequences = np.random.randn(10, 3, 6, 8)
        sequence_dates = pd.DataFrame({
            'sequence_id': range(10),
            'start_date': pd.date_range('2012-01-01', periods=10),
            'end_date': pd.date_range('2012-01-08', periods=10)
        })
        
        input_region_bounds = {'lat_min': -5, 'lat_max': 5, 'lon_min': 210, 'lon_max': 220}
        output_region_bounds = {'lat_min': -3, 'lat_max': 3, 'lon_min': 215, 'lon_max': 218}
        sequence_config = {'input_length_days': 5, 'output_length_days': 3, 'stride_days': 1}
        
        save_pytorch_training_data(
            input_sequences, output_sequences, sequence_dates,
            input_region_bounds, output_region_bounds, sequence_config,
            processed_data_path
        )
        
        # Check files were created (CSV no longer created in updated version)
        assert (processed_data_path / 'tiw_sst_training_data.pt').exists()
        
        # Check loaded data (allow pandas DataFrame for sequence metadata)
        loaded_data = torch.load(processed_data_path / 'tiw_sst_training_data.pt', weights_only=False)
        assert 'input_sequences' in loaded_data
        assert 'output_sequences' in loaded_data
        assert 'sequence_metadata' in loaded_data
        assert 'input_region_bounds' in loaded_data
        assert 'output_region_bounds' in loaded_data
        assert 'sequence_config' in loaded_data
        
        # Check tensor types
        assert isinstance(loaded_data['input_sequences'], torch.Tensor)
        assert isinstance(loaded_data['output_sequences'], torch.Tensor)
        assert loaded_data['input_sequences'].dtype == torch.float32
        assert loaded_data['output_sequences'].dtype == torch.float32


class TestPreprocessTiwData:
    """Test complete preprocessing pipeline."""
    
    def test_preprocess_tiw_data_integration(self, temp_dirs, sample_theta_data):
        """Test complete preprocessing pipeline."""
        raw_data_path, processed_data_path = temp_dirs
        
        # Create test NetCDF file
        dataset = xr.Dataset({'THETA': sample_theta_data})
        test_file = raw_data_path / "test_data.nc"
        dataset.to_netcdf(test_file)
        
        # Define region bounds that work with sample data
        input_region_bounds = {
            'lat_min': -4.0, 'lat_max': 4.0,
            'lon_min': 216.0, 'lon_max': 224.0
        }
        output_region_bounds = {
            'lat_min': -2.0, 'lat_max': 2.0,
            'lon_min': 218.0, 'lon_max': 222.0
        }
        
        # Run preprocessing
        results = preprocess_tiw_data(
            raw_data_path, processed_data_path, "test_data.nc",
            input_region_bounds, output_region_bounds,
            input_sequence_length_days=5,
            output_sequence_length_days=3,
            stride_days=2
        )
        
        # Check results structure
        assert 'total_sequences' in results
        assert 'input_shape' in results
        assert 'output_shape' in results
        assert 'input_anomaly_stats' in results
        assert 'output_anomaly_stats' in results
        assert 'nan_values' in results
        
        # Check that files were created (now HDF5 format)
        assert (processed_data_path / 'tiw_sst_training_data.h5').exists()
        
        # Check that no NaN values were introduced
        assert results['nan_values']['input'] == 0
        assert results['nan_values']['output'] == 0


class TestSaveTrainingDataHDF5:
    """Test save_training_data_hdf5 function."""
    
    def test_save_training_data_hdf5(self, temp_dirs):
        """Test saving training data in HDF5 format."""
        _, processed_data_path = temp_dirs
        
        # Create sample data
        input_sequences = np.random.randn(10, 5, 8, 12).astype(np.float32)
        output_sequences = np.random.randn(10, 3, 6, 8).astype(np.float32)
        sequence_dates = pd.DataFrame({
            'sequence_id': range(10),
            'start_date': pd.date_range('2012-01-01', periods=10),
            'end_date': pd.date_range('2012-01-08', periods=10)
        })
        
        input_region_bounds = {'lat_min': -5, 'lat_max': 5, 'lon_min': 210, 'lon_max': 220}
        output_region_bounds = {'lat_min': -3, 'lat_max': 3, 'lon_min': 215, 'lon_max': 218}
        sequence_config = {'input_length_days': 5, 'output_length_days': 3, 'stride_days': 1}
        
        save_training_data_hdf5(
            input_sequences, output_sequences, sequence_dates,
            input_region_bounds, output_region_bounds, sequence_config,
            processed_data_path
        )
        
        # Check files were created (CSV no longer created, metadata embedded in HDF5)
        assert (processed_data_path / 'tiw_sst_training_data.h5').exists()
        
        # Verify HDF5 file contents
        import h5py
        with h5py.File(processed_data_path / 'tiw_sst_training_data.h5', 'r') as f:
            assert 'input_sequences' in f
            assert 'output_sequences' in f
            assert f['input_sequences'].shape == input_sequences.shape
            assert f['output_sequences'].shape == output_sequences.shape
            assert f['input_sequences'].dtype == np.float32
            assert f['output_sequences'].dtype == np.float32
            
            # Check attributes
            assert 'input_region_bounds' in f.attrs
            assert 'output_region_bounds' in f.attrs
            assert 'sequence_config' in f.attrs
            assert 'input_shape' in f.attrs
            assert 'output_shape' in f.attrs
            assert 'total_sequences' in f.attrs


class TestLoadTrainingDataHDF5:
    """Test load_training_data_hdf5 function."""
    
    def test_load_training_data_hdf5_success(self, temp_dirs):
        """Test successful loading of HDF5 training data."""
        _, processed_data_path = temp_dirs
        
        # Create and save sample data first
        input_sequences = np.random.randn(5, 3, 4, 6).astype(np.float32)
        output_sequences = np.random.randn(5, 2, 3, 4).astype(np.float32)
        sequence_dates = pd.DataFrame({
            'sequence_id': range(5),
            'start_date': pd.date_range('2012-01-01', periods=5),
            'end_date': pd.date_range('2012-01-05', periods=5)
        })
        
        input_region_bounds = {'lat_min': -5, 'lat_max': 5, 'lon_min': 210, 'lon_max': 220}
        output_region_bounds = {'lat_min': -3, 'lat_max': 3, 'lon_min': 215, 'lon_max': 218}
        sequence_config = {'input_length_days': 3, 'output_length_days': 2, 'stride_days': 1}
        
        save_training_data_hdf5(
            input_sequences, output_sequences, sequence_dates,
            input_region_bounds, output_region_bounds, sequence_config,
            processed_data_path
        )
        
        # Load the data
        loaded_data = load_training_data_hdf5(processed_data_path)
        
        # Check loaded data structure
        assert 'input_sequences' in loaded_data
        assert 'output_sequences' in loaded_data
        assert 'sequence_metadata' in loaded_data
        assert 'input_region_bounds' in loaded_data
        assert 'output_region_bounds' in loaded_data
        assert 'sequence_config' in loaded_data
        assert 'input_shape' in loaded_data
        assert 'output_shape' in loaded_data
        assert 'total_sequences' in loaded_data
        
        # Check tensor types and values
        assert isinstance(loaded_data['input_sequences'], torch.Tensor)
        assert isinstance(loaded_data['output_sequences'], torch.Tensor)
        assert loaded_data['input_sequences'].dtype == torch.float32
        assert loaded_data['output_sequences'].dtype == torch.float32
        
        # Check shapes match
        assert loaded_data['input_sequences'].shape == input_sequences.shape
        assert loaded_data['output_sequences'].shape == output_sequences.shape
        
        # Check metadata
        assert loaded_data['input_region_bounds'] == input_region_bounds
        assert loaded_data['output_region_bounds'] == output_region_bounds
        assert loaded_data['sequence_config'] == sequence_config
        assert loaded_data['total_sequences'] == 5
        
        # Check that values are approximately equal (allowing for compression artifacts)
        np.testing.assert_array_almost_equal(
            loaded_data['input_sequences'].numpy(), input_sequences, decimal=5
        )
        np.testing.assert_array_almost_equal(
            loaded_data['output_sequences'].numpy(), output_sequences, decimal=5
        )
    
    def test_load_training_data_hdf5_file_not_found(self, temp_dirs):
        """Test error when HDF5 file doesn't exist."""
        _, processed_data_path = temp_dirs
        
        with pytest.raises(FileNotFoundError, match="HDF5 file not found"):
            load_training_data_hdf5(processed_data_path)


class TestBackwardCompatibility:
    """Test backward compatibility of old PyTorch format."""
    
    def test_pytorch_format_still_works(self, temp_dirs):
        """Test that old PyTorch format functions still work but show deprecation warning."""
        _, processed_data_path = temp_dirs
        
        # Create sample data
        input_sequences = np.random.randn(3, 2, 4, 5)
        output_sequences = np.random.randn(3, 1, 2, 3)
        sequence_dates = pd.DataFrame({
            'sequence_id': range(3),
            'start_date': pd.date_range('2012-01-01', periods=3),
            'end_date': pd.date_range('2012-01-03', periods=3)
        })
        
        input_region_bounds = {'lat_min': -5, 'lat_max': 5, 'lon_min': 210, 'lon_max': 220}
        output_region_bounds = {'lat_min': -3, 'lat_max': 3, 'lon_min': 215, 'lon_max': 218}
        sequence_config = {'input_length_days': 2, 'output_length_days': 1, 'stride_days': 1}
        
        # Test that deprecated function works but shows warning
        with pytest.warns(DeprecationWarning, match="save_pytorch_training_data is deprecated"):
            save_pytorch_training_data(
                input_sequences, output_sequences, sequence_dates,
                input_region_bounds, output_region_bounds, sequence_config,
                processed_data_path
            )
        
        # Check that files were created (CSV no longer created in updated version)
        assert (processed_data_path / 'tiw_sst_training_data.pt').exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
#!/usr/bin/env python3
"""
TIW SST Data Preprocessing

This module preprocesses MITgcm SST data for machine learning forecasting of
Tropical Instability Waves (TIW).
"""

import json
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import xarray as xr
from tqdm import tqdm

warnings.filterwarnings("ignore")


def load_theta_data(netcdf_file_path: Path) -> xr.DataArray:
    """
    Load THETA (SST) data from NetCDF file and remove singleton Z dimension.

    Args:
        netcdf_file_path: Path to NetCDF file containing THETA data

    Returns:
        xarray DataArray with THETA data (time, YC, XC)

    Raises:
        FileNotFoundError: If NetCDF file doesn't exist
        KeyError: If THETA variable not found in dataset
    """
    if not netcdf_file_path.exists():
        raise FileNotFoundError(f"NetCDF file not found: {netcdf_file_path}")

    dataset = xr.open_dataset(netcdf_file_path)

    if "THETA" not in dataset.data_vars:
        raise KeyError("THETA variable not found in dataset")

    # Extract THETA and remove Z dimension if present
    theta_data = dataset["THETA"]
    if "Z" in theta_data.dims:
        theta_data = theta_data.squeeze("Z")

    return theta_data


def crop_region(
    theta_data: xr.DataArray,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> xr.DataArray:
    """
    Crop THETA data to specified lat/lon bounds.

    Args:
        theta_data: xarray DataArray with XC, YC coordinates
        lat_min: Minimum latitude (degrees)
        lat_max: Maximum latitude (degrees)
        lon_min: Minimum longitude (degrees)
        lon_max: Maximum longitude (degrees)

    Returns:
        Cropped xarray DataArray

    Raises:
        ValueError: If no data points fall within specified bounds
    """
    # Create boolean masks for lat/lon selection
    lat_mask = (theta_data.YC >= lat_min) & (theta_data.YC <= lat_max)
    lon_mask = (theta_data.XC >= lon_min) & (theta_data.XC <= lon_max)

    # Apply spatial cropping
    cropped_data = theta_data.where(lat_mask & lon_mask, drop=True)

    # Check if any data remains
    if cropped_data.size == 0:
        raise ValueError(
            f"No data points found within bounds: "
            f"lat=[{lat_min}, {lat_max}], lon=[{lon_min}, {lon_max}]"
        )

    return cropped_data


def compute_sst_anomalies(theta_data: xr.DataArray) -> xr.DataArray:
    """
    Compute SST anomalies relative to temporal mean at each grid point.

    Args:
        theta_data: xarray DataArray with THETA (SST) data

    Returns:
        xarray DataArray of SST anomalies
    """
    # Compute climatological mean at each grid point
    theta_climatology = theta_data.mean(dim="time")

    # Compute anomalies
    theta_anomalies = theta_data - theta_climatology

    return theta_anomalies


def generate_training_sequences(
    input_anomalies: xr.DataArray,
    output_anomalies: xr.DataArray,
    input_length: int,
    output_length: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Generate training sequences with sliding window approach.

    Args:
        input_anomalies: Input region SST anomalies (time, lat, lon)
        output_anomalies: Output region SST anomalies (time, lat, lon)
        input_length: Number of days for input sequence
        output_length: Number of days for output sequence
        stride: Stride between sequences in days

    Returns:
        Tuple of (input_sequences, output_sequences, sequence_dates)
        - input_sequences: numpy array (samples, time, lat, lon)
        - output_sequences: numpy array (samples, time, lat, lon)
        - sequence_dates: DataFrame with sequence metadata

    Raises:
        ValueError: If sequences would be too short given the data length
    """
    total_time_steps = len(input_anomalies.time)
    sequence_length = input_length + output_length

    if total_time_steps < sequence_length:
        raise ValueError(
            f"Data too short: {total_time_steps} steps < {sequence_length} required"
        )

    # Calculate number of possible sequences
    num_sequences = (total_time_steps - sequence_length) // stride + 1

    if num_sequences <= 0:
        raise ValueError(f"No sequences possible with given parameters")

    # Initialize lists to store sequences
    input_sequences = []
    output_sequences = []
    sequence_start_dates = []
    sequence_end_dates = []

    for sequence_idx in tqdm(range(num_sequences), desc="Generating sequences"):
        start_idx = sequence_idx * stride
        input_end_idx = start_idx + input_length
        output_end_idx = input_end_idx + output_length

        # Extract input and output sequences
        input_seq = input_anomalies.isel(time=slice(start_idx, input_end_idx))
        output_seq = output_anomalies.isel(time=slice(input_end_idx, output_end_idx))

        # Store sequences as numpy arrays
        input_sequences.append(input_seq.values)
        output_sequences.append(output_seq.values)

        # Store date information
        start_date = input_anomalies.time[start_idx].values
        end_date = output_anomalies.time[output_end_idx - 1].values
        sequence_start_dates.append(start_date)
        sequence_end_dates.append(end_date)

    # Convert to numpy arrays
    input_sequences = np.array(input_sequences)
    output_sequences = np.array(output_sequences)

    # Create DataFrame with sequence metadata
    sequence_dates = pd.DataFrame(
        {
            "sequence_id": range(num_sequences),
            "start_date": sequence_start_dates,
            "end_date": sequence_end_dates,
        }
    )

    return input_sequences, output_sequences, sequence_dates


def save_training_data_hdf5(
    input_sequences: np.ndarray,
    output_sequences: np.ndarray,
    sequence_dates: pd.DataFrame,
    input_region_bounds: Dict[str, float],
    output_region_bounds: Dict[str, float],
    sequence_config: Dict[str, int],
    save_path: Path,
) -> None:
    """
    Save training data in HDF5 format for optimal performance and compression.

    Args:
        input_sequences: Input sequence array (samples, time, lat, lon)
        output_sequences: Output sequence array (samples, time, lat, lon)
        sequence_dates: DataFrame with sequence date metadata
        input_region_bounds: Dictionary with input region bounds
        output_region_bounds: Dictionary with output region bounds
        sequence_config: Dictionary with sequence configuration
        save_path: Path to save processed data
    """
    save_path.mkdir(exist_ok=True)

    # Save arrays in HDF5 format with compression
    hdf5_path = save_path / "tiw_sst_training_data.h5"

    with h5py.File(hdf5_path, "w") as f:
        # Save input and output sequences with gzip compression
        f.create_dataset(
            "input_sequences",
            data=input_sequences.astype(np.float32),
            compression="gzip",
            compression_opts=6,
            chunks=True,
        )

        f.create_dataset(
            "output_sequences",
            data=output_sequences.astype(np.float32),
            compression="gzip",
            compression_opts=6,
            chunks=True,
        )

        # Save sequence metadata as datasets (better than CSV)
        f.create_dataset("sequence_ids", data=sequence_dates["sequence_id"].values)

        # Convert timestamps to strings for HDF5 storage
        start_dates_str = sequence_dates["start_date"].astype(str).values
        end_dates_str = sequence_dates["end_date"].astype(str).values

        f.create_dataset(
            "start_dates",
            data=start_dates_str.astype("S32"),  # 32-char strings
            compression="gzip",
        )
        f.create_dataset(
            "end_dates", data=end_dates_str.astype("S32"), compression="gzip"
        )

        # Save configuration metadata as attributes
        f.attrs["input_region_bounds"] = json.dumps(input_region_bounds)
        f.attrs["output_region_bounds"] = json.dumps(output_region_bounds)
        f.attrs["sequence_config"] = json.dumps(sequence_config)

        # Save array shapes and data types for easy access
        f.attrs["input_shape"] = input_sequences.shape
        f.attrs["output_shape"] = output_sequences.shape
        f.attrs["total_sequences"] = len(sequence_dates)
        f.attrs["created_date"] = pd.Timestamp.now().isoformat()
        f.attrs["format_version"] = "2.0"  # Track format version


def load_training_data_hdf5(data_path: Path) -> Dict[str, Any]:
    """
    Load training data from HDF5 format.

    Args:
        data_path: Path to directory containing HDF5 file

    Returns:
        Dictionary containing loaded data and metadata

    Raises:
        FileNotFoundError: If HDF5 file doesn't exist
    """
    hdf5_path = data_path / "tiw_sst_training_data.h5"

    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    with h5py.File(hdf5_path, "r") as f:
        # Load arrays as PyTorch tensors
        input_sequences = torch.from_numpy(f["input_sequences"][:]).float()
        output_sequences = torch.from_numpy(f["output_sequences"][:]).float()

        # Load sequence metadata from HDF5 datasets
        sequence_metadata = None
        if "sequence_ids" in f and "start_dates" in f and "end_dates" in f:
            sequence_metadata = pd.DataFrame(
                {
                    "sequence_id": f["sequence_ids"][:],
                    "start_date": [s.decode() for s in f["start_dates"][:]],
                    "end_date": [s.decode() for s in f["end_dates"][:]],
                }
            )

        # Load metadata from attributes
        input_region_bounds = json.loads(f.attrs["input_region_bounds"])
        output_region_bounds = json.loads(f.attrs["output_region_bounds"])
        sequence_config = json.loads(f.attrs["sequence_config"])

        # Load additional info
        input_shape = tuple(f.attrs["input_shape"])
        output_shape = tuple(f.attrs["output_shape"])
        total_sequences = f.attrs["total_sequences"]

        # Load optional attributes (for newer format versions)
        created_date = f.attrs.get("created_date", "Unknown")
        format_version = f.attrs.get("format_version", "1.0")

    return {
        "input_sequences": input_sequences,
        "output_sequences": output_sequences,
        "sequence_metadata": sequence_metadata,
        "input_region_bounds": input_region_bounds,
        "output_region_bounds": output_region_bounds,
        "sequence_config": sequence_config,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "total_sequences": total_sequences,
        "created_date": created_date,
        "format_version": format_version,
    }


# Keep the old function for backward compatibility
def save_pytorch_training_data(
    input_sequences: np.ndarray,
    output_sequences: np.ndarray,
    sequence_dates: pd.DataFrame,
    input_region_bounds: Dict[str, float],
    output_region_bounds: Dict[str, float],
    sequence_config: Dict[str, int],
    save_path: Path,
) -> None:
    """
    Save training data in PyTorch format (deprecated - use save_training_data_hdf5).

    This function is kept for backward compatibility. Use save_training_data_hdf5
    for better performance and compression.
    """
    warnings.warn(
        "save_pytorch_training_data is deprecated. Use save_training_data_hdf5 for "
        "better performance and compression.",
        DeprecationWarning,
        stacklevel=2,
    )

    save_path.mkdir(exist_ok=True)

    # Convert to PyTorch tensors
    input_tensor = torch.from_numpy(input_sequences).float()
    output_tensor = torch.from_numpy(output_sequences).float()

    # Save data dictionary
    training_data = {
        "input_sequences": input_tensor,
        "output_sequences": output_tensor,
        "sequence_metadata": sequence_dates,
        "input_region_bounds": input_region_bounds,
        "output_region_bounds": output_region_bounds,
        "sequence_config": sequence_config,
    }

    # Save as PyTorch file
    torch.save(training_data, save_path / "tiw_sst_training_data.pt")

    # Note: No longer creating separate CSV - use HDF5 format instead


def preprocess_tiw_data(
    raw_data_path: Path,
    processed_data_path: Path,
    input_netcdf_filename: str,
    input_region_bounds: Dict[str, float],
    output_region_bounds: Dict[str, float],
    input_sequence_length_days: int = 14,
    output_sequence_length_days: int = 5,
    stride_days: int = 3,
) -> Dict[str, Any]:
    """
    Complete preprocessing pipeline for TIW SST data.

    Args:
        raw_data_path: Path to raw data directory
        processed_data_path: Path to processed data directory
        input_netcdf_filename: Name of input NetCDF file
        input_region_bounds: Dictionary with input region lat/lon bounds
        output_region_bounds: Dictionary with output region lat/lon bounds
        input_sequence_length_days: Number of days for input sequences
        output_sequence_length_days: Number of days for output sequences
        stride_days: Stride between sequences in days

    Returns:
        Dictionary with preprocessing results and statistics
    """
    # Load data
    netcdf_path = raw_data_path / input_netcdf_filename
    theta_data = load_theta_data(netcdf_path)

    # Crop regions
    input_region_data = crop_region(theta_data, **input_region_bounds)
    output_region_data = crop_region(theta_data, **output_region_bounds)

    # Compute anomalies
    input_sst_anomalies = compute_sst_anomalies(input_region_data)
    output_sst_anomalies = compute_sst_anomalies(output_region_data)

    # Generate sequences
    input_sequences, output_sequences, sequence_dates = generate_training_sequences(
        input_sst_anomalies,
        output_sst_anomalies,
        input_sequence_length_days,
        output_sequence_length_days,
        stride_days,
    )

    # Save data
    sequence_config = {
        "input_length_days": input_sequence_length_days,
        "output_length_days": output_sequence_length_days,
        "stride_days": stride_days,
    }

    save_training_data_hdf5(
        input_sequences,
        output_sequences,
        sequence_dates,
        input_region_bounds,
        output_region_bounds,
        sequence_config,
        processed_data_path,
    )

    # Compute statistics
    results = {
        "total_sequences": len(sequence_dates),
        "input_shape": input_sequences.shape,
        "output_shape": output_sequences.shape,
        "input_anomaly_stats": {
            "mean": float(np.mean(input_sequences)),
            "std": float(np.std(input_sequences)),
            "min": float(np.min(input_sequences)),
            "max": float(np.max(input_sequences)),
        },
        "output_anomaly_stats": {
            "mean": float(np.mean(output_sequences)),
            "std": float(np.std(output_sequences)),
            "min": float(np.min(output_sequences)),
            "max": float(np.max(output_sequences)),
        },
        "nan_values": {
            "input": int(np.isnan(input_sequences).sum()),
            "output": int(np.isnan(output_sequences).sum()),
        },
    }

    return results


if __name__ == "__main__":
    # Default configuration for TPOSE6 data
    raw_data_path = Path("data/raw")
    processed_data_path = Path("data/processed")
    input_netcdf_filename = "TPOSE6_Daily_2012_surface.nc"

    # Regional bounds adjusted for TPOSE6 data coordinates
    input_region_bounds = {
        "lat_min": -10.0,
        "lat_max": 10.0,
        "lon_min": 210.0,
        "lon_max": 250.0,
    }

    output_region_bounds = {
        "lat_min": -3.0,
        "lat_max": 5.0,
        "lon_min": 215.0,
        "lon_max": 225.0,
    }

    # Run preprocessing
    results = preprocess_tiw_data(
        raw_data_path,
        processed_data_path,
        input_netcdf_filename,
        input_region_bounds,
        output_region_bounds,
    )

    print("Preprocessing completed successfully!")
    print(f"Generated {results['total_sequences']} training sequences")
    print(f"Input shape: {results['input_shape']}")
    print(f"Output shape: {results['output_shape']}")

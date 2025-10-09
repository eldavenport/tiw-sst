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


def extract_sequence_dates(
    sequence_metadata: pd.DataFrame,
    input_length: int,
    output_length: int,
) -> pd.DataFrame:
    """
    Extract full date sequences for each training sample.

    Args:
        sequence_metadata: DataFrame with sequence start/end dates (actual datetime)
        input_length: Number of input time steps
        output_length: Number of output time steps

    Returns:
        DataFrame with sequence_id and list of dates for each sequence
    """
    sequence_dates = []

    for _, row in sequence_metadata.iterrows():
        # Convert string datetime to pandas Timestamp
        if isinstance(row["start_date"], str):
            start_date = pd.Timestamp(row["start_date"])
        else:
            start_date = pd.Timestamp(row["start_date"])

        # Generate all dates in the sequence (input + output)
        total_days = input_length + output_length
        dates = [start_date + pd.Timedelta(days=i) for i in range(total_days)]

        sequence_dates.append(
            {"sequence_id": row["sequence_id"], "dates": dates, "start_date": start_date}
        )

    return pd.DataFrame(sequence_dates)


def compute_day_of_year_encoding(dates_list: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute day-of-year and seasonal angle encoding for a list of dates.

    Args:
        dates_list: List of pandas Timestamps

    Returns:
        Tuple of (day_of_year, angle_radians) arrays
    """
    day_of_year = np.array([date.dayofyear for date in dates_list])
    # Convert to radians: theta = 2π * (day_of_year) / 365
    angle_radians = 2 * np.pi * day_of_year / 365.0

    return day_of_year, angle_radians


def create_doy_channels(
    angles: np.ndarray, spatial_shape: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sin/cos seasonal encoding channels with same spatial dimensions as SST.

    Args:
        angles: Array of angles in radians for each time step
        spatial_shape: (height, width) of spatial dimensions

    Returns:
        Tuple of (sin_channel, cos_channel) arrays with shape (time, height, width)
    """
    n_time = len(angles)
    height, width = spatial_shape

    # Create channels: same value at all spatial points for each time step
    sin_channel = np.zeros((n_time, height, width), dtype=np.float32)
    cos_channel = np.zeros((n_time, height, width), dtype=np.float32)

    for t in range(n_time):
        sin_channel[t, :, :] = np.sin(angles[t])
        cos_channel[t, :, :] = np.cos(angles[t])

    return sin_channel, cos_channel


def add_seasonal_encoding_to_sequences(
    input_sequences: torch.Tensor,
    output_sequences: torch.Tensor,
    sequence_metadata: pd.DataFrame,
    input_length: int,
    output_length: int,
) -> Dict[str, Any]:
    """
    Add seasonal encoding channels to existing SST sequences.

    Args:
        input_sequences: Input SST sequences (samples, time, lat, lon)
        output_sequences: Output SST sequences (samples, time, lat, lon)
        sequence_metadata: DataFrame with sequence date information (actual dates)
        input_length: Number of input time steps
        output_length: Number of output time steps

    Returns:
        Dictionary containing enhanced sequences with seasonal encoding
    """
    # Extract dates for all sequences
    sequence_dates_df = extract_sequence_dates(
        sequence_metadata, input_length, output_length
    )

    # Get spatial dimensions from input sequences
    n_samples, _, input_height, input_width = input_sequences.shape
    _, _, output_height, output_width = output_sequences.shape

    # Initialize seasonal encoding arrays
    input_sin_doy = np.zeros((n_samples, input_length, input_height, input_width), dtype=np.float32)
    input_cos_doy = np.zeros((n_samples, input_length, input_height, input_width), dtype=np.float32)
    output_sin_doy = np.zeros((n_samples, output_length, output_height, output_width), dtype=np.float32)
    output_cos_doy = np.zeros((n_samples, output_length, output_height, output_width), dtype=np.float32)

    # Store metadata for each sequence
    enhanced_metadata = []

    for idx, row in sequence_dates_df.iterrows():
        sequence_id = row["sequence_id"]
        dates = row["dates"]

        # Compute day-of-year and angles for full sequence
        day_of_year, angles = compute_day_of_year_encoding(dates)

        # Split into input and output portions
        input_angles = angles[:input_length]
        output_angles = angles[input_length:]
        input_doy = day_of_year[:input_length]
        output_doy = day_of_year[input_length:]

        # Create seasonal channels for input sequences
        input_sin_seq, input_cos_seq = create_doy_channels(
            input_angles, (input_height, input_width)
        )
        input_sin_doy[idx] = input_sin_seq
        input_cos_doy[idx] = input_cos_seq

        # Create seasonal channels for output sequences
        output_sin_seq, output_cos_seq = create_doy_channels(
            output_angles, (output_height, output_width)
        )
        output_sin_doy[idx] = output_sin_seq
        output_cos_doy[idx] = output_cos_seq

        # Store enhanced metadata
        enhanced_metadata.append(
            {
                "sequence_id": sequence_id,
                "start_date": row["start_date"].isoformat(),
                "dates": [d.isoformat() for d in dates],
                "input_day_of_year": input_doy.tolist(),
                "output_day_of_year": output_doy.tolist(),
                "input_angles_rad": input_angles.tolist(),
                "output_angles_rad": output_angles.tolist(),
            }
        )

    return {
        "input_sst": input_sequences.numpy(),
        "output_sst": output_sequences.numpy(),
        "input_sin_doy": input_sin_doy,
        "input_cos_doy": input_cos_doy,
        "output_sin_doy": output_sin_doy,
        "output_cos_doy": output_cos_doy,
        "enhanced_metadata": pd.DataFrame(enhanced_metadata),
    }


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



def generate_training_sequences(
    input_sst: xr.DataArray,
    output_sst: xr.DataArray,
    input_length: int,
    output_length: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Generate training sequences with anomalies computed relative to day 14.

    This function creates sequences where anomalies are computed relative to 
    the last day (day 14) of each input sequence, providing a reference point
    for forecasting changes in SST patterns.

    Args:
        input_sst: Input region SST data (time, lat, lon) - raw values
        output_sst: Output region SST data (time, lat, lon) - raw values  
        input_length: Number of days for input sequence (should be 14)
        output_length: Number of days for output sequence
        stride: Stride between sequences in days

    Returns:
        Tuple of (input_sequences, output_sequences, sequence_dates)
        - input_sequences: numpy array (samples, time, lat, lon) - anomalies relative to day 14
        - output_sequences: numpy array (samples, time, lat, lon) - anomalies relative to day 14
        - sequence_dates: DataFrame with sequence metadata

    Raises:
        ValueError: If sequences would be too short given the data length
    """
    total_time_steps = len(input_sst.time)
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

        # Extract raw SST sequences
        input_seq_raw = input_sst.isel(time=slice(start_idx, input_end_idx))
        output_seq_raw = output_sst.isel(time=slice(input_end_idx, output_end_idx))
        
        # Get day 14 (last day of input) as reference for anomalies
        # Day 14 is at index input_length-1 (e.g., index 13 for 14-day sequence)
        reference_day = input_seq_raw.isel(time=input_length-1)
        
        # Compute anomalies relative to day 14
        input_anomalies = input_seq_raw - reference_day
        output_anomalies = output_seq_raw - reference_day

        # Store sequences as numpy arrays
        input_sequences.append(input_anomalies.values)
        output_sequences.append(output_anomalies.values)

        # Store date information
        start_date = input_sst.time[start_idx].values
        end_date = output_sst.time[output_end_idx - 1].values
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


def save_sequences_with_date_encoding_hdf5(
    enhanced_data: Dict[str, Any],
    input_region_bounds: Dict[str, float],
    output_region_bounds: Dict[str, float],
    sequence_config: Dict[str, int],
    save_path: Path,
    filename: str = "tiw_sst_sequences.h5",
) -> None:
    """
    Save sequences with date encoding to HDF5 format.

    Args:
        enhanced_data: Dictionary containing sequences and seasonal encoding
        input_region_bounds: Dictionary with input region bounds
        output_region_bounds: Dictionary with output region bounds
        sequence_config: Dictionary with sequence configuration
        save_path: Path to save processed data
        filename: Name of the HDF5 file
    """
    save_path.mkdir(exist_ok=True)
    hdf5_path = save_path / filename

    with h5py.File(hdf5_path, "w") as f:
        # Save SST sequences
        f.create_dataset(
            "input_sst",
            data=enhanced_data["input_sst"],
            compression="gzip",
            compression_opts=6,
            chunks=True,
        )
        f.create_dataset(
            "output_sst",
            data=enhanced_data["output_sst"],
            compression="gzip",
            compression_opts=6,
            chunks=True,
        )

        # Save seasonal encoding channels
        f.create_dataset(
            "input_sin_doy",
            data=enhanced_data["input_sin_doy"],
            compression="gzip",
            compression_opts=6,
            chunks=True,
        )
        f.create_dataset(
            "input_cos_doy",
            data=enhanced_data["input_cos_doy"],
            compression="gzip",
            compression_opts=6,
            chunks=True,
        )
        f.create_dataset(
            "output_sin_doy",
            data=enhanced_data["output_sin_doy"],
            compression="gzip",
            compression_opts=6,
            chunks=True,
        )
        f.create_dataset(
            "output_cos_doy",
            data=enhanced_data["output_cos_doy"],
            compression="gzip",
            compression_opts=6,
            chunks=True,
        )

        # Save enhanced metadata
        metadata_df = enhanced_data["enhanced_metadata"]
        f.create_dataset("sequence_ids", data=metadata_df["sequence_id"].values)

        # Store complex metadata as JSON strings
        start_dates = metadata_df["start_date"].values.astype("S32")
        f.create_dataset("start_dates", data=start_dates, compression="gzip")

        # Store dates, day_of_year, and angles as JSON for flexibility
        dates_json = [json.dumps(dates) for dates in metadata_df["dates"]]
        input_doy_json = [json.dumps(doy) for doy in metadata_df["input_day_of_year"]]
        output_doy_json = [json.dumps(doy) for doy in metadata_df["output_day_of_year"]]
        input_angles_json = [json.dumps(angles) for angles in metadata_df["input_angles_rad"]]
        output_angles_json = [json.dumps(angles) for angles in metadata_df["output_angles_rad"]]

        f.create_dataset(
            "sequence_dates", data=np.array(dates_json, dtype="S2000"), compression="gzip"
        )
        f.create_dataset(
            "input_day_of_year", data=np.array(input_doy_json, dtype="S500"), compression="gzip"
        )
        f.create_dataset(
            "output_day_of_year", data=np.array(output_doy_json, dtype="S300"), compression="gzip"
        )
        f.create_dataset(
            "input_angles_rad", data=np.array(input_angles_json, dtype="S1000"), compression="gzip"
        )
        f.create_dataset(
            "output_angles_rad", data=np.array(output_angles_json, dtype="S800"), compression="gzip"
        )

        # Save configuration metadata as attributes
        f.attrs["input_region_bounds"] = json.dumps(input_region_bounds)
        f.attrs["output_region_bounds"] = json.dumps(output_region_bounds)
        f.attrs["sequence_config"] = json.dumps(sequence_config)

        # Save array shapes and data types for easy access
        f.attrs["input_sst_shape"] = enhanced_data["input_sst"].shape
        f.attrs["output_sst_shape"] = enhanced_data["output_sst"].shape
        f.attrs["input_sin_doy_shape"] = enhanced_data["input_sin_doy"].shape
        f.attrs["input_cos_doy_shape"] = enhanced_data["input_cos_doy"].shape
        f.attrs["output_sin_doy_shape"] = enhanced_data["output_sin_doy"].shape
        f.attrs["output_cos_doy_shape"] = enhanced_data["output_cos_doy"].shape
        f.attrs["total_sequences"] = len(metadata_df)
        f.attrs["created_date"] = pd.Timestamp.now().isoformat()
        f.attrs["format_version"] = "3.0"
        f.attrs["includes_date_encoding"] = True
        f.attrs["description"] = "TIW SST sequences with seasonal day-of-year encoding"


def save_training_data_hdf5(
    input_sequences: np.ndarray,
    output_sequences: np.ndarray,
    sequence_dates: pd.DataFrame,
    input_region_bounds: Dict[str, float],
    output_region_bounds: Dict[str, float],
    sequence_config: Dict[str, int],
    save_path: Path,
    filename: str = "tiw_sst_sequences.h5",
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
    hdf5_path = save_path / filename

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
        f.attrs["format_version"] = "3.0"  # Track format version
        f.attrs["includes_date_encoding"] = False


def load_training_data_hdf5(data_path: Path, filename: str = "tiw_sst_sequences.h5") -> Dict[str, Any]:
    """
    Load training data from HDF5 format.

    Args:
        data_path: Path to directory containing HDF5 file
        filename: Name of the HDF5 file

    Returns:
        Dictionary containing loaded data and metadata

    Raises:
        FileNotFoundError: If HDF5 file doesn't exist
    """
    hdf5_path = Path(data_path) / filename

    if not hdf5_path.exists():
        # Try old filename for backward compatibility
        old_path = Path(data_path) / "tiw_sst_training_data.h5"
        if old_path.exists():
            hdf5_path = old_path
        else:
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    with h5py.File(hdf5_path, "r") as f:
        # Check if this includes date encoding
        includes_date_encoding = f.attrs.get("includes_date_encoding", False)
        
        # Load SST sequences (check for both old and new field names)
        if "input_sst" in f:
            input_sst = torch.from_numpy(f["input_sst"][:]).float()
            output_sst = torch.from_numpy(f["output_sst"][:]).float()
        else:  # Backward compatibility with old format
            input_sst = torch.from_numpy(f["input_sequences"][:]).float()
            output_sst = torch.from_numpy(f["output_sequences"][:]).float()
        
        # Load date encoding if present
        input_sin_doy = None
        input_cos_doy = None
        output_sin_doy = None
        output_cos_doy = None
        enhanced_metadata = None
        
        if includes_date_encoding and "input_sin_doy" in f:
            input_sin_doy = torch.from_numpy(f["input_sin_doy"][:]).float()
            input_cos_doy = torch.from_numpy(f["input_cos_doy"][:]).float()
            output_sin_doy = torch.from_numpy(f["output_sin_doy"][:]).float()
            output_cos_doy = torch.from_numpy(f["output_cos_doy"][:]).float()
            
            # Load enhanced metadata
            sequence_ids = f["sequence_ids"][:]
            start_dates = [s.decode() for s in f["start_dates"][:]]
            
            # Load JSON-encoded arrays
            dates = [json.loads(s.decode()) for s in f["sequence_dates"][:]]
            input_doy = [json.loads(s.decode()) for s in f["input_day_of_year"][:]]
            output_doy = [json.loads(s.decode()) for s in f["output_day_of_year"][:]]
            input_angles = [json.loads(s.decode()) for s in f["input_angles_rad"][:]]
            output_angles = [json.loads(s.decode()) for s in f["output_angles_rad"][:]]

            enhanced_metadata = pd.DataFrame({
                "sequence_id": sequence_ids,
                "start_date": start_dates,
                "dates": dates,
                "input_day_of_year": input_doy,
                "output_day_of_year": output_doy,
                "input_angles_rad": input_angles,
                "output_angles_rad": output_angles,
            })

        # Load basic sequence metadata (for backward compatibility)
        sequence_metadata = None
        if "sequence_ids" in f and "start_dates" in f:
            if "end_dates" in f:  # Old format
                sequence_metadata = pd.DataFrame({
                    "sequence_id": f["sequence_ids"][:],
                    "start_date": [s.decode() for s in f["start_dates"][:]],
                    "end_date": [s.decode() for s in f["end_dates"][:]],
                })
            elif enhanced_metadata is not None:  # Use enhanced metadata
                sequence_metadata = enhanced_metadata[["sequence_id", "start_date"]].copy()

        # Load metadata from attributes
        input_region_bounds = json.loads(f.attrs["input_region_bounds"])
        output_region_bounds = json.loads(f.attrs["output_region_bounds"])
        sequence_config = json.loads(f.attrs["sequence_config"])

        # Load additional info
        input_shape = tuple(f.attrs.get("input_sst_shape", f.attrs.get("input_shape", input_sst.shape)))
        output_shape = tuple(f.attrs.get("output_sst_shape", f.attrs.get("output_shape", output_sst.shape)))
        total_sequences = f.attrs["total_sequences"]

        # Load optional attributes (for newer format versions)
        created_date = f.attrs.get("created_date", "Unknown")
        format_version = f.attrs.get("format_version", "1.0")

    result = {
        "input_sst": input_sst,
        "output_sst": output_sst,
        "sequence_metadata": sequence_metadata,
        "input_region_bounds": input_region_bounds,
        "output_region_bounds": output_region_bounds,
        "sequence_config": sequence_config,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "total_sequences": total_sequences,
        "created_date": created_date,
        "format_version": format_version,
        "includes_date_encoding": includes_date_encoding,
    }
    
    # Add date encoding data if present
    if includes_date_encoding:
        result.update({
            "input_sin_doy": input_sin_doy,
            "input_cos_doy": input_cos_doy,
            "output_sin_doy": output_sin_doy,
            "output_cos_doy": output_cos_doy,
            "enhanced_metadata": enhanced_metadata,
        })
    
    # Backward compatibility: provide old field names
    result["input_sequences"] = input_sst
    result["output_sequences"] = output_sst
    
    return result




def preprocess_tiw_data(
    raw_data_path: Path,
    processed_data_path: Path,
    input_netcdf_filename: str = "TPOSE6_Daily_2012_SST.nc",
    input_region_bounds: Dict[str, float] = None,
    output_region_bounds: Dict[str, float] = None,
    input_sequence_length_days: int = 14,
    output_sequence_length_days: int = 5,
    stride_days: int = 3,
    include_date_encoding: bool = True,
) -> Dict[str, Any]:
    """
    Preprocessing pipeline for TIW SST data with optional date encoding.

    This function takes a NetCDF file, subsets the domain to specified regions,
    creates 19-day sequences (14 input + 5 output) with anomalies computed 
    relative to day 14 of each sequence, optionally encodes dates into sin/cos 
    seasonal channels, and stores everything in HDF5 format.

    Args:
        raw_data_path: Path to raw data directory
        processed_data_path: Path to processed data directory
        input_netcdf_filename: Name of input NetCDF file (default: TPOSE6_Daily_2012_SST.nc)
        input_region_bounds: Dictionary with input region lat/lon bounds (default: TIW region)
        output_region_bounds: Dictionary with output region lat/lon bounds (default: TIW core)
        input_sequence_length_days: Number of days for input sequences
        output_sequence_length_days: Number of days for output sequences
        stride_days: Stride between sequences in days
        include_date_encoding: Whether to add sin/cos date encoding channels

    Returns:
        Dictionary with preprocessing results and statistics
    """
    # Set default region bounds if not provided
    if input_region_bounds is None:
        input_region_bounds = {
            "lat_min": -10.0,
            "lat_max": 10.0,
            "lon_min": 210.0,
            "lon_max": 250.0,
        }
    
    if output_region_bounds is None:
        output_region_bounds = {
            "lat_min": -3.0,
            "lat_max": 5.0,
            "lon_min": 215.0,
            "lon_max": 225.0,
        }
    # Load data
    netcdf_path = raw_data_path / input_netcdf_filename
    theta_data = load_theta_data(netcdf_path)

    # Crop regions
    input_region_data = crop_region(theta_data, **input_region_bounds)
    output_region_data = crop_region(theta_data, **output_region_bounds)

    # Generate sequences (anomalies computed relative to day 14 within each sequence)
    input_sequences, output_sequences, sequence_dates = generate_training_sequences(
        input_region_data,
        output_region_data,
        input_sequence_length_days,
        output_sequence_length_days,
        stride_days,
    )

    # Prepare sequence config
    sequence_config = {
        "input_length_days": input_sequence_length_days,
        "output_length_days": output_sequence_length_days,
        "stride_days": stride_days,
        "includes_date_encoding": include_date_encoding,
    }

    # Add date encoding if requested
    if include_date_encoding:
        
        # Convert to PyTorch tensors for date encoding
        input_tensor = torch.from_numpy(input_sequences).float()
        output_tensor = torch.from_numpy(output_sequences).float()
        
        # Add seasonal encoding
        enhanced_data = add_seasonal_encoding_to_sequences(
            input_tensor, output_tensor, sequence_dates, 
            input_sequence_length_days, output_sequence_length_days
        )
        
        # Save with date encoding
        save_sequences_with_date_encoding_hdf5(
            enhanced_data,
            input_region_bounds,
            output_region_bounds,
            sequence_config,
            processed_data_path,
        )
        
        # Update results with date encoding info
        input_sequences = enhanced_data["input_sst"]
        output_sequences = enhanced_data["output_sst"]
        
    else:
        # Save basic sequences without date encoding
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
        "includes_date_encoding": include_date_encoding,
    }

    return results


if __name__ == "__main__":
    # Default configuration for TPOSE6 data with actual dates and date encoding
    raw_data_path = Path("data/raw")
    processed_data_path = Path("data/processed")

    # Run preprocessing with date encoding enabled by default
    results = preprocess_tiw_data(
        raw_data_path, 
        processed_data_path, 
        include_date_encoding=True
    )

    print(f"Generated {results['total_sequences']} training sequences")
    print(f"Input shape: {results['input_shape']}")
    print(f"Output shape: {results['output_shape']}")
    print(f"Includes date encoding: {results['includes_date_encoding']}")

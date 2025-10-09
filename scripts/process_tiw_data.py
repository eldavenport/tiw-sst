#!/usr/bin/env python3
"""
Streamlined TIW SST Processing Pipeline

This script provides a simple one-step pipeline that:
1. Takes NetCDF file with SST data
2. Subsets the domain to TIW regions  
3. Creates 19-day sequences (14 input + 5 output)
4. Encodes dates into sin/cos seasonal channels
5. Stores everything in a single HDF5 file

Run from project root as: python -m scripts.process_tiw_data
"""

from pathlib import Path
from data.preprocessing import preprocess_tiw_data


def main():
    """Run the complete TIW SST preprocessing pipeline."""
    print("🌊 TIW SST Processing Pipeline")
    print("=" * 40)
    
    # Set up paths
    raw_data_path = Path("data/raw")
    processed_data_path = Path("data/processed")
    
    # Clean start - remove old processed files
    print("🧹 Cleaning existing processed data...")
    for file in processed_data_path.glob("*.h5"):
        file.unlink()
        print(f"   Removed: {file.name}")
    
    # Run complete pipeline with date encoding
    print("\n📊 Processing NetCDF → Sequences with Date Encoding")
    print("-" * 50)
    
    results = preprocess_tiw_data(
        raw_data_path=raw_data_path,
        processed_data_path=processed_data_path,
        include_date_encoding=True  # This is the key parameter
    )
    
    print("✓ Processing completed successfully!")
    print(f"  Generated: {results['total_sequences']} training sequences")
    print(f"  Input shape: {results['input_shape']}")
    print(f"  Output shape: {results['output_shape']}")
    print(f"  Includes date encoding: {results.get('includes_date_encoding', False)}")
    
    # Show file information
    output_file = processed_data_path / "tiw_sst_sequences.h5"
    if output_file.exists():
        file_size = output_file.stat().st_size / (1024**3)  # GB
        print(f"  Output file: {output_file.name} ({file_size:.2f} GB)")
    
    # Show seasonal encoding verification
    if results.get('includes_date_encoding'):
        print(f"\n📅 Date Encoding Verification:")
        print(f"  Input channels: SST + sin_doy + cos_doy")
        print(f"  Output channels: SST + sin_doy + cos_doy")
        print(f"  Seasonal encoding ranges:")
        print(f"    sin(day_of_year): [-1.000, 1.000]")
        print(f"    cos(day_of_year): [-1.000, 1.000]")
    
    print(f"\n🎉 Ready for ConvLSTM training!")
    print(f"   Load data with: data.preprocessing.load_training_data_hdf5()")


if __name__ == "__main__":
    main()
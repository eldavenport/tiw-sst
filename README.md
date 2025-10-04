# TIW SST Forecasting

[![Tests](https://github.com/eldavenport/tiw-sst/actions/workflows/tests.yml/badge.svg)](https://github.com/eldavenport/tiw-sst/actions/workflows/tests.yml)
[![Documentation](https://github.com/eldavenport/tiw-sst/actions/workflows/docs.yml/badge.svg)](https://github.com/eldavenport/tiw-sst/actions/workflows/docs.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Predicting SST in a small region using 2 weeks of daily input SST data. (aimed at predicting TIW SST)

1. **Input**: 14 days of daily SST anomaly fields from a large Pacific region (10°N-10°S, 150°W-110°W)
2. **Output**: 5-day forecast of SST anomalies in the TIW-active region (5°N-3°S, 145°W-135°W)  

Visit the [online documentation](https://eldavenport.github.io/tiw-sst/).

## Data Requirements

The current expected raw data format is: 

- **Format**: NetCDF files from MITgcm simulations
- **Variable**: `THETA` (potential temperature) with coordinates `(time, Z, YC, XC)`
- **Resolution**: Daily timesteps

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
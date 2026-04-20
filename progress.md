# EE 244 Final Project - Progress & Matrix

This document tracks the implementation progress, technical challenges encountered, and the reasoning behind each major change in the project. It also serves as the master tracking matrix for weekly deliverables.

## Project Tracking Matrix

| Week | Phase | Goal / Deliverable | Status |
| :--- | :--- | :--- | :--- |
| **Weeks 1-2** | **Data Preparation** | Preprocess THÖR-MAGNI dataset. Extract synchronized triplets at 10Hz. | In Progress 🟢 |
| **Week 3** | **Initial Training** | Train and validate on THÖR-MAGNI. Hold out 20% sequences. Establish MAE baseline against Kalman filter. | Not Started ⚪ |
| **Week 4** | **Domain Adaptation** | Collect 30-min domain adaptation set in deployment environment. Generate pseudo-labels using Kalman output. | Not Started ⚪ |
| **Weeks 5-6** | **Fine-Tuning** | Fine-tune model on collected data (learning rate ≤ 1e-4, freezing early layers) to bridge sensor gap. | Not Started ⚪ |
| **Week 7** | **Demo Prep** | Focus on A/B comparison (DWA with/without velocity estimates). Visualize in RViz2 with velocity arrows. | Not Started ⚪ |

---

## Progress Log

### [2026-04-19 09:00 AM] Initial Setup & Dataset Exploration
**Scripts Renamed/Created:** 
- `01_fetch_zenodo_dataset.py` (formerly `download_dataset.py`) created to handle fetching the main dataset (THÖR-MAGNI) via the Zenodo API.
- `00_verify_zenodo_size.py` (formerly `download_test.py`) created to test the Zenodo API endpoint and determine the total size of the dataset.

**Reasoning:** 
We needed to verify that the dataset could be automatically downloaded to the robot/workstation, but because it contains gigabytes of data, a structural check was essential to understand the memory footprint first before downloading the 21GB payload.

### [2026-04-19 02:00 PM] Dataset Downloaded & Initial Extraction
**Scripts Renamed/Created:** 
- Dataset fully downloaded to the local workspace.
- `02_clean_and_extract_trajectories.py` (formerly `extract_data.py`) written to iteratively parse the raw `CSV_Scenarios` files and extract position coordinates for both the robot and pedestrians.

**Reasoning:** 
We needed a pipeline to translate the massive THÖR-MAGNI dataset, containing high-frequency (100Hz) motion capture points, into a structured and downsampled (10Hz) set of trajectory DataFrames that a velocity regression model could consume.

### [2026-04-19 08:30 PM] Data Quality Sanity Check & Velocity Spikes Found
**Scripts Renamed/Created:** 
- `00_inspect_csv_headers.py` (formerly `view_dataset_properties.py`)
- `00_inspect_zip_contents.py` (formerly `test_dataset_size.py`)
- `04_plot_trajectory_stats.py` (formerly `test_csv_file.py`) 

**Reasoning:** 
Sanity checks are crucial when working with raw sensor or motion-capture data. When analyzing the extracted features using our statistical plotting scripts, the output revealed impossible human speeds—specifically, a minimum velocity of -208 m/s and a maximum velocity of 215 m/s. This indicated that the raw data contained significant noise, occlusion gaps, or missing frames that resulted in violent geographical "jumps" when taking the position derivative.

### [2026-04-19 10:30 PM] Dataset Sanitization & Noise Filtering
**Scripts Modified:** 
- `02_clean_and_extract_trajectories.py` refactored entirely to implement the `clean_and_compute_velocity()` function.
- Added thresholds for physical plausibility (`MAX_SPEED_MS = 3.0` and `MAX_POSITION_JUMP_M = 0.5`).

**Reasoning:** 
To train an accurate velocity estimation network, the ground-truth velocity labels must be entirely noise-free and physically plausible. By monitoring the distance vector between subsequent timestamps, the script now catches frames where the motion capture temporarily loses track of a pedestrian and then regains them. It masks out those unnatural position jumps (and excessive gradient derivatives) with NaNs, effectively eliminating the ±200 m/s noise spikes from our final processed dataset.

### [2026-04-19 11:30 PM] Feature Windowing for ML Input
**Scripts Renamed/Created:** 
- `03_build_training_windows.py` (formerly `build_windows.py`) created.

**Reasoning:** 
With clean data extracted, the MLP regression network requires sequences of past data to predict the final velocity. We implemented a sliding window approach with parameters `T=10` (1 second history at 10Hz) and `stride=1` frame step. This script successfully partitions our cleaned pedestrian trajectories into structured `X` (features matrix, shape `T * 4`) and `y` (target `[vx, vy]`) matrices for our models, rigorously separated into Train, Validation, and Test sequences without cross-contamination.

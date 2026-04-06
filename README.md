# Fiber Photometry Analysis

Analysis scripts for fiber photometry experiments studying A2A receptors in the dorsolateral striatum (DLS).

## Repository Structure

```
.
├── fiber_photometry_analysis.py    # Main Python analysis script
├── archive/                        # Original Jupyter notebooks
│   ├── 03_19_2025.ipynb
│   ├── main_1.ipynb
│   └── main_backup.ipynb
├── A2A cocain sensitization.xlsx  # Data file
├── SKF data.pptx                  # Presentation
└── data_analysis_description.pdf  # Documentation
```

## Overview

This repository contains analysis code for calcium imaging data from fiber photometry experiments comparing:
- **Control vs KO groups** (A2A receptor knockout)
- **Different treatments** (saline, cocaine, halo)
- **Multiple timepoints** (mid/end analysis, start/end analysis)

## Usage

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scipy tdt
```

### Running the Analysis

```bash
python fiber_photometry_analysis.py
```

This will:
1. Load data from `Notes/` directory
2. Run 900s trace analysis
3. Generate mid/end and start/end plots
4. Create heatmap visualizations
5. Save all outputs to `plots/` directory

## Data Structure

### Input Files (in `Notes/` directory)
- `900s_whole_trace.csv` - Start/end timepoints for 900s recordings
- `mid_end_transposed.csv` - Mid/end timepoints for groups 1-4
- `start_end_transposed.csv` - Start/end timepoints for groups 1-7

### TDT Data Folders
Folder naming convention: `JZ_exp-YYYYMMDD-HHMMSS-[genotype]-[condition]`
- Contains `Notes.txt` and `.avi` video files
- Photometry streams: `_465p` (signal) and `_405p` (control)

## Experimental Groups

| Group | Description |
|-------|-------------|
| 1 | D1-Cre Control (Day 5) |
| 2 | D1-Cre Knockout (Day 5) |
| 3 | A2A Control (Day 5) |
| 4 | A2A Knockout (Day 5) |

## Analysis Pipeline

The main script performs:
1. **Signal Processing**: Savitzky-Golay smoothing, polynomial fitting for motion correction
2. **Delta F/F Calculation**: Normalized fluorescence signal
3. **AUC Analysis**: Area under curve calculations
4. **Peak Detection**: Using median absolute deviation threshold
5. **Visualization**: SEM plots, heatmaps, and comparative analyses

## Outputs

All plots are saved to the `plots/` directory:
- `midend-mid-*.svg/png` - Mid timepoint comparisons
- `midend-end-*.svg/png` - End timepoint comparisons
- `*_heatmap.png` - Group heatmaps

## Archive

The `archive/` folder contains the original Jupyter notebooks that were converted to the Python script for easier version control and reproducibility.

## Contact

For questions about this analysis, please refer to the data analysis description PDF or contact the repository owner.

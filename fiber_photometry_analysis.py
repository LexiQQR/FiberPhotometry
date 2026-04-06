"""
Fiber Photometry Analysis Script
Converts main_1.ipynb notebook to clean Python script
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter, find_peaks
from scipy import stats
import math
import tdt


# Configuration
DATA_DIR = "Notes"
OUTPUT_DIR = "plots"
FS = None  # Sampling frequency, set during data loading

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def read_folder_data2(folder, span=1000):
    """Read TDT photometry data and return raw control, signal, and sampling frequency."""
    data = tdt.read_block(folder)
    signal = data.streams._465p.data[:]
    signal = savgol_filter(signal, span, 1)
    control = data.streams._405p.data[:]
    control = savgol_filter(control, span, 1)
    return control, signal, data.streams._405p.fs


def read_folder_data(folder, span=1000):
    """Read TDT photometry data and return delta F/F."""
    data = tdt.read_block(folder)
    signal = data.streams._465p.data[:]
    signal = savgol_filter(signal, span, 1)
    control = data.streams._405p.data[:]
    control = savgol_filter(control, span, 1)
    adj_control = controlFit(control, signal)
    deviations = delta_f_over_f(adj_control, signal)
    return deviations


def controlFit(control, signal):
    """Fit polynomial regression between control and signal."""
    p = np.polynomial.polynomial.Polynomial.fit(control, signal, 1).convert().coef
    arr = (p[1] * control) + p[0]
    return arr


def delta_f_over_f(control, signal):
    """Calculate delta F/F (delta F over F) fluorescence signal."""
    return (signal - control) / control


def sem_plot(array, label="", axis=0, color="red", fill_color="red", alpha=0.2):
    """Plot mean +/- SEM with shaded error bands."""
    mean = array.mean(axis=axis)
    sem_plus = mean + stats.sem(array, axis=axis)
    sem_minus = mean - stats.sem(array, axis=axis)

    plt.fill_between(
        np.arange(mean.shape[0]),
        sem_plus,
        sem_minus,
        alpha=alpha,
        color=fill_color,
        linewidth=0,
    )
    return plt.plot(mean, label=label, color=color)


def calc_group_array_midend(group, mid_end_ticks, fs, secs=8):
    """Process mid/end timepoints for a group.

    Returns:
        tuple: (overall_mid, overall_end) arrays
    """
    overall_mid, overall_end = [], []
    for i in mid_end_ticks[mid_end_ticks.Group == group].index:
        folder = mid_end_ticks.at[i, "folder"]
        deviation = read_folder_data(folder)

        mids = mid_end_ticks.loc[
            i, ["mid1", "mid2", "mid3", "end1", "end2", "end3"]
        ].values[:3]
        ends = mid_end_ticks.loc[
            i, ["mid1", "mid2", "mid3", "end1", "end2", "end3"]
        ].values[3:]

        for i in range(3):
            m_start, m_end = int(mids[i] * fs), int(mids[i] * fs) + int(secs * fs)
            overall_mid.append(deviation[m_start:m_end])
            e_start, e_end = int(ends[i] * fs), int(ends[i] * fs) + int(secs * fs)
            overall_end.append(deviation[e_start:e_end])

    return np.array(overall_mid), np.array(overall_end)


def calc_group_array_startend(group, start_end_ticks, fs, secs=10):
    """Process start/end timepoints for a group.

    Returns:
        tuple: (starts, ends) - lists of 5 trials each
    """
    starts, ends = [], []

    deviations = {}
    for i in start_end_ticks[start_end_ticks.Group == group].index:
        folder = start_end_ticks.at[i, "folder"]
        deviations[folder] = [read_folder_data(folder), i]

    for a, b in [
        ["start1", "end1"],
        ["start2", "end2"],
        ["start3", "end3"],
        ["start4", "end4"],
        ["start5", "end5"],
    ]:
        tmp_starts, tmp_ends = [], []
        for folder in deviations:
            deviation, j = deviations[folder]
            s = start_end_ticks.at[j, a]
            e = start_end_ticks.at[j, b]

            if np.isnan(s) or np.isnan(e):
                continue

            s_start, s_end = int(s * fs), int(s * fs) + int(secs * fs)
            e_start, e_end = int(e * fs) - int(secs * fs), int(e * fs)

            tmp_starts.append(deviation[s_start:s_end])
            tmp_ends.append(deviation[e_start:e_end])

        starts.append(np.array(tmp_starts))
        ends.append(np.array(tmp_ends))

    return starts, ends


def analyze_900s_traces(start_end_ticks):
    """Analyze 900s whole trace data.
    Calculates Stds, StdOverMeans, and AUCs for each recording.
    """
    global FS

    Stds = []
    StdOverMeans = []
    AUCs = []

    for i in range(len(start_end_ticks)):
        folder = start_end_ticks.folder[i]
        start = start_end_ticks.loc[start_end_ticks.folder == folder, "start"].values[0]
        end = start_end_ticks.loc[start_end_ticks.folder == folder, "end"].values[0]

        control, signal, fs = read_folder_data2(folder)
        FS = fs  # Store sampling frequency globally

        signal = signal[int(start * fs) : int(end * fs)]
        control = control[int(start * fs) : int(end * fs)]

        coef = (
            np.polynomial.polynomial.Polynomial.fit(control, signal, 1).convert().coef
        )
        adj_control = coef[0] + control * coef[1]
        dff = (signal - adj_control) / adj_control

        # Calculate metrics
        dff_mean = dff.mean()
        dff_shifted = np.array([val - dff_mean for val in dff])
        Stds.append(dff_shifted.std())

        dff_min = min(dff_shifted)
        dff_shifted2 = np.array([val - dff_min for val in dff_shifted])
        StdOverMeans.append(dff_shifted2.std() / dff_shifted2.mean())

        # Calculate AUC
        auc = dff.mean() * len(dff)
        AUCs.append(auc)

    return Stds, StdOverMeans, AUCs


def plot_midend_analysis(mid1, end1, mid2, end2, mid3, end3, mid4, end4, fs):
    """Generate mid/end analysis plots."""

    # D1 Control vs KO - Mid
    plt.figure(figsize=(15, 5))
    sem_plot(mid1 * 100 + 5, "mid_1", color="blue", fill_color="blue")
    sem_plot(mid2 * 100 + 5, "mid_2", color="green", fill_color="green")
    plt.margins(x=0)
    plt.ylim((-2, 15))
    plt.gca().xaxis.set_major_formatter(lambda x, pos: f"{math.ceil(x / fs)}")
    plt.savefig(
        f"{OUTPUT_DIR}/midend-mid-D1-control-ko-day5.svg", format="svg", dpi=300
    )
    plt.savefig(f"{OUTPUT_DIR}/midend-mid-D1-control-ko-day5.png", dpi=300)
    plt.close()

    # D1 Control vs KO - End
    plt.figure(figsize=(15, 5))
    sem_plot(end1 * 100 + 5, "end_1", color="blue", fill_color="blue")
    sem_plot(end2 * 100 + 5, "end_2", color="green", fill_color="green")
    plt.margins(x=0)
    plt.ylim((-2, 15))
    plt.gca().xaxis.set_major_formatter(lambda x, pos: f"{math.ceil(x / fs)}")
    plt.savefig(
        f"{OUTPUT_DIR}/midend-end-D1-control-ko-day5.svg", format="svg", dpi=300
    )
    plt.savefig(f"{OUTPUT_DIR}/midend-end-D1-control-ko-day5.png", dpi=300)
    plt.close()

    # A2A Control vs KO - Mid
    plt.figure(figsize=(15, 5))
    sem_plot(mid3 * 100 + 3, "mid_3", color="blue", fill_color="blue")
    sem_plot(mid4 * 100 + 3, "mid_4", color="orange", fill_color="orange")
    plt.margins(x=0)
    plt.ylim((-2, 15))
    plt.gca().xaxis.set_major_formatter(lambda x, pos: f"{math.ceil(x / fs)}")
    plt.savefig(
        f"{OUTPUT_DIR}/midend-mid-A2A-control-ko-day5.svg", format="svg", dpi=300
    )
    plt.savefig(f"{OUTPUT_DIR}/midend-mid-A2A-control-ko-day5.png", dpi=300)
    plt.close()

    # A2A Control vs KO - End
    plt.figure(figsize=(15, 5))
    sem_plot((end3 + 0.03) * 100, "end_3", color="blue", fill_color="blue")
    sem_plot((end4 + 0.03) * 100, "end_4", color="orange", fill_color="orange")
    plt.margins(x=0)
    plt.ylim((-2, 15))
    plt.gca().xaxis.set_major_formatter(lambda x, pos: f"{math.ceil(x / fs)}")
    plt.savefig(
        f"{OUTPUT_DIR}/midend-end-A2A-control-ko-day5.svg", format="svg", dpi=300
    )
    plt.savefig(f"{OUTPUT_DIR}/midend-end-A2A-control-ko-day5.png", dpi=300)
    plt.close()


def plot_heatmaps(mid1, mid2, mid3, mid4, end1, end2, end3, end4, fs):
    """Generate heatmap visualizations for all groups."""
    from matplotlib.text import Text

    heatmap_configs = [
        (mid1, 0.05, 2, 8, "mid1"),
        (mid2, 0.05, 2, 8, "mid2"),
        (mid3, 0.03, 0, 8, "mid3"),
        (mid4, 0.03, 0, 8, "mid4"),
        (end1, 0.05, 4, 8, "end1"),
        (end2, 0.05, 4, 12, "end2"),
        (end3, 0.03, 2, 10, "end3"),
        (end4, 0.03, 2, 10, "end4"),
    ]

    for data, offset, vmin, vmax, name in heatmap_configs:
        plt.figure(figsize=(10, 6))
        g = sns.heatmap(
            pd.DataFrame(data=(data + offset) * 100),
            cmap="plasma",
            vmin=vmin,
            vmax=vmax,
        )
        g.set_xticks(range(0, len(data[0]), 1000))
        labels = [Text(i * 1000, 0, f"{i}") for i in range(9)]
        g.set_xticklabels(labels, rotation=0)
        plt.savefig(f"{OUTPUT_DIR}/{name}_heatmap.png")
        plt.close()


def plot_startend_analysis(group_startends, fs, ylim=(-1, 14)):
    """Generate start/end analysis plots for all groups."""
    colors = ["#060352", "#09049C", "#0900FF", "#3366FF", "#80BCF2"]

    # Example: Group 2 ends
    plt.figure(figsize=(15, 5))
    data = group_startends[1][1]  # Group 2 ends
    sem_plot(
        np.array([(d.mean(axis=0) + 0.05) * 100 for d in data]),
        color="green",
        fill_color="green",
    )
    plt.margins(x=0)
    plt.ylim(ylim)
    plt.gca().xaxis.set_major_formatter(lambda x, pos: f"{math.ceil(x / fs)}")

    data = group_startends[3][1]  # Group 4 ends
    sem_plot(
        np.array([(d.mean(axis=0) + 0.04) * 100 for d in data]),
        color="#99F7A2",
        fill_color="#99F7A2",
    )
    plt.margins(x=0)
    plt.ylim(ylim)
    plt.gca().xaxis.set_major_formatter(lambda x, pos: f"{math.ceil(x / fs)}")
    plt.savefig(f"{OUTPUT_DIR}/startend-end-knockout-day1&5.png", dpi=300)
    plt.close()


def main():
    """Main analysis pipeline."""
    global FS

    # Load data
    print("Loading data...")
    start_end_ticks_900s = (
        pd.read_csv(f"{DATA_DIR}/900s_whole_trace.csv", index_col=0)
        .reset_index()
        .rename(columns={"index": "folder"})
    )

    mid_end_ticks = (
        pd.read_csv(f"{DATA_DIR}/mid_end_transposed.csv", index_col=0)
        .reset_index()
        .rename(columns={"index": "folder"})
    )

    start_end_ticks = (
        pd.read_csv(f"{DATA_DIR}/start_end_transposed.csv", index_col=0)
        .reset_index()
        .rename(columns={"index": "folder"})
    )

    # Initialize sampling frequency from first file
    data = tdt.read_block("JZ_exp-230114-153939-halo-day5-0351-ctrl1")
    FS = data.streams._405p.fs

    # Run 900s trace analysis
    print("Running 900s trace analysis...")
    Stds, StdOverMeans, AUCs = analyze_900s_traces(start_end_ticks_900s)

    # Mid/End Analysis
    print("Running mid/end analysis...")
    mid1, end1 = calc_group_array_midend(1, mid_end_ticks, FS)
    mid2, end2 = calc_group_array_midend(2, mid_end_ticks, FS)
    mid3, end3 = calc_group_array_midend(3, mid_end_ticks, FS)
    mid4, end4 = calc_group_array_midend(4, mid_end_ticks, FS)

    plot_midend_analysis(mid1, end1, mid2, end2, mid3, mid3, mid4, end4, FS)

    # Heatmaps
    print("Generating heatmaps...")
    plot_heatmaps(mid1, mid2, mid3, mid4, end1, end2, end3, end4, FS)

    # Start/End Analysis
    print("Running start/end analysis...")
    group_startends = [
        calc_group_array_startend(g, start_end_ticks, FS) for g in range(1, 7)
    ]
    plot_startend_analysis(group_startends, FS)

    print(f"Analysis complete. Outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

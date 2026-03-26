"""
Spike-Triggered Photometry Visualization

Extracts photometry (dF/F) and EEG segments centered on detected spikes and plots
a 2x2 grid:
  - Top-left:    individual photometry traces overlaid
  - Bottom-left: mean photometry trace with 95% CI band
  - Top-right:   individual EEG traces overlaid
  - Bottom-right: mean EEG trace with 95% CI band

Usage:
    python spike_triggered_photometry.py
    python spike_triggered_photometry.py --directory output/3154
    python spike_triggered_photometry.py --signals path/to/signals.npz --results path/to/results.json
    python spike_triggered_photometry.py --directory output/3154 --output spike_triggered.png
"""

import argparse
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def find_matching_files(directory, results_suffix=""):
    """Auto-discover signals NPZ and results JSON from a directory."""
    json_pattern = f"*_results{results_suffix}.json"
    npz_files = sorted(glob.glob(os.path.join(directory, "*_signals.npz")))
    json_files = sorted(glob.glob(os.path.join(directory, json_pattern)))

    if not npz_files:
        raise FileNotFoundError(f"No *_signals.npz files found in {directory}")
    if not json_files:
        raise FileNotFoundError(f"No {json_pattern} files found in {directory}")

    if len(npz_files) > 1:
        print(f"Warning: Multiple signal files found, using {npz_files[0]}")
    if len(json_files) > 1:
        print(f"Warning: Multiple result files found, using {json_files[0]}")

    return npz_files[0], json_files[0]


def load_data(signals_path, results_path):
    """Load photometry signal, EEG signal, and spike times."""
    data = np.load(signals_path, allow_pickle=True)

    if "photometry_processed" not in data:
        print("Error: No photometry_processed in signals file.")
        sys.exit(1)

    photometry = data["photometry_processed"]
    eeg = data["eeg_zscored"]
    fs = float(data["sampling_rate"])

    with open(results_path) as f:
        results = json.load(f)

    spike_samples = [s["time_samples"] for s in results["spikes"]]

    return photometry, eeg, fs, spike_samples


def filter_spikes(spike_samples, signal_length, half_window, proximity_samples):
    """Filter spikes that are too close to edges or to each other."""
    # Edge filter: need full window on both sides
    edge_valid = [
        s for s in spike_samples
        if s >= half_window and s + half_window <= signal_length
    ]

    # Proximity filter: skip spikes within proximity_samples of previous accepted spike
    accepted = []
    for s in edge_valid:
        if not accepted or (s - accepted[-1]) > proximity_samples:
            accepted.append(s)

    return accepted


def extract_segments(signal, spike_samples, half_window):
    """Extract signal segments centered on each spike."""
    segments = np.empty((len(spike_samples), 2 * half_window))
    for i, s in enumerate(spike_samples):
        segments[i] = signal[s - half_window : s + half_window]
    return segments


def _plot_traces(ax, time, segments, color, alpha, ylabel, title):
    """Plot individual overlaid traces on ax."""
    # Clip y-axis to 99th percentile so outlier spikes don't squish the view
    ylim = np.percentile(np.abs(segments), 99)
    for seg in segments:
        ax.plot(time, seg, color=color, alpha=alpha, linewidth=0.3)
    ax.axvline(0, color="red", linestyle="--", linewidth=0.8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-ylim, ylim)


def _plot_mean_ci(ax, time, segments, color, ylabel):
    """Plot mean + 95% CI on ax."""
    n = segments.shape[0]
    mean_trace = segments.mean(axis=0)
    sem_trace = segments.std(axis=0) / np.sqrt(n)
    ax.plot(time, mean_trace, color=color, linewidth=1.5, label="Mean")
    ax.fill_between(
        time,
        mean_trace - 1.96 * sem_trace,
        mean_trace + 1.96 * sem_trace,
        color=color, alpha=0.25, label="95% CI"
    )
    ax.axvline(0, color="red", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Time relative to spike (s)")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ylim = max(abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]))
    ax.set_ylim(-ylim, ylim)


def plot_spike_triggered(photo_segments, eeg_segments, fs, half_window,
                         output_path=None):
    """Create a 2x2 spike-triggered figure (photometry left, EEG right)."""
    n_spikes = photo_segments.shape[0]
    time = np.arange(-half_window, half_window) / fs  # seconds

    center_idx = half_window

    # Align EEG polarity: flip segments where the spike peak is negative
    # so all spikes appear as upward deflections before zero-centering
    peak_window = int(0.05 * fs)  # ±50 ms around center
    lo, hi = center_idx - peak_window, center_idx + peak_window
    peak_vals = eeg_segments[:, lo:hi].mean(axis=1)
    polarity = np.where(peak_vals < 0, -1, 1)
    eeg_segments = eeg_segments * polarity[:, np.newaxis]

    # Zero-center both signals at t=0 (spike onset)
    photo_zeroed = photo_segments - photo_segments[:, center_idx:center_idx+1]
    eeg_zeroed = eeg_segments - eeg_segments[:, center_idx:center_idx+1]

    sns.set_theme(style="whitegrid", context="paper", palette="muted")
    palette = sns.color_palette("muted")
    photo_color = palette[0]       # blue
    photo_mean_color = palette[1]  # orange
    eeg_color = palette[2]         # green
    eeg_mean_color = palette[3]    # red-ish

    trace_alpha = max(0.05, min(0.3, 15.0 / n_spikes))

    fig, axes = plt.subplots(
        2, 2, figsize=(14, 8), constrained_layout=True,
        sharex="col"
    )
    ax_photo_traces, ax_eeg_traces = axes[0, 0], axes[0, 1]
    ax_photo_mean,   ax_eeg_mean   = axes[1, 0], axes[1, 1]

    # --- Left column: Photometry ---
    _plot_traces(
        ax_photo_traces, time, photo_zeroed,
        color=photo_color, alpha=trace_alpha,
        ylabel="dF/F (zeroed at spike)",
        title="Photometry — individual traces"
    )
    _plot_mean_ci(
        ax_photo_mean, time, photo_zeroed,
        color=photo_mean_color,
        ylabel="dF/F (zeroed at spike)"
    )
    ax_photo_mean.set_title("Photometry — mean ± 95% CI", fontsize=11, fontweight="bold")

    # --- Right column: EEG ---
    _plot_traces(
        ax_eeg_traces, time, eeg_zeroed,
        color=eeg_color, alpha=trace_alpha,
        ylabel="EEG z-score (zeroed at spike)",
        title="EEG — individual traces"
    )
    _plot_mean_ci(
        ax_eeg_mean, time, eeg_zeroed,
        color=eeg_mean_color,
        ylabel="EEG z-score (zeroed at spike)"
    )
    ax_eeg_mean.set_title("EEG — mean ± 95% CI", fontsize=11, fontweight="bold")

    fig.suptitle("Spike-Triggered Analysis", fontsize=14, fontweight="bold")

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure to {output_path}")
    else:
        plt.show()


def load_combined_segments(segments_path):
    """Load pre-extracted segments from a combined NPZ (produced by combine.py)."""
    data = np.load(segments_path, allow_pickle=True)
    photo_segments = data["photo_segments"]
    eeg_segments = data["eeg_segments"]
    fs = float(data["fs"])
    half_window = int(data["half_window"])
    n_datasets = len(np.unique(data["dataset_labels"])) if "dataset_labels" in data else "?"
    return photo_segments, eeg_segments, fs, half_window, n_datasets


def main():
    parser = argparse.ArgumentParser(
        description="Spike-triggered photometry visualization"
    )
    parser.add_argument(
        "--segments", default=None,
        help="Path to combined segments NPZ from combine.py (skips all other load args)"
    )
    parser.add_argument(
        "--directory", "-d", default="output/3154",
        help="Directory containing signals NPZ and results JSON (default: output/3154)"
    )
    parser.add_argument(
        "--signals", "-s", default=None,
        help="Path to *_signals.npz file (overrides --directory)"
    )
    parser.add_argument(
        "--results", "-r", default=None,
        help="Path to *_results.json file (overrides --directory)"
    )
    parser.add_argument(
        "--window", "-w", type=float, default=60.0,
        help="Total window duration in seconds (default: 60.0)"
    )
    parser.add_argument(
        "--proximity", type=float, default=0.05,
        help="Minimum inter-spike interval in seconds for filtering (default: 0.05)"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output file path for saving figure (if not set, displays interactively)"
    )
    parser.add_argument(
        "--results-suffix", default="",
        help="Suffix for results JSON files, e.g. '_preictal' for *_results_preictal.json"
    )
    args = parser.parse_args()

    if args.segments:
        # --- Combined mode: load pre-extracted segments from combine.py ---
        print(f"Loading combined segments: {args.segments}")
        photo_segments, eeg_segments, fs, half_window, n_datasets = load_combined_segments(args.segments)
        print(f"Total spikes: {photo_segments.shape[0]} across {n_datasets} dataset(s)")
    else:
        # --- Single-dataset mode: load full signals and extract on the fly ---
        if args.signals and args.results:
            signals_path, results_path = args.signals, args.results
        else:
            signals_path, results_path = find_matching_files(args.directory, args.results_suffix)

        print(f"Signals: {signals_path}")
        print(f"Results: {results_path}")

        photometry, eeg, fs, spike_samples = load_data(signals_path, results_path)
        print(f"Photometry: {len(photometry)} samples at {fs} Hz ({len(photometry)/fs:.1f}s)")
        print(f"EEG: {len(eeg)} samples at {fs} Hz ({len(eeg)/fs:.1f}s)")
        print(f"Total spikes: {len(spike_samples)}")

        half_window = int((args.window / 2) * fs)
        proximity_samples = int(args.proximity * fs)

        min_length = min(len(photometry), len(eeg))
        accepted = filter_spikes(spike_samples, min_length, half_window, proximity_samples)
        n_rejected = len(spike_samples) - len(accepted)
        print(f"Accepted spikes: {len(accepted)} ({n_rejected} rejected)")

        if len(accepted) == 0:
            print("Error: No valid spikes after filtering.")
            sys.exit(1)

        photo_segments = extract_segments(photometry, accepted, half_window)
        eeg_segments = extract_segments(eeg, accepted, half_window)

    plot_spike_triggered(photo_segments, eeg_segments, fs, half_window, args.output)


if __name__ == "__main__":
    main()

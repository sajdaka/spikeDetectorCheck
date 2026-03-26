"""
Combine spike-triggered segments across all output datasets.

Scans every subdirectory of an output root for matching *_signals.npz and
*_results.json pairs, extracts a fixed-width window around each accepted spike,
and saves the concatenated segments to a single NPZ file that can be passed
directly to spike_triggered_photometry.py via --segments.

Usage:
    python combine.py
    python combine.py --output-dir output --output combined_spike_segments.npz
    python combine.py --window 10.0 --proximity 0.05
"""

import argparse
import glob
import json
import os
import sys

import numpy as np


def find_pairs(output_dir, results_suffix=""):
    """Return list of (npz_path, json_path, label) for every valid subdir.

    results_suffix: e.g. "_preictal" to match *_results_preictal.json
    """
    json_pattern = f"*_results{results_suffix}.json"
    pairs = []
    for subdir in sorted(os.listdir(output_dir)):
        full = os.path.join(output_dir, subdir)
        if not os.path.isdir(full):
            continue

        npz_files = sorted(glob.glob(os.path.join(full, "*_signals.npz")))
        json_files = sorted(glob.glob(os.path.join(full, json_pattern)))

        if not npz_files or not json_files:
            print(f"  Skipping {subdir}: missing NPZ or {json_pattern}")
            continue

        if len(npz_files) > 1:
            print(f"  Warning: multiple NPZ in {subdir}, using {os.path.basename(npz_files[0])}")
        if len(json_files) > 1:
            print(f"  Warning: multiple JSON in {subdir}, using {os.path.basename(json_files[0])}")

        pairs.append((npz_files[0], json_files[0], subdir))

    return pairs


def load_signals(npz_path, json_path):
    """Load photometry, EEG, fs, and spike sample indices from a dataset pair."""
    data = np.load(npz_path, allow_pickle=True)

    if "photometry_processed" not in data:
        raise ValueError(f"No photometry_processed in {npz_path}")

    photometry = data["photometry_processed"]
    eeg = data["eeg_zscored"]
    fs = float(data["sampling_rate"])

    with open(json_path) as f:
        results = json.load(f)

    spike_samples = [s["time_samples"] for s in results["spikes"]]
    return photometry, eeg, fs, spike_samples


def filter_spikes(spike_samples, signal_length, half_window, proximity_samples):
    edge_valid = [
        s for s in spike_samples
        if s >= half_window and s + half_window <= signal_length
    ]
    accepted = []
    for s in edge_valid:
        if not accepted or (s - accepted[-1]) > proximity_samples:
            accepted.append(s)
    return accepted


def extract_segments(signal, spike_samples, half_window):
    segments = np.empty((len(spike_samples), 2 * half_window))
    for i, s in enumerate(spike_samples):
        segments[i] = signal[s - half_window: s + half_window]
    return segments


def main():
    parser = argparse.ArgumentParser(description="Combine spike-triggered segments across datasets")
    parser.add_argument(
        "--output-dir", "-d", default="output",
        help="Root directory containing per-dataset subdirectories (default: output)"
    )
    parser.add_argument(
        "--output", "-o", default="combined_spike_segments.npz",
        help="Output NPZ file path (default: combined_spike_segments.npz)"
    )
    parser.add_argument(
        "--window", "-w", type=float, default=60.0,
        help="Total window duration in seconds around each spike (default: 60.0)"
    )
    parser.add_argument(
        "--proximity", type=float, default=0.05,
        help="Minimum inter-spike interval in seconds for filtering (default: 0.05)"
    )
    parser.add_argument(
        "--results-suffix", default="",
        help="Suffix for results JSON files, e.g. '_preictal' for *_results_preictal.json"
    )
    args = parser.parse_args()

    pairs = find_pairs(args.output_dir, args.results_suffix)
    if not pairs:
        print(f"No valid dataset pairs found in {args.output_dir}")
        sys.exit(1)

    print(f"Found {len(pairs)} dataset(s)\n")

    all_photo_segments = []
    all_eeg_segments = []
    all_labels = []
    fs_ref = None
    half_window = None

    for npz_path, json_path, label in pairs:
        print(f"Processing: {label}")
        try:
            photometry, eeg, fs, spike_samples = load_signals(npz_path, json_path)
        except Exception as e:
            print(f"  ERROR loading {label}: {e}")
            continue

        if fs_ref is None:
            fs_ref = fs
            half_window = int((args.window / 2) * fs)
        elif fs != fs_ref:
            print(f"  WARNING: {label} has fs={fs}, expected {fs_ref} — skipping")
            continue

        proximity_samples = int(args.proximity * fs)
        min_length = min(len(photometry), len(eeg))
        accepted = filter_spikes(spike_samples, min_length, half_window, proximity_samples)
        n_rejected = len(spike_samples) - len(accepted)

        print(f"  Spikes: {len(spike_samples)} total, {len(accepted)} accepted, {n_rejected} rejected")

        if not accepted:
            print(f"  No valid spikes — skipping")
            continue

        photo_seg = extract_segments(photometry, accepted, half_window)
        eeg_seg = extract_segments(eeg, accepted, half_window)

        all_photo_segments.append(photo_seg)
        all_eeg_segments.append(eeg_seg)
        all_labels.extend([label] * len(accepted))

    if not all_photo_segments:
        print("No segments extracted — nothing to save.")
        sys.exit(1)

    photo_combined = np.concatenate(all_photo_segments, axis=0)
    eeg_combined = np.concatenate(all_eeg_segments, axis=0)
    labels_arr = np.array(all_labels)

    np.savez(
        args.output,
        photo_segments=photo_combined,
        eeg_segments=eeg_combined,
        dataset_labels=labels_arr,
        fs=np.float64(fs_ref),
        half_window=np.int64(half_window),
    )

    print(f"\nCombined: {photo_combined.shape[0]} total spikes across {len(pairs)} datasets")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()

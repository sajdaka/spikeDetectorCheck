"""
Cohen's d effect size bar chart for spike classifier features.

Compares true spikes vs artifacts using labeled training data.
Horizontal bars with feature names on y-axis and reference lines
for small (0.2), medium (0.5), and large (0.8) effect sizes.

Usage: python plot_cohens_d.py <signals.npz> <training.csv>

The training CSV must have 'Time' and 'Class' columns (as used by classifier.py).
Class 1 = true spike, Class 0 = artifact.
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from plot_features import extract_features, FEATURE_NAMES


def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / (pooled_std + 1e-10)


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <signals.npz> <training.csv>")
        sys.exit(1)

    # Load EEG
    data = np.load(sys.argv[1], allow_pickle=True)
    eeg = data['eeg_processed']
    eeg_z = data['eeg_zscored'] if 'eeg_zscored' in data else eeg
    fs = float(data['sampling_rate'])

    # Load training labels
    import csv
    times = []
    labels = []
    with open(sys.argv[2], 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row['Time']))
            labels.append(int(row['Class']))
    times = np.array(times, dtype=int)
    labels = np.array(labels)

    # Filter to valid spike locations
    valid = times < len(eeg) - 100
    times = times[valid]
    labels = labels[valid]

    if len(times) == 0:
        print("No valid spike locations found.")
        sys.exit(1)

    # Extract features
    features = extract_features(times, eeg, eeg_z, fs)

    # Split by class
    true_spikes = features[labels == 1]
    artifacts = features[labels == 0]

    print(f"True spikes: {len(true_spikes)}, Artifacts: {len(artifacts)}")

    if len(true_spikes) < 2 or len(artifacts) < 2:
        print("Need at least 2 samples in each class.")
        sys.exit(1)

    # Compute Cohen's d for each feature
    d_values = np.array([cohens_d(true_spikes[:, i], artifacts[:, i])
                         for i in range(features.shape[1])])

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 8))

    y_pos = np.arange(len(FEATURE_NAMES))
    colors = ['#c44e52' if d < 0 else '#4c72b0' for d in d_values]

    ax.barh(y_pos, d_values, color=colors, edgecolor='black', linewidth=0.5, height=0.7)

    # Reference lines for effect size thresholds
    for thresh, label in [(0.2, 'Small'), (0.5, 'Medium'), (0.8, 'Large')]:
        ax.axvline(thresh, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax.axvline(-thresh, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax.text(thresh, len(FEATURE_NAMES) - 0.3, label, ha='center', va='bottom',
                fontsize=8, color='gray')

    ax.axvline(0, color='black', linewidth=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(FEATURE_NAMES, fontsize=10)
    ax.set_xlabel("Cohen's d (True Spikes vs Artifacts)")
    ax.set_title("Feature Effect Sizes (Cohen's d)")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig('cohens_d_features.png', dpi=300, bbox_inches='tight')
    print("Saved cohens_d_features.png")
    plt.show()


if __name__ == '__main__':
    main()

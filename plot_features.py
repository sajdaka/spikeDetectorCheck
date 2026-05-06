"""
Visualize spike classifier features:
  1. Normalized feature heatmap (spikes x features)
  2. Width distribution box plot

Usage: python plot_features.py <signals.npz> <results.json>

Extracts features from the EEG signal at each detected spike location
using the same logic as classifier.py, then plots them.
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


# --- Feature names matching classifier.py build_features order ---
SHAPE_BINS = [f'shape_bin_{i+1}' for i in range(10)]
FEATURE_NAMES = SHAPE_BINS + [
    'width', 'zband_fraction', 'max_slope', 'min_slope', 'min_to_max',
    'peak_freq', 'spectral_entropy', 'power_ratio',
    'symmetry', 'sharpness', 'rise_fall_ratio'
]


# --- Feature extraction (mirrored from classifier.py) ---

def get_segment(spike_sample, eeg, window=100):
    start = int(spike_sample - window)
    end = int(spike_sample + window)
    seg = eeg[max(start, 0):min(end, len(eeg))].copy()
    if np.abs(np.min(seg)) > np.abs(np.max(seg)):
        seg = -seg
    return seg


def extract_features(spike_samples, eeg, eeg_z, fs=1000.0):
    """Return (n_spikes, n_features) array."""
    rows = []
    for spike in spike_samples:
        seg = get_segment(spike, eeg)
        feat = []

        # shape bins (10 bins of 20 samples over 200-sample window)
        for i in range(0, 200, 20):
            if i + 20 <= len(seg):
                feat.append(np.mean(seg[i:i+20]))
            else:
                feat.append(np.mean(seg[i:]) if len(seg) > i else 0.0)

        # width (FWHM in samples)
        if len(seg) >= 200:
            center = seg[50:150]
            peak_val = np.max(np.abs(center))
            peak_local = np.argmax(np.abs(center))
            peak_idx = 50 + peak_local
            half = peak_val / 2
            left = next((i for i in range(peak_idx, 0, -1) if np.abs(seg[i]) <= half), None)
            right = next((i for i in range(peak_idx, len(seg)) if np.abs(seg[i]) <= half), None)
            width = (right - left) if (left is not None and right is not None) else 50
        else:
            width = 100
        feat.append(width)

        # zband fraction
        z_seg = eeg_z[max(int(spike-100), 0):min(int(spike+100), len(eeg_z))]
        above = np.sum((z_seg >= 1) | (z_seg <= -1))
        feat.append(above / 200)

        # slopes from shape bins
        shape = feat[:10]
        slopes = [(shape[i+1] - shape[i]) / 20 for i in range(len(shape)-1)]
        feat.append(np.max(np.abs(slopes)))   # max_slope
        feat.append(np.min(np.abs(slopes)))   # min_slope

        # min to max
        feat.append(np.abs(np.max(seg) - np.min(seg)))

        # frequency features
        freqs = np.fft.rfftfreq(len(seg), 1/fs)
        fft_vals = np.abs(np.fft.rfft(seg))
        feat.append(freqs[np.argmax(fft_vals[1:]) + 1])  # peak_freq
        psd = fft_vals ** 2
        psd_norm = psd / (np.sum(psd) + 1e-10)
        feat.append(-np.sum(psd_norm * np.log2(psd_norm + 1e-10)))  # spectral_entropy
        low = np.sum(fft_vals[(freqs >= 1) & (freqs < 10)])
        high = np.sum(fft_vals[(freqs >= 15) & (freqs < 40)])
        feat.append(high / (low + 1e-10))  # power_ratio

        # symmetry features
        raw_seg = eeg[max(int(spike-100), 0):min(int(spike+100), len(eeg))]
        pk = np.argmax(np.abs(raw_seg))
        pk_val = np.abs(raw_seg[pk])
        left_area = np.sum(np.abs(raw_seg[:pk]))
        right_area = np.sum(np.abs(raw_seg[pk:]))
        feat.append((left_area - right_area) / (left_area + right_area + 1e-10))  # symmetry
        if 1 < pk < len(raw_seg) - 1:
            feat.append(np.abs(raw_seg[pk+1] - 2*raw_seg[pk] + raw_seg[pk-1]))  # sharpness
        else:
            feat.append(0.0)
        half_pk = pk_val / 2
        rise_start = next((i for i in range(pk, 0, -1) if np.abs(raw_seg[i]) <= half_pk), None)
        rise_t = pk - rise_start if rise_start is not None else pk
        fall_end = next((i for i in range(pk, len(raw_seg)) if np.abs(raw_seg[i]) <= half_pk), None)
        fall_t = fall_end - pk if fall_end is not None else len(raw_seg) - pk
        feat.append(rise_t / (fall_t + 1e-10))  # rise_fall_ratio

        rows.append(feat)

    return np.array(rows)


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <signals.npz> <results.json>")
        sys.exit(1)

    # Load data
    data = np.load(sys.argv[1], allow_pickle=True)
    eeg = data['eeg_processed']
    eeg_z = data['eeg_zscored'] if 'eeg_zscored' in data else eeg
    fs = float(data['sampling_rate'])

    with open(sys.argv[2]) as f:
        spikes = json.load(f)['spikes']

    spike_samples = np.array([s['time_samples'] for s in spikes], dtype=int)
    spike_samples = spike_samples[spike_samples < len(eeg) - 100]

    if len(spike_samples) == 0:
        print("No valid spikes found.")
        sys.exit(1)

    features = extract_features(spike_samples, eeg, eeg_z, fs)

    # Normalize each feature column to [0, 1] for the heatmap
    feat_min = features.min(axis=0)
    feat_max = features.max(axis=0)
    feat_range = feat_max - feat_min
    feat_range[feat_range == 0] = 1
    features_norm = (features - feat_min) / feat_range

    # --- Plot (transposed heatmap on top, box plot below) ---
    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(spike_samples) * 0.15), 10),
                             gridspec_kw={'height_ratios': [3, 1]})

    # 1) Normalized feature heatmap (transposed: features on y-axis, spikes on x-axis)
    ax_heat = axes[0]
    im = ax_heat.imshow(features_norm.T, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax_heat.set_yticks(range(len(FEATURE_NAMES)))
    ax_heat.set_yticklabels(FEATURE_NAMES, fontsize=8)
    ax_heat.set_xlabel('Spike index')
    ax_heat.set_title('Normalized Feature Heatmap')
    fig.colorbar(im, ax=ax_heat, fraction=0.02, pad=0.02, label='Normalized value')

    # 2) Width distribution box plot (horizontal)
    ax_box = axes[1]
    width_idx = FEATURE_NAMES.index('width')
    widths_ms = features[:, width_idx] * (1000 / fs)  # samples → ms
    bp = ax_box.boxplot(widths_ms, vert=False, patch_artist=True,
                        boxprops=dict(facecolor='#4C72B0', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2))
    ax_box.set_xlabel('Width (ms)')
    ax_box.set_title('Spike Width Distribution')
    ax_box.set_yticklabels(['All spikes'])

    plt.tight_layout()
    plt.savefig('feature_heatmap_and_width.png', dpi=300, bbox_inches='tight')
    print("Saved feature_heatmap_and_width.png")
    plt.show()


if __name__ == '__main__':
    main()

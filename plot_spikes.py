"""
Simple EEG + spike overlay viewer.
Usage: python plot_spikes.py <signals.npz> <results.json>
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <signals.npz> <results.json>")
        sys.exit(1)

    npz_path = sys.argv[1]
    json_path = sys.argv[2]

    # Load EEG
    data = np.load(npz_path, allow_pickle=True)
    eeg = data['eeg_zscored'] if 'eeg_zscored' in data else data['eeg_processed']
    fs = float(data['sampling_rate'])
    time = np.arange(len(eeg)) / fs

    # Load spikes
    with open(json_path) as f:
        results = json.load(f)
    spikes = results['spikes']

    spike_times = np.array([s['time_seconds'] for s in spikes])
    spike_samples = np.array([s['time_samples'] for s in spikes], dtype=int)
    # Clamp sample indices to valid range
    spike_samples = spike_samples[spike_samples < len(eeg)]
    spike_times = spike_times[:len(spike_samples)]
    spike_amps = eeg[spike_samples]

    # Plot
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(time, eeg, linewidth=0.5, color='black', label='EEG (z-scored)')
    ax.scatter(spike_times, spike_amps, marker='x', color='red', s=40,
               linewidths=1.5, zorder=5, label=f'Spikes (n={len(spike_times)})')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (z-score)')
    ax.set_title('EEG with Detected Spikes')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

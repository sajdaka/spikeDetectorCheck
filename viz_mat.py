import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from dataLoad import MATLoader


def plot_mat(filepath: str):
    loader = MATLoader()
    record = loader.load(filepath)

    n_channels = record.n_channels
    fig, axes = plt.subplots(n_channels, 1, figsize=(14, 3 * n_channels), sharex=True)
    if n_channels == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(record.timestamps, record.data[i], linewidth=0.5, color='steelblue')
        ax.set_ylabel(f'Ch {i} (µV)')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(Path(filepath).name, fontsize=12)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python viz_mat.py <path/to/file.mat>")
        sys.exit(1)
    plot_mat(sys.argv[1])

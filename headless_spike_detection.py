"""
Headless spike detection pipeline for batch processing
Runs EEG preprocessing, spike detection, and exports results without GUI
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, List
from collections import Counter
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import json
from dataclasses import asdict
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from config import ConfigManager
from dataLoad import DataManager
from DataPreprocessing import PreprocessingPipeline
from SpikeDetection import SpikeDetector, SpikeDetectionParams, BaselineNormalizer, SpikeEvent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HeadlessSpikeDetection:
    """Headless spike detection pipeline"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize headless pipeline

        Args:
            config_path: Path to configuration YAML file
        """
        self.config_manager = ConfigManager(config_path)
        self.data_manager = DataManager()

        # Initialize preprocessing pipeline
        config = self.config_manager.config
        preprocessing_config = {
            'eeg': {
                'sample_rate': config.preprocessing.eeg_sampling_freq,
                'notch_frequency': config.preprocessing.notch_frequency,
                'notch_quality': config.preprocessing.notch_quality,
                'bandpass_low': config.preprocessing.bandpass_low,
                'bandpass_high': config.preprocessing.bandpass_high,
                'artifact_threshold': config.preprocessing.interpolation_threshold
            },
            'photometry': {
                'sample_rate': config.detection.fs,
                'lowpass_cutoff': config.preprocessing.lowpass_cutoff,
                'gaussian_sigma': config.preprocessing.gaussian_sigma,
                'savgol_window': config.preprocessing.savgol_window,
                'savgol_polyorder': config.preprocessing.savgol_polyorder
            },
            'baseline_start_time': config.detection.baseline_start_time,
            'baseline_end_time': config.detection.baseline_end_time
        }

        self.preprocessing_pipeline = PreprocessingPipeline(preprocessing_config)

        # Will be initialized per run
        self.spike_detector = None
        self.current_channel = None

    def _resolve_eeg_path(self, eeg_file: str) -> str:
        """
        Resolve EEG file path - if just filename, look in config.data_paths.eeg_data_dir

        Args:
            eeg_file: Full path or filename

        Returns:
            Resolved full path to EEG file
        """
        eeg_path = Path(eeg_file)

        # If it's an absolute path and exists, use it
        if eeg_path.is_absolute() and eeg_path.exists():
            logger.info(f"Using absolute path: {eeg_file}")
            return str(eeg_path)

        # If it's a relative path and exists, use it
        if eeg_path.exists():
            logger.info(f"Using relative path: {eeg_file}")
            return str(eeg_path.resolve())

        # Otherwise, look in config.data_paths.eeg_data_dir
        config_eeg_dir = Path(self.config_manager.config.data_paths.eeg_data_dir)
        potential_path = config_eeg_dir / eeg_file

        if potential_path.exists():
            logger.info(f"Found file in config EEG directory: {potential_path}")
            return str(potential_path)

        # If still not found, raise error
        raise FileNotFoundError(
            f"Could not find EEG file: {eeg_file}\n"
            f"Searched in:\n"
            f"  - Current directory\n"
            f"  - {config_eeg_dir}\n"
            f"Please provide full path or ensure file exists in config.data_paths.eeg_data_dir"
        )

    def run_detection(self,
                     eeg_file: str,
                     channel: str,
                     output_dir: Optional[str] = None,
                     processing_steps: Optional[List[str]] = None) -> dict:
        """
        Run complete spike detection pipeline

        Args:
            eeg_file: Path to EEG data file (full path or filename to search in config.data_paths.eeg_data_dir)
            channel: Channel to analyze
            output_dir: Directory for output files (defaults to config.data_paths.output_dir)
            processing_steps: List of processing steps (e.g., ["Meiling Denoise", "Meiling Detrend"])

        Returns:
            Dictionary with results
        """
        # Resolve EEG file path
        eeg_file_path = self._resolve_eeg_path(eeg_file)

        # Use config output dir if not specified
        if output_dir is None:
            output_dir = self.config_manager.config.data_paths.output_dir
            logger.info(f"Using config output directory: {output_dir}")

        logger.info(f"Starting headless spike detection pipeline")
        logger.info(f"EEG file: {eeg_file_path}")
        logger.info(f"Channel: {channel}")
        logger.info(f"Output directory: {output_dir}")

        self.current_channel = channel

        # Initialize spike detector with current channel
        config = self.config_manager.config
        detection_params = SpikeDetectionParams(
            fs=config.detection.fs,
            tmul=config.detection.tmul,
            absthresh=config.detection.absthresh,
            sur_time=config.detection.sur_time,
            close_to_edge=config.detection.close_to_edge,
            too_high_abs=config.detection.too_high_abs,
            spkdur_min=config.detection.spkdur_min,
            spkdur_max=config.detection.spkdur_max,
            channel=channel,
            baseline_end_time=config.detection.baseline_end_time,
            baseline_start_time=config.detection.baseline_start_time
        )
        self.spike_detector = SpikeDetector(detection_params)

        # Load EEG data
        logger.info("Loading EEG data...")
        eeg_data, eeg_record = self.data_manager.load_eeg_data(eeg_file_path, channel=channel)

        # Window data to 10:00-16:00 (3-9 hours from start)
        fs = config.detection.fs
        start_sample = int(10800 * fs)  # 10:00 (3 hours)
        end_sample = int(32400 * fs)    # 16:00 (9 hours)

        logger.info(f"Windowing data from 10:00 to 16:00 (samples {start_sample} to {end_sample})")
        #eeg_data = eeg_data[start_sample:end_sample]

        # Preprocess EEG
        logger.info("Preprocessing EEG data...")
        eeg_result = self.preprocessing_pipeline.process_eeg(eeg_data, eeg_record.sample_rate)

        # Detect spikes (do this BEFORE z-scoring to match GUI)
        logger.info("Detecting spikes...")
        spikes = self.spike_detector.detect_spikes(eeg_result.data)

        # Calculate z-scored version AFTER detection (matches GUI)
        eeg_zscored = BaselineNormalizer.baseline_zscore(
            eeg_result.data,
            config.detection.baseline_end_time,
            config.detection.baseline_start_time,
            fs
        )

        summary = self.spike_detector.get_detection_summary(spikes)

        logger.info(f"Detected {len(spikes)} spikes")

        # Free memory - delete raw data (no longer needed)
        del eeg_data
        import gc
        gc.collect()

        # Prepare results - include eeg_zscored
        analysis_results = {
            'spikes': spikes,
            'summary': summary,
            'eeg_result': eeg_result,
            'eeg_zscored': eeg_zscored
        }

        # Export results
        logger.info("Exporting results...")
        self._save_results(analysis_results, eeg_file_path, output_dir, processing_steps or [])

        logger.info("Pipeline complete!")

        return {
            'n_spikes': len(spikes),
            'summary': summary,
            'output_dir': output_dir
        }

    def _save_results(self, analysis_results: dict, eeg_file: str,
                     output_dir: str, processing_steps: List[str]):
        """Save results to directory"""
        import re
        import os

        # Get base filename (without extension)
        base_name = Path(eeg_file).stem

        # Create output directory structure: output_dir/eeg_filename/channel_name/
        output_path = Path(output_dir) / base_name / self.current_channel
        logger.info(f"Output path: {output_path}")
        output_path.mkdir(parents=True, exist_ok=True)

        spikes = analysis_results['spikes']
        summary = analysis_results['summary']

        logger.info(f"Saving results to {output_path}")

        # Save JSON results
        spikes_data = []
        for spike in spikes:
            spikes_data.append({
                'time_samples': int(spike.time_samples),
                'time_seconds': float(spike.time_seconds),
                'amplitude': float(spike.amplitude),
                'width_ms': float(spike.width_ms),
                'prominence': float(spike.prominence),
                'channel': spike.channel
            })

        results = {
            'summary': summary,
            'spikes': spikes_data,
            'configuration': {
                'detection_params': asdict(self.spike_detector.params),
                'processing_strategy': processing_steps
            },
            'files': {
                'eeg_file': eeg_file
            }
        }

        results_file = output_path / f"{base_name}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Saved JSON results: {results_file}")

        # Save processed signals
        eeg_result = analysis_results['eeg_result']
        eeg_zscored = analysis_results['eeg_zscored']

        save_data = {
            'eeg_processed': eeg_result.data,
            'eeg_zscored': eeg_zscored,
            'eeg_metadata': eeg_result.metadata,
            'sampling_rate': self.config_manager.config.detection.fs,
            'channel': self.current_channel
        }

        signals_file = output_path / f"{base_name}_signals.npz"
        np.savez_compressed(signals_file, **save_data)
        logger.info(f"Saved processed signals: {signals_file}")

        # Generate 15-minute interval statistics
        logger.info("Generating 15-minute interval statistics...")

        # Create 24 intervals (10:00-16:00 = 6 hours = 24 fifteen-minute intervals)
        interval_data = []
        interval_duration = 900  # 15 minutes in seconds

        # Get the processed EEG data for microVolt calculations
        eeg_processed = analysis_results['eeg_result'].data

        for interval_idx in range(24):
            # Calculate time window
            interval_start_seconds = interval_idx * interval_duration
            interval_end_seconds = (interval_idx + 1) * interval_duration

            # Format time window as HH:MM-HH:MM
            start_time = timedelta(hours=10) + timedelta(seconds=interval_start_seconds)
            end_time = timedelta(hours=10) + timedelta(seconds=interval_end_seconds)

            # Format as HH:MM
            start_str = f"{start_time.seconds // 3600:02d}:{(start_time.seconds % 3600) // 60:02d}"
            end_str = f"{end_time.seconds // 3600:02d}:{(end_time.seconds % 3600) // 60:02d}"
            time_window = f"{start_str}-{end_str}"

            # Filter spikes in this interval
            interval_spikes = [
                s for s in spikes
                if interval_start_seconds <= s.time_seconds < interval_end_seconds
            ]

            # Calculate statistics
            spike_count = len(interval_spikes)
            avg_amplitude_zscore = np.mean([s.amplitude for s in interval_spikes]) if interval_spikes else 0.0
            avg_width = np.mean([s.width_ms for s in interval_spikes]) if interval_spikes else 0.0

            # Calculate average amplitude in microvolts from processed EEG
            if interval_spikes:
                amplitudes_uv = []
                for spike in interval_spikes:
                    # Get the amplitude from the processed (non-z-scored) EEG signal
                    spike_idx = int(spike.time_samples)
                    if 0 <= spike_idx < len(eeg_processed):
                        # Convert to microvolts (multiply by 10^6)
                        amplitude_uv = eeg_processed[spike_idx] * (10 ** 6)
                        amplitudes_uv.append(amplitude_uv)
                avg_amplitude_uv = np.mean(amplitudes_uv) if amplitudes_uv else 0.0
            else:
                avg_amplitude_uv = 0.0

            interval_data.append({
                'Channel': self.current_channel,
                'Date': '',
                'Time_Window': time_window,
                'Total_Spike_Count': spike_count,
                'Average_Spike_Amplitude_uV': avg_amplitude_uv,
                'Average_Spike_Amplitude_ZScore': avg_amplitude_zscore,
                'Average_Spike_Width_ms': avg_width
            })

        df = pd.DataFrame(interval_data)
        interval_stats_file = output_path / f'{base_name}_15min_interval_stats.xlsx'
        df.to_excel(interval_stats_file, index=False)
        logger.info(f"Saved 15-minute interval statistics: {interval_stats_file}")

        logger.info(f"Results saved to {output_path}")


def main():
    """Command-line interface for headless spike detection"""
    parser = argparse.ArgumentParser(
        description='Headless spike detection pipeline for EEG data'
    )
    parser.add_argument(
        'eeg_file',
        type=str,
        help='Path to EEG data file or directory'
    )
    parser.add_argument(
        '--channel',
        type=str,
        default='4 RP',
        help='Channel to analyze (default: 4 RP)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results (default: from config file)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--processing-steps',
        nargs='+',
        default=['Meiling Denoise', 'Meiling Detrend'],
        help='Processing steps (default: Meiling Denoise, Meiling Detrend)'
    )

    args = parser.parse_args()

    # Run pipeline
    pipeline = HeadlessSpikeDetection(config_path=args.config)

    try:
        results = pipeline.run_detection(
            eeg_file=args.eeg_file,
            channel=args.channel,
            output_dir=args.output_dir,
            processing_steps=args.processing_steps
        )

        print(f"\n{'='*50}")
        print("SPIKE DETECTION COMPLETE")
        print(f"{'='*50}")
        print(f"Detected spikes: {results['n_spikes']}")
        print(f"Output directory: {results['output_dir']}")
        print(f"{'='*50}\n")

        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, List
import traceback

from alignment import SignalAlignment
from config import ConfigManager
from dataLoad import DataManager
from DataPreprocessing import PreprocessingPipeline
from SpikeDetection import SpikeDetector, SpikeDetectionParams
from visualization import InteractivePlotter
from gui import SpikeDetectionGUI
from logger import setup_logging

logger = logging.getLogger(__name__)

class SpikeDetectionApplication:
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        
        setup_logging(
            level=self.config.log_level,
            format_string=self.config.log_format
        )
        
        self.data_manager = DataManager()
        self.preprocessing_pipeline = None
        self.spike_detector = None
        self.plotter = None
        
        logger.info("Spike Detection Application initialized")
    
    def initialize_components(self):
        
        try:
            if not self.config_manager.validate_config():
                raise ValueError("Config is invalid")
            
            preprocessing_config = {
                'eeg': {
                    'sample_rate': self.config.preprocessing.eeg_sampling_freq,
                    'notch_frequency': self.config.preprocessing.notch_frequency,
                    'notch_quality': self.config.preprocessing.notch_quality,
                    'bandpass_low': self.config.preprocessing.bandpass_low,
                    'bandpass_high': self.config.preprocessing.bandpass_high,
                    'artifact_threshold': self.config.preprocessing.interpolation_threshold
                },
                'photometry': {
                    'sample_rate': self.config.detection.fs,
                    'lowpass_cutoff': self.config.preprocessing.lowpass_cutoff,
                    'gaussian_sigma': self.config.preprocessing.gaussian_sigma,
                    'savgol_window': self.config.preprocessing.savgol_window,
                    'savgol_polyorder': self.config.preprocessing.savgol_polyorder
                },
                'baseline_start_time': self.config.detection.baseline_start_time,
                'baseline_end_time': self.config.detection.baseline_end_time
            }
            
            self.preprocessing_pipeline = PreprocessingPipeline(preprocessing_config)
            
            detection_params = SpikeDetectionParams(
                fs=self.config.detection.fs,
                tmul=self.config.detection.tmul,
                absthresh=self.config.detection.absthresh,
                sur_time=self.config.detection.sur_time,
                close_to_edge=self.config.detection.close_to_edge,
                too_high_abs=self.config.detection.too_high_abs,
                spkdur_min=self.config.detection.spkdur_min,
                spkdur_max=self.config.detection.spkdur_max,
                channel=self.config.default_channel,
                baseline_end_time=self.config.detection.baseline_end_time,
                baseline_start_time=self.config.detection.baseline_start_time
            )
            
            self.spike_detector = SpikeDetector(detection_params)
            
            self.plotter = InteractivePlotter(self.config.visualization)
            
            logger.info("All components are set up and ready to use")
        except Exception as e:
            logger.error(f"Failed to set up the components and recieved back: {e}")
            raise
        
    def run_cli(self, args):
        
        try:
            self.initialize_components()
            
            logger.info(f"Loading EEG data")
            eeg_data, eeg_record = self.data_manager.load_eeg_data(
                args.eeg_file,
                channel=args.channel or self.config.default_channel
            )
            
            photometry_record = None
            if args.photometry_file:
                logger.info(f"Loading photometry data")
                photometry_record = self.data_manager.load_photometry_data(args.photometry_file)
                
            aligned_data = {}
            if photometry_record:
                alignment = SignalAlignment()
                aligned_data = alignment.align_signals(eeg_record, photometry_record)
                eeg_data = aligned_data['eeg']
                logger.info("Signals aligned successfully")
            
            logger.info("Preprocessing EEG data")
            eeg_result = self.preprocessing_pipeline.eeg_preprocessor(eeg_data)
            
            photometry_result = None
            if photometry_record:
                logger.info(f"Preprocessing Photometry data")
                photometry_result = self.preprocessing_pipeline.process_photometry(
                    aligned_data['gcamp'],
                    aligned_data['isos'],
                    strategy=args.photometry_strategy,
                    time_vector=aligned_data['time'],
                    denoise_first=args.denoise_first
                )
            
            if args.baseline_end > 0:
                from SpikeDetection import BaselineNormalizer
                normalizer = BaselineNormalizer()
                eeg_normalized = normalizer.baseline_zscore(
                    eeg_result.data,
                    args.baseline_end,
                    args.baseline_start,
                    self.config.detection.fs
                )
                eeg_result.data = eeg_normalized.data
                
            logger.info("Detecting spikes")
            spikes = self.spike_detector.detect_spikes(
                eeg_result.data,
                apply_normalization=(args.baseline_end == 0)
            )
            
            summary = self.spike_detector.get_detection_summary(spikes)
            logger.info(f"Spike detection completed: {summary}")
            
            self._print_results(spikes, summary, args)
            
            if args.plot:
                self._generate_plots(
                    eeg_result,
                    photometry_result,
                    spikes,
                    aligned_data,
                    args
                )

            if args.output:
                self._save_results(spikes, summary, eeg_result, photometry_result, args)
                
            return 0
        
        except Exception as e:
            logger.error(f"Total process error: {e}")
            if args.debug:
                logger.error(traceback.format_exc())
            return 1
        
    def run_gui(self):
        
        try:
            from gui import create_and_run_gui
            
            return create_and_run_gui(self.config_manager.config_path)
        
        except Exception as e:
            logger.error(f"GUI error: {e}")
            return 1
        
    def _generate_plots(self, eeg_result, photometry_result, spikes, aligned_data, args):
        
        try:
            logger.info("Generating plots")
            
            seizure_onset = None
            if args.eeg_file:
                filename = Path(args.eeg_file).name
                seizure_onset = self.config_manager.get_seizure_onset(filename)
                
            fig = self.plotter.create_comprehensive_plot(
                eeg_data=eeg_result.data,
                photometry_data=photometry_result.data,
                spikes=spikes,
                time_vector=aligned_data.get('time', None),
                seizure_onset=seizure_onset,
                title=f'Analysis: {Path(args.eeg_file).name}'
            )
            
            fig.show()
            logger.info("Plots generated successfully")
        
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
            
    def _save_results(self, spikes, summary, eeg_result, photometry_result, args):
        
        try: 
            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)
            
            import json
            import numpy as np
            
            spikes_data = []
            for spike in spikes:
                spikes_data.append({
                    'time_samples': int(spike.time_samples),
                    'time_seconds': float(spike.time_seconds),
                    'amplitude': float(spike.amplitude),
                    'width_ms': float(spike.width_ms),
                    'prominence': float(spike.prominence),
                    'channel': int(spike.channel)
                })
                
            results = {
                'summary': summary,
                'spikes': spikes_data,
                'configuration': {
                    'detection_params': self.spike_detector.params.__dict__,
                    'processing_strategy': args.photometry_strategy if hasattr(args, 'photometry_strategy') else 'none'
                }
            }
            
            results_file = output_path / f'spike_detection_results_{Path(args.eeg_file).stem}.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            #TODO: add file that contains plots
            signals_file = output_path / f"processed_signal_{Path(args.eeg_file).stem}.npz"
            
        
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            
def create_argument_parser():
    parser = argparse.ArgumentParser(
        description='Spike Detection Analysis Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to config file'
    )
    
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--gui',
        action='store_true',
        help='Lauch GUI'
    )
    mode_group.add_argument(
        '--cli',
        action='store_true',
        default=True,
        help='Run command-line interface (default)'
    )
    
    parser.add_argument(
        '--eeg-file',
        type=str,
        required=False,
        help='Path to EEG data file or directory'
    )
    
    parser.add_argument(
        '--channel',
        type=int,
        help='EEG channel used for analysis'
    )
    parser.add_argument(
        '--photometry-strategy',
        choices=['meiling', 'chandni'],
        default='meiling',
        help='Photometry preprocessing strategy'
    )
    parser.add_argument(
        '--denoise-first',
        action='store_true',
        default=True,
        help="Apply denoising before detrending for Meiling's strategy"
    )
    parser.add_argument(
        '--baseline-end',
        type=float,
        default=0.0,
        help='End time for baseline period (seconds, 0 = no baseline normalization)'
    )
    parser.add_argument(
        '--baseline-start',
        type = float,
        default=0.0,
        help='Start time for baseline period'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output directory for results'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generatoe visualization plots'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging and error traceback'
    )
    
    return parser
    
def main():
        
    parser = create_argument_parser()
    args = parser.parse_args()
    
    if len(sys.argv) == 1:
        args.gui = True
        
    try:
        
        app = SpikeDetectionApplication(config_path=args.config)
        
        if args.debug:
            import logging
            logging.getLogger().setLevel(logging.DEBUG)
            
        if args.gui:
            return app.run_gui()
        else:
            if not args.eeg_file:
                parser.error("--eeg-file is required for CLI mode")
            return app.run_cli(args)
    except Exception as e:
        print(f"Fatal error: {e}")
        if args.debug if 'args' in locals() else False:
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
            

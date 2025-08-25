import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class DataPaths:
    
    eeg_data_dir: str = "./EEGData"
    photometry_data_dir: str = "./FiberPhotometryData"
    output_dir: str = "./output"
    seizure_onsets_file: str = "./config/seizure_onsets.yaml"
    
    def __post_init__(self):
        for path_name in ['eeg_data_dir', 'photometry_data_dir', 'output_dir']:
            path = Path(getattr(self, path_name))
            path.mkdir(parents=True, exist_ok=True)
            
@dataclass
class PreprocessingConfig:
    # EEG preprocessing
    eeg_sampling_freq: float = 1000.0
    notch_frequency: float = 60.0
    notch_quality: float = 30.0
    bandpass_low: float = 1.0
    bandpass_high: float = 70.0
    interpolation_threshold: float = 3.0
    
    # Photometry preprocessing
    savgol_window: int = 301
    savgol_polyorder: int = 4
    gaussian_sigma: float = 75.0
    lowpass_cutoff: float = 10.0
    
    # Meiling-specific parameters
    exp_fit_max_iterations: int = 1000
    exp_fit_bounds_multiplier: float = 2.0
    
@dataclass
class DetectionConfig:
    fs: float = 1000.0
    tmul: float = 3.0
    absthresh: float = 0.4
    sur_time: float = 1000.0
    close_to_edge: float = 0.1
    too_high_abs: float = 1e9
    spkdur_min: float = 70.0
    spkdur_max: float = 200.0
    baseline_end_time: float = 0.0
    baseline_start_time: float = 0.0
    use_adaptive_detection: bool = False
    multi_channel_requirements: bool = False
    
@dataclass
class VisualizationConfig:
    default_renderer: str = 'browser'
    max_plot_points: int = 10000
    downsample_factor: int = 10
    figure_width: int = 1200
    figure_height: int = 800
    line_width: float = 1.0
    spike_marker_size: int = 10
    spike_marker_color: str = 'red'
    
@dataclass
class GuiConfig:
    data_paths: DataPaths
    preprocessing: PreprocessingConfig
    detection: DetectionConfig
    visualization: VisualizationConfig
    
    # Logging configuration
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Application settings
    default_channel: int = 2
    processing_strategy: str = "meiling"  # "meiling" or "chandni"
    
class ConfigManager:
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = Path(config_path) if config_path else Path("config/default_config.yaml")
        self._config: Optional[GuiConfig] = None
        self._seizure_onsets: Optional[Dict[str, int]] = None
        
    @property
    def config(self) -> GuiConfig:
        
        if self._config is None:
            self._config = self.load_config()
        return self._config
    
    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> GuiConfig:
        
        if config_path:
            self.config_path = Path(config_path)
            
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
            else:
                logger.warning(f"Config file not found from {self.config_path}, using defaults")
                config_data = {}
            
            return self._create_config_from_dict(config_data)
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.info("Using default config")
            return self._create_default_config()
        
    def save_config(self, config_path: Optional[Union[str, Path]] = None) -> None:
        
        if config_path:
            self.config_path = Path(config_path)
            
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            config_dict = asdict(self.config)
            with open(self.config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            logger.info(f"Config saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            raise
    def _create_config_from_dict(self, config_data: Dict[str, Any]) -> GuiConfig:
        
        default_config = self._create_default_config()
        
        if 'data_paths' in config_data:
            for key, value in config_data['data_paths'].items():
                if hasattr(default_config.data_paths, key):
                    setattr(default_config.data_paths, key, value)
        
        if 'preprocessing' in config_data:
            for key, value in config_data['preprocessing'].items():
                if hasattr(default_config.preprocessing, key):
                    setattr(default_config.preprocessing, key, value)
        
        if 'detection' in config_data:
            for key, value in config_data['detection'].items():
                if hasattr(default_config.detection, key):
                    setattr(default_config.detection, key, value)
        
        if 'visualization' in config_data:
            for key, value in config_data['visualization'].items():
                if hasattr(default_config.visualization, key):
                    setattr(default_config.visualization, key, value)
        
        for key in ['log_level', 'log_format', 'default_channel', 'processing_strategy']:
            if key in config_data:
                setattr(default_config, key, config_data[key])
                
        return default_config
    
    def _create_default_config(self) -> GuiConfig:
        
        return GuiConfig(
            data_paths=DataPaths(),
            preprocessing=PreprocessingConfig(),
            detection=DetectionConfig(),
            visualization=VisualizationConfig()
        )
        
    def load_seizure_onsets(self) -> Dict[str, int]:
        
        if self._seizure_onsets is not None:
            return self._seizure_onsets
        
        onset_file = Path(self.config.data_paths.seizure_onsets_file)
        
        try:
            if onset_file.exists():
                with open(onset_file, 'r') as f:
                    if onset_file.suffix.lower() == '.json':
                        self._seizure_onsets = json.load(f)
                    else:
                        self._seizure_onsets = yaml.safe_load(f)
                logger.info(f"Loaded seizure onsets from {onset_file}")
            else:
                logger.warning(f"Seizure onset file not found")
                self._seizure_onsets = {}
        except Exception as e:
            logger.error(f"Error loading seizure onset file: {e}")
            self._seizure_onsets = {}
        
        return self._seizure_onsets
    
    def get_seizure_onset(self, filename: str) -> Optional[int]:
        onsets = self.load_seizure_onsets()
        return onsets.get(filename)   
    
    def update_detection_params(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self.config.detection, key):
                setattr(self.config.detection, key, value)
            else:
                logger.warning(f"Unknown detection key: {key}")
                
    def validate_config(self) -> bool:
        
        try:
            data_paths = self.config.data_paths
            if not Path(data_paths.eeg_data_dir).exists():
                logger.error(f"EEG data directory does not exist: {data_paths.eeg_data_dir}")
                return False
            
            if not Path(data_paths.photometry_data_dir).exists():
                logger.error(f"Photometry data directory does not exist: {data_paths.photometry_data_dir}")
                return False

            detection = self.config.detection
            if detection.fs <= 0:
                logger.error("Sampling frequency must be positive")
                return False
            
            if detection.spkdur_min >= detection.spkdur_max:
                logger.error("Minimum spike duration must be less than maximum")
                return False
            
            if not 0 <= detection.close_to_edge <= 1:
                logger.error("close_to_edge must be between 0 and 1")
                return False
            
            preprocessing = self.config.preprocessing
            if preprocessing.eeg_sampling_freq <= 0:
                logger.error("EEG sampling frequency must be positive")
                return False
            
            if preprocessing.bandpass_low >= preprocessing.bandpass_high:
                logger.error("Bandpass low frequency must be less than high frequency")
                return False
            
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error validating configuration: {e}")
            return False     
        
        
if __name__ == "__main__":
    
    config_manager = ConfigManager()
    if config_manager.validate_config():
        print("Config is valid")
        print(f'EEG data directory: {config_manager.config.data_paths.eeg_data_dir}')
    else:
        print("Configuration validation failed")
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np
import logging
from open_ephys.analysis import Session

logger = logging.getLogger(__name__)

@dataclass
class DataRecord:
    filename: str
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(filename={self.filename})"
    
@dataclass
class EEGRecord(DataRecord):
    data: np.ndarray
    sample_rate: float
    timestamps: np.ndarray
    sample_numbers: np.ndarray
    events: Dict[str, np.ndarray]
    
    @property
    def n_channels(self) -> int:
        return self.data.shape[0] if self.data.ndim == 2 else 1
    
    @property
    def n_samples(self) -> int:
        return self.data.shape[-1]
    
    @property
    def duration_seconds(self) -> float:
        return self.n_samples / self.sample_rate
    
@dataclass
class PhotometryRecord(DataRecord):
    photo: np.ndarray
    isos: np.ndarray
    sync: np.ndarray
    sample_rate: float
    
    @property
    def n_samples(self) -> int:
        return len(self.photo)
    
    @property
    def duration_seconds(self) -> float:
        return self.n_samples / self.sample_rate
    
class DataLoader(ABC):
    
    @abstractmethod
    def load(self, filepath: Union[str, Path]) -> DataRecord:
        pass
    
    @abstractmethod
    def can_load(self, filepath: Union[str, Path]) -> bool:
        pass
    
class EEGLoader(DataLoader):
    
    def __init__(self, max_samples: Optional[int] = None):
        self.max_samples = max_samples or 1220000
        
    def can_load(self, filepath: Union[str, Path]) -> bool:
        path = Path(filepath)
        if not path.exists():
            return False
        
        try:
            session = Session(str(path))
            return len(session.recordnodes) > 0
        except Exception:
            return False
    
    def load(self, filepath: Union[str, Path]) -> EEGRecord:
        
        try:
            session = Session(str(filepath))
            recording = session.recordnodes[0].recordings[0]
            
            continuous = recording.continuous[0]
            data = continuous.get_samples(0, self.max_samples)
            
            if data.ndim == 1:
                data = data.reshape(1, -1)
            elif data.shape[0] > data.shape[1]:
                data = data.T
                
            
            events = self._extract_events(recording)
            

            
            return EEGRecord(
                filename=Path(filepath).name,
                data=data,
                sample_rate=continuous.metadata['sample_rate'],
                timestamps=continuous.timestamps[:data.shape[1]],
                sample_numbers=continuous.sample_numbers[:data.shape[1]],
                events=events
            )
        except Exception as e:
            logger.error(f"Error loading EEG data: {e}")
            raise
        
    def _extract_events(self, recording) -> Dict[str, np.ndarray]:
        
        events = {
            'sample_number': np.array([]),
            'timestamps': np.array([]),
            'ttl_lines': np.array([]),
            'states': np.array([])
        }    
        
        if hasattr(recording, 'events') and len(recording.events) > 0:
            events_df = recording.events
            
            if not events_df.empty:
                events['sample_number'] = events_df['sample_number'].values
                events['timestamps'] = events_df['timestamp'].values
                
                if 'line' in events_df.columns:
                    events['ttl_lines'] = events_df['line'].values
                if 'state' in events_df.columns:
                    events['states'] = events_df['state'].values
                    
        return events
    
class PPDLoader(DataLoader):
    
    def can_load(self, filepath: Union[str, Path]) -> bool:
        path = Path(filepath)
        return path.suffix.lower() == '.ppd' and path.exists()
    
    def load(self, filepath: Union[str, Path]) -> PhotometryRecord:
        try:
            from data_import import import_ppd
            
            data = import_ppd(str(filepath))
            
            
            return PhotometryRecord(
                filename=Path(filepath).name,
                photo=data['analog_1'],
                isos=data['analog_2'],
                sync=data['digital_1'],
                sample_rate=data['sampling_rate']
            )
            
        except Exception as e:
            logger.error(f"Error loading photometry data: {e}")
            raise
        
class DataManager:
    
    def __init__(self):
        self.loaders = [
            EEGLoader(),
            PPDLoader(),
        ]
        self._cache: Dict[str, DataRecord] = {}
        
    def load_data(self, filepath: Union[str, Path], force_reload: bool = False) -> DataRecord:
        
        filepath_str = str(filepath)
        
        if not force_reload and filepath_str in self._cache:
            logger.info(f"Loading {filepath}")
            return self._cache[filepath_str]
        
        loader = self._find_loader(filepath)
        if loader is None:
            raise ValueError(f"File is neither EEG nor PPD")
        
        logger.info(f"Loading {filepath} using {loader.__class__.__name__}")
        data = loader.load(filepath)
        
        self._cache[filepath_str] = data
        
        return data
    
    def _find_loader(self, filepath: Union[str, Path]) -> Optional[DataLoader]:
        for loader in self.loaders:
            if loader.can_load(filepath):
                return loader
        return None
    
    def load_eeg_data(self,
                      filepath: Union[str, Path],
                      channel: Optional[int] = None) -> Tuple[np.ndarray, EEGRecord]:
        record = self.load_data(filepath)
        
        if not isinstance(record, EEGRecord):
            raise ValueError(f"Expected EEG data, got photometry")
        
        if channel is not None:
            if record.data.ndim == 1:
                if channel != 0:
                    raise ValueError(f"Channel {channel} is not available in EEG data")
                channel_data = record.data
            else:
                if channel >= record.n_channels:
                    raise ValueError(f"Channel is not available data has {record.n_channels} channels")
                channel_data = record.data[channel, :]
        else:
            channel_data = record.data
        
        return channel_data, record
    
    def load_photometry_data(self, filepath: Union[str, Path]) -> PhotometryRecord:
        record = self.load_data(filepath)
        
        if not isinstance(record, PhotometryRecord):
            raise ValueError(f"Expected photometry data got EEG data")
        
        return record
    
    def clear_cache(self) -> None:
        self._cache.clear()
        logger.info("Data cache cleared")
        
    def get_cache_info(self) -> Dict[str, Any]:
        return {
            'cached_files': list(self._cache.keys()),
            'cache_size': len(self._cache)
        }

if __name__ == "__main__":
    data_manager = DataManager()
    
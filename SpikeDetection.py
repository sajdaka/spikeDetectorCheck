from typing import List, Protocol
from dataclasses import dataclass
import numpy as np
from scipy.signal import find_peaks
import logging

logger = logging.getLogger(__name__)

@dataclass
class SpikeDetectionParams:
    fs: float = 1000.0
    tmul: float = 3.0
    absthresh: float = 0.4
    sur_time: float = 10000.0
    close_to_edge: float = 0.1
    too_high_abs: float = 1e9
    spkdur_min: float = 70.0
    spkdur_max: float = 200.0
    channel: int = 9
    baseline_end_time: float = 0.0
    
    def __post_init__(self):
        if self.fs <= 0:
            raise ValueError("Sampling frequency must be a positive value")
        if self.spkdur_min >= self.spkdur_max:
            raise ValueError("Minimum spike duration must be less than it's associated maximum")
        if not 0 <= self.close_to_edge <= 1:
            raise ValueError("close_to_edge must be a value between 0 and 1")
        
@dataclass
class SpikeEvent:
    time_samples: int
    time_seconds: float
    amplitude: float
    width_samples: float
    width_ms: float
    prominence: float
    channel: int = 9
    
    def __str__(self) -> str:
        return (f"Spike detected at {self.time_seconds:.3f}s: "
                f"amplitude={self.amplitude:.3f}, width={self.width_ms:.1f}ms")
        
class SignalProcessor(Protocol):
    
    def process(self, signal: np.ndarray, params: SpikeDetectionParams) -> np.ndarray:
        if self.Protocol == "meiling":
            #branch to meiling's preprocessing
            return
        else:
            #branch to chandni's preprocessing
            return

class BaselineNormalizer:
    
    @staticmethod
    def baseline_zscore(data: np.ndarray, end_baseline: float, fs: float) -> np.ndarray:
        
        end_time = int(end_baseline * fs)
        if end_time<= 0 or end_time >= len(data):
            logger.warning("Invalid baseline period: using full data set for normalization")
            baseline = data
        else:
            baseline = data[:end_time]
            
        baseline_mean = np.nanmean(baseline)
        baseline_std = np.nanstd(baseline)
        
        if baseline_std == 0:
            logger.warning("Baseline standard deviation was found to be zero, returning zeros")
            return np.zeros_like(data)
        
        return (data - baseline_mean) /baseline_std
    
    @staticmethod
    def full_zscore(data: np.ndarray) -> np.ndarray:
        data_mean = np.nanmean(data)
        data_std = np.nanstd(data)
        
        if data_std == 0:
            logger.warning("Data's standard deviation is zero, returning zeroes")
            return np.zeros_like(data)

        return (data - data_mean) / data_std
    
class SpikeDetector:
    
    def __init__(self, params: SpikeDetectionParams):
        self.params = params
        self.normalizer = BaselineNormalizer()
        logger.info(f"Initialized the SpikeDetector with the given parameters: {params}")
        
    def detect_spikes(self, signal: np.ndarray, apply_normalization: bool = True) -> List[SpikeEvent]:
        
        try:
            signal = self._validate_input(signal)
            
            if apply_normalization and self.params.baseline_end_time > 0:
                signal = self.normalizer.baseline_zscore(signal, self.params.baseline_end_time, self.params.fs)
                
                spikes = self._detect_spikes_core(signal)
                
                filtered_spikes = self._filter_spikes(spikes, signal)
                
                logger.info(f"Detected {len(filtered_spikes)} spikes from the data")
                return filtered_spikes
        
        except Exception as e:
            logger.error(f"Error in spike detection: {e}")
            raise
    
    def _validate_input(self, signal: np.ndarray) -> np.ndarray:
        
        if signal.ndim > 2:
            if signal.shape[0] == 1:
                signal = signal[0, :]
            elif self.params.channel < signal.shape[0]:
                signal = signal[self.params.channel, :]
            else:
                raise ValueError(f"Channel {self.params.channel} not available")
            
        if len(signal) == 0:
            raise ValueError("Input data is empty")
        
        if np.all(np.isnan(signal)):
            raise ValueError("Input data is full of NaN values")
        
        return signal
    
    def _detect_spikes_core(self, signal: np.ndarray) -> List[SpikeEvent]:
        
        hpdata = signal - np.nanmean(signal)
    
        
        lthresh = np.median(np.abs(hpdata))
        thresh = lthresh * self.params.tmul
        effective_thresh = max(thresh, self.params.absthresh)
        
        
        spkdur_samples = (
            self.params.spkdur_min * self.params.fs / 1000,
            self.params.spkdur_max * self.params.fs /1000
        )
        
        all_spikes = []
        
        
        
        # Detect both positive and negative spikes
        for polarity in ['positive', 'negative']:
            if polarity == 'negative':
                signal_data = -hpdata  # Invert for negative spike detection
            else:
                signal_data = hpdata
            
            
            peaks, peak_properties = find_peaks(
                signal_data,
                height=effective_thresh,          
                distance=int(spkdur_samples[0]),  
                width=(spkdur_samples[0]/4, spkdur_samples[1]/2),  
                prominence=self.params.absthresh/2,   
                rel_height=0.5              
            )
            
            for i, peak_idx in enumerate(peaks):
                spike = SpikeEvent(
                    time_samples = peak_idx,
                    time_seconds = peak_idx / self.params.fs,
                    amplitude=peak_properties['peak_heights'][i],
                    width_samples=peak_properties['widths'][i],
                    width_ms=peak_properties['widths'][i] * 1000 / self.params.fs,
                    prominence=peak_properties['prominences'][i],
                    channel=self.params.channel
                )
                all_spikes.append(spike)
                
        return all_spikes
    
    def _filter_spikes(self, spikes: List[SpikeEvent], signal: np.ndarray) -> List[SpikeEvent]:
        
        if not spikes:
            return []
        
        filtered = []
        close_samples = int(self.params.close_to_edge * self.params.fs)
        
        for spike in spikes:
            
            if (spike.time_samples < close_samples or spike.time_samples > len(signal) - close_samples):
                continue
            
            if spike.amplitude > self.params.too_high_abs:
                continue
            
            if not (self.params.spkdur_min <= spike.width_ms <= self.params.spkdur_max):
                continue
            
            filtered.append(spike)
            
        filtered = self._remove_duplicates(filtered)
            
        return filtered
    
    def _remove_duplicates(self, spikes: List[SpikeEvent]) -> List[SpikeEvent]:
        
        if len(spikes) <= 1:
            return spikes
        
        spikes.sort(key=lambda x: x.time_samples)
        
        filtered = []
        min_seperation = int(100e-3 * self.params.fs)
        
        for spike in spikes:
            
            too_close = False
            for accepted_spike in filtered:
                if abs(spike.time_samples - accepted_spike.time_samples) < min_seperation:
                    if spike.prominence > accepted_spike.prominence:
                        filtered.remove(accepted_spike)
                        break
                    else:
                        too_close = True
                        break
                    
            if not too_close:
                filtered.append(spike)
                
        return filtered
    
    def get_detection_summary(self, spikes: List[SpikeEvent]) -> Dict:
        
        if not spikes:
            return {
                'n_spikes': 0,
                'spike_rate': 0.0,
                'mean_amplitude': 0.0,
                'mean_width': 0.0
            }
            
        amplitudes = [s.amplitude for s in spikes]
        widths = [s.width_ms for s in spikes]
        
        duration_seconds = max(s.time_seconds for s in spikes)
        
        return {
            'n_spikes': len(spikes),
            'spike_rate': len(spikes)/ duration_seconds if duration_seconds > 0 else 0,
            'mean_amplitude': np.mean(amplitudes),
            'std_amplitudes': np.std(amplitudes),
            'mean_width': np.mean(widths),
            'std_width': np.std(widths),
            'amplitude_range': (min(amplitudes), max(amplitudes)),
            'width_range': (min(widths), max(widths))
        }
        
            
            
        

from typing import List, Protocol, Dict, Any
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
    baseline_start_time: float = 0.0
    
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
        


class BaselineNormalizer:
    
    @staticmethod
    def baseline_zscore(data: np.ndarray, end_baseline: float, start_baseline: float, fs: float) -> np.ndarray:
        
        end_time = int(end_baseline * fs)
        start_time = int(start_baseline * fs)
        if end_time<= 0 or end_time >= len(data) or start_time >= end_time:
            logger.warning("Invalid baseline period: using full data set for normalization")
            baseline = data
        else:
            baseline = data[start_time:end_time]
            
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
        
    def detect_spikes(self, signal: np.ndarray) -> List[SpikeEvent]:
        
        try:
            signal = self._validate_input(signal)
            
                
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
        
        baseline_start = self.params.baseline_start_time
        baseline_end = self.params.baseline_end_time
        segment_duration = 5 * 60 * self.params.fs
        segments_num = 6
        
        baseline_period = signal[int(baseline_start * self.params.fs) : int(baseline_end * self.params.fs)]
        segments = []
        for i in range(segments_num):
            segment_start = i * segment_duration
            segment_end = (i + 1) * segment_duration
            segment = baseline_period[segment_start: segment_end]
            segments.append(np.std(segment))
            
        segments_std = np.array(segments)
        
        mean_std = np.mean(segments_std)
        std_of_std = np.std(segments_std)
        cv = (std_of_std / mean_std) *100
        min_cv = cv
        min_baseline_start = baseline_start
        min_baseline_end = baseline_end
        
        while(cv > 20):
            if(baseline_end + 5 * 60 > 3600):
                cv = min_cv
                baseline_start = min_baseline_start
                baseline_end = min_baseline_end
                break
            if min_cv > cv:
                min_baseline_start = baseline_start
                min_baseline_end = baseline_end
                min_cv = cv
            segments.pop(0)
            baseline_start += 5 * 60 
            baseline_end += 5 * 60
            new_segment_start = int((baseline_start + (5 * 60))* self.params.fs) 
            new_segment_end = int(baseline_end * self.params.fs)
            added_segment = signal[new_segment_start: new_segment_end]
            segments.append(np.std(added_segment))
            segments_std = np.array(segments)
            mean_std = np.mean(segments_std)
            std_of_std = np.std(segments_std)
            cv = (std_of_std / mean_std) * 100
        
        signal_zscored = BaselineNormalizer.baseline_zscore(
           signal,
           self.params.baseline_end_time,
           self.params.baseline_start_time,
           self.params.fs
        )
       
        baseline_start_idx = int(self.params.baseline_start_time * self.params.fs)
        baseline_end_idx = int(self.params.baseline_end_time * self.params.fs)
        baseline_segment = signal_zscored[baseline_start_idx:baseline_end_idx]
       
        baseline_std = np.std(baseline_segment)
        
        #lthresh = signal.mean()
        thresh = baseline_std * self.params.tmul
        effective_thresh = max(thresh, self.params.absthresh)
        logger.info(f"Baseline std: {baseline_std:.3f} (should be ~1.0)")
        logger.info(f"Threshold: {effective_thresh:.2f} SD: (tmul={self.params.tmul})")
        
        
        spkdur_samples = (
            self.params.spkdur_min * self.params.fs / 1000,
            self.params.spkdur_max * self.params.fs /1000
        )
        
        all_spikes = []
        
        
        
        # Detect both positive and negative spikes
        for polarity in ['positive', 'negative']:
            if polarity == 'negative':
                signal_data = -signal_zscored  # Invert for negative spike detection
            else:
                signal_data = signal_zscored
            
            
            peaks, peak_properties = find_peaks(
                signal_data,
                height=effective_thresh,          
                distance=int(spkdur_samples[0]),  
                width=(spkdur_samples[0], spkdur_samples[1]),  
                prominence=effective_thresh*0.5,   
                #rel_height=0.5              
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
        seizureTimes = [(0,0), (0,0)]
        
        for spike in spikes:
            
            if (spike.time_samples < close_samples or spike.time_samples > len(signal) - close_samples):
                continue
            
            if spike.amplitude > self.params.too_high_abs:
                continue
            
            inSeizure = False
            for seizure in seizureTimes:
                if spike.time_seconds > seizure[0] and spike.time_seconds < seizure[1]:
                    inSeizure = True
            if inSeizure:
                continue
            
            #if not (self.params.spkdur_min <= spike.width_ms <= self.params.spkdur_max):
             #   continue
            
            filtered.append(spike)
            
        filtered = self._remove_duplicates(filtered)
            
        return filtered
    
    def _remove_duplicates(self, spikes: List[SpikeEvent]) -> List[SpikeEvent]:
        """
        Remove duplicate spikes that are too close together.
        Optimized O(n) version - only checks last accepted spike since spikes are sorted.
        """
        if len(spikes) <= 1:
            return spikes

        # Sort by time
        spikes.sort(key=lambda x: x.time_samples)

        filtered = []
        min_seperation = int(10e-3 * self.params.fs)

        for spike in spikes:
            # Only check the last accepted spike (spikes are sorted!)
            if filtered:
                last_spike = filtered[-1]
                distance = spike.time_samples - last_spike.time_samples

                if distance < min_seperation:
                    # Too close - keep the one with higher prominence
                    if spike.prominence > last_spike.prominence:
                        filtered[-1] = spike  # Replace last with current
                    # else: skip current spike (last is better)
                else:
                    # Far enough apart - accept this spike
                    filtered.append(spike)
            else:
                # First spike - always accept
                filtered.append(spike)

        return filtered
    
    def get_detection_summary(self, spikes: List[SpikeEvent]) -> Dict[str, Any]:
        
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
            'std_amplitude': np.std(amplitudes),
            'mean_width': np.mean(widths),
            'std_width': np.std(widths),
            'amplitude_range': (min(amplitudes), max(amplitudes)),
            'width_range': (min(widths), max(widths))
        }
        
            
            
        

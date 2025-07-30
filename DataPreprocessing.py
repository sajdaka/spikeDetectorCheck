from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, Protocol, Union
import numpy as np
from scipy import signal, stats
from scipy import interpolate
from scipy.signal import butter, filtfilt, savgol_filter, iirnotch
from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy.interpolate import interp1d
import logging

logger = logging.getLogger(__name__)

@dataclass
class PreprocessingResult:
    data: np.ndarray
    metadata: Dict[str, Any]
    preprocessing_steps: list[str]
    
class PreprocessingStrategy(Protocol):
    
    def process(self, data: np.ndarray, **kwargs) -> PreprocessingResult:
       pass
  
 
class BasePreprocessor(ABC):
    
    def __init__(self, name: str):
        self.name = name
        
    @abstractmethod
    def process(self, data: np.ndarray, **kwargs) -> PreprocessingResult:
        pass
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

class EEGPreprocessor(BasePreprocessor):
    
    def __init__(self,
                 sample_rate: float = 1000.0,
                 notch_freq: float = 60.0,
                 notch_quality: float = 30.0,
                 bandpass_low: float = 1.0,
                 bandpass_high: float = 70.0,
                 artifact_threshold: float = 3.0):
        super().__init__("EEG_Preprocessor")
        self.sample_rate = sample_rate
        self.notch_freq = notch_freq
        self.notch_quality = notch_quality
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.artifact_threshold = artifact_threshold
        
    def process(self, data: np.ndarray, **kwargs) -> PreprocessingResult:
        
        result = PreprocessingResult(
            data=data.copy(),
            metadata={'original_shape': data.shape},
        )
        
        try:
            result.data = self._apply_notch_filter(result.data)
            
            if self.artifact_threshold > 0:
                result.data, interpolated_regions = self._interpolate_artifacts(
                    result.data, self.artifact_threshold
                )
                result.metadata['interpolated_regions'] = interpolated_regions
                
            result.data = self._apply_bandpass_filter(result.data)
            result.metadata['final_shape'] = result.data.shape
            logger.info(f"EEG preprocessing completed")
            
            return result
        except Exception as e:
            logger.error(f"Error in EEG preprocessing: {e}")
            raise
        
    def _apply_notch_filter(self, data: np.ndarray) -> np.ndarray:
        b,a = iirnotch(self.notch_freq, Q=self.notch_quality, fs=self.sample_rate)
        return filtfilt(b,a, data)
    
    def _apply_bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        nyquist = 0.5 * self.sample_rate
        low = self.bandpass_low / nyquist
        high = self.bandpass_high / nyquist
        
        if high >= 1.0:
            high = 0.99
        
        b, a = butter(4, [low, high], btype='band')
        return filtfilt(b, a, data)
    
    def _interpolate_artifacts(self, data: np.ndarray, threshold: float) -> Tuple[np.ndarray, list]:
        cleaned_signal = data.copy()
        artifact_mask = np.abs(data) > threshold
        artifact_indices = np.where(artifact_mask)
        
        if len(artifact_indices) == 0:
            return data, []
        
        artifact_regions = self._find_artifact_regions(artifact_indices)
        interpolated_regions = []
        
        for start, end in artifact_regions:
            try:
                interpolated_start = max(0, start - 10)
                interpolated_end = min(len(data) -1, end + 10)
                
                before_start = max(0, interpolated_start - 20)
                before_end = interpolated_start - 1
                after_start = interpolated_end + 1
                after_end = min(len(data) - 1, interpolated_end + 20)
                
                if before_end >= before_start and after_end >= after_start:
                    time_before = np.arange(before_start, before_end + 1)
                    time_after = np.arange(after_start, after_end + 1)
                    x_good = np.concatenate([time_before, time_after])
                    y_good = np.concatenate([data[time_before], data[time_after]])
                    
                    if len(x_good) > 1:
                        interpolator = interp1d(x_good, y_good, kind='linear', fill_value='extrapolate')
                        artifact_span = np.arange(interpolated_start, interpolated_end + 1)
                        cleaned_signal[artifact_span] = interpolator(artifact_span)
                        interpolated_regions.append((interpolated_start, interpolated_end))
                        
            except Exception as e:
                logger.warning(f"Failed to interpolate artifact at {start}-{end}: {e}")
                
        return cleaned_signal, interpolated_regions
    
    def _find_artifact_regions(self, artifact_indices: np.ndarray) -> list:
        
        if len(artifact_indices) == 0:
            return []
        
        regions = []
        start = artifact_indices[0]
        end = artifact_indices[0]
        
        for i in range(1, len(artifact_indices)):
            if artifact_indices[i] == end + 1:
                end = artifact_indices[i]
            else:
                regions.append((start, end))
                start = artifact_indices[i]
                end = artifact_indices[i]
                
        regions.append((start, end))
        return regions
    
class PhotometryPreprocessor(BasePreprocessor):
    
    def __init__(self, name: str, sample_rate: float = 1000.0):
        super().__init__(name)
        self.sample_rate = sample_rate
        
class MeilingPhotometryPreprocessor(PhotometryPreprocessor):
    
    def __init__(self,
                 sample_rate: float = 1000.0,
                 lowpass_cutoff: float = 10.0,
                 savgol_window: int = 301,
                 savgol_polyorder: int = 4):
        super().__init__("Meling_Photometry", sample_rate)
        self.lowpass_cutoff = lowpass_cutoff
        self.savgol_window = savgol_window
        self.savgol_polyorder = savgol_polyorder
        
    def process(self,
                photo_data: np.ndarray,
                isos_data: np.ndarray,
                time_vector: np.ndarray,
                denoise_first: bool = True) -> PreprocessingResult:
        
        result = PreprocessingResult(
            data=np.array([photo_data.copy(), isos_data.copy()]),
            metadata={
                'original_lengths': [len(photo_data), len(isos_data)],
                'processing_order': 'denoise_first' if denoise_first else 'detrend_first'
            }
        )
        
        try:
            photo, isos = result.data[0], result.data[1]
            
            if denoise_first:
                photo, isos = self._denoise_signals(photo, isos)
                photo_detrended, photo_expfit, photo_est_motion, isos_detrended = self._detrend_signals(
                    photo, isos, time_vector
                )
                
            else:
                photo_detrended, photo_expfit, photo_est_motion, isos_detrended = self._detrend_signals(
                    photo, isos, time_vector
                )
                
                photo_detrended, isos_detrended = self._denoise_signals(photo_detrended, isos_detrended)
                
            df_f_precorrected, df_f_corrected, z_df_f = self._calculate_df_f(
                photo_detrended, photo_expfit, photo_est_motion
            )
            
            result.data = df_f_corrected
            result.metadata.update({
                'photo_dtrended': photo_detrended,
                'isos_detrended': isos_detrended,
                'photo_expfit': photo_expfit,
                'photo_est_motion': photo_est_motion,
                'df_f_precorrected': df_f_precorrected,
                'z_df_f': z_df_f,
                'final_shape': df_f_corrected.shape
            })
            
            logger.info(f"Meiling photometry preprocessing completed")
            return result
        
        except Exception as e:
            logger.error(f"Error in Meiling's photometry preprocessing: {e}")
            raise
        
    def _denoise_signals(self,
                         photo: np.ndarray,
                         isos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        b, a = butter(4, self.lowpass_cutoff, btype='low', fs=self.sample_rate)
        photo_denoised = filtfilt(b, a, photo)
        isos_denoised = filtfilt(b, a, isos)
        
        return photo_denoised, isos_denoised
    
    def _detrend_signals(self,
                         photo: np.ndarray,
                         isos: np.ndarray,
                         time_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        photo_params, photo_expfit = self._fit_double_exponential(photo, time_vector)
        isos_params, isos_expfit = self._fit_double_exponential(isos, time_vector)
        
        photo_detrended = photo - photo_expfit
        isos_detrended = isos - isos_expfit
        
        slope, intercept, rvalue, pvalue, stderr = linregress(x=isos_detrended, y=photo_detrended)
        
        photo_est_motion = slope * isos_detrended + intercept
        
        return photo_detrended, photo_expfit, photo_est_motion, isos_detrended
    
    def _fit_double_exponential(self,
                                signal: np.ndarray,
                                time_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        max_sig = np.max(signal)
        initial_params = [max_sig/2, max_sig/4, max_sig/4, 3600, 0.1]
        bounds = ([0, 0, 0, 600, 0], [max_sig, max_sig, max_sig, 36000, 1])
        
        try:
            params, _ = curve_fit(
                self._fit_double_exponential,
                time_vector,
                signal,
                p0=initial_params,
                bounds=bounds,
                maxfev=1000
            )
            expfit = self._double_exponential(time_vector, *params)
            return params, expfit
        except Exception as e:
            logger.warning(f"Double exponential fit failed: {e}, using linear fit")
            poly_coeffs = np.polyfit(time_vector, signal, 1)
            expfit = np.polyval(poly_coeffs, time_vector)
            return poly_coeffs, expfit
        
    def _double_exponential(self, t, const, amp_fast, amp_slow, tau_slow, tau_muliplier):
        
        tau_fast = tau_slow * tau_muliplier
        return const+ amp_slow * np.exp(-t/tau_slow) + amp_fast * np.exp(-t/tau_fast)
    
    def _calculate_df_f(self,
                        photo_cleaned: np.ndarray,
                        photo_expfit: np.ndarray,
                        photo_est_motion:np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        photo_corrected = photo_cleaned - photo_est_motion
        
        df_f_precorrected = photo_cleaned / photo_expfit
        df_f_corrected = photo_corrected / photo_expfit
        
        z_df_f = stats.zscore(df_f_corrected)
        
        return df_f_precorrected, df_f_corrected, z_df_f
    
class ChandniPhotometryPreprocessor(PhotometryPreprocessor):
    
    def __init__(self,
                 sample_rate: float = 1000.0,
                 gaussian_sigma: float = 75.0,
                 savgol_window: int = 301,
                 savgol_polyorder: int = 4):
        super().__init__("Chandni_Photometry", sample_rate)
        self.gaussian_sigma = gaussian_sigma
        self.savgol_window = savgol_window
        self.savgol_polyorder = savgol_polyorder
        
    def process(self,
                photo_data: np.ndarray,
                isos_data: np.ndarray) -> PreprocessingResult:
        
        result = PreprocessingResult(
            data=np.array([photo_data.copy(), isos_data.copy()]),
            metadata={
                'original_lengths': [len(photo_data), len(isos_data)],
                'processing_method': 'chandni'
            }
        )        
        
        try:
            photo, isos = result.data[0], result.data[1]
            
            photo_filtered, isos_filtered = self._apply_gaussian_filter(photo, isos)
            
            df_f_preprocessed = (photo - isos) / isos
            df_f_processed = (photo_filtered - isos_filtered) /isos_filtered
            z_df_f = stats.zscore(df_f_processed)
            
            result.data = df_f_processed
            result.metadata.update({
                'photo_filtered': photo_filtered,
                'isos_filtered': isos_filtered,
                'df_f_preprocessed': df_f_preprocessed,
                'z_df_f': z_df_f,
                'final_shape': df_f_processed.shape
            })
            
            logger.info(f"Chandi's photometry preprocessing is completed")
            return result
        except Exception as e:
            logger.errror(f"Error in Chandni's preprocessing: {e}")
            raise
        
    def _apply_gaussian_filter(self,
                               photo: np.ndarray,
                               isos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        kernel_size = int(self.gaussian_sigma *3)
        x = np.arange(-kernel_size, kernel_size +1)
        gaussian_kernel = np.exp(-x**2 / (2 * self.gaussian_sigma**2))
        gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)
        
        photo_filtered = filtfilt(gaussian_kernel, 1, photo)
        isos_filtered = filtfilt(gaussian_kernel, 1, isos)
        
        return photo_filtered, isos_filtered
    
class PreprocessingPipeline:
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.eeg_preprocessor = self._create_eeg_preprocessor()
        self.photometry_preprocessors = self._create_photometry_preprocessors()
        
    def _create_eeg_preprocessor(self) -> EEGPreprocessor:
        
        eeg_config = self.config.get('eeg', {})
        return EEGPreprocessor(
            sample_rate=eeg_config.get('sample_rate', 1000.0),
            notch_freq=eeg_config.get('notch_frequency', 60.0),
            notch_quality=eeg_config.get('notch_quality', 30.0),
            bandpass_low=eeg_config.get('bandpass_low', 1.0),
            bandpass_high=eeg_config.get('bandpass_high', 70.0),
            artifact_threshold=eeg_config.get('artifact_threshold', 3.0)
        )
    def _create_photometry_preprocessors(self) -> Dict[str, PhotometryPreprocessor]:
        photo_config = self.config.get('photometry', {})
        
        return {
            'meiling': MeilingPhotometryPreprocessor(
                sample_rate=photo_config.get('sample_rate', 1000.0),
                lowpass_cutoff=photo_config.get('lowpass_cutoff', 10.0),
                savgol_window=photo_config.get('savgol_window', 301),
                savgol_polyorder=photo_config.get('savgol_polyorder', 4)
            ),
            'chandni': ChandniPhotometryPreprocessor(
                sample_rate=photo_config.get('sample_rate', 1000.0),
                gaussian_sigma=photo_config.get('gaussian_sigma', 75.0),
                savgol_window=photo_config.get('savgol_window', 301),
                savgol_polyorder=photo_config.get('savgol_polyorder', 4)
            )
        }
        
    def process_eeg(self, data: np.ndarray, **kwargs) -> PreprocessingResult:
        return self.eeg_preprocessor.process(data, **kwargs)
    
    def process_photometry(self,
                           photo_data: np.ndarray,
                           isos_data: np.ndarray,
                           path: str = 'meiling',
                           **kwargs) -> PreprocessingResult:
        
        if path not in self.photometry_preprocessors[path]:
            raise ValueError(f"Unknown preprocessing pathway: {path}")
        
        preprocessor = self.photometry_preprocessors[path]
        
        if path == 'meiling':
            return preprocessor.process(photo_data, isos_data, **kwargs)
        elif path == 'chandni':
            return preprocessor.process(photo_data, isos_data, **kwargs)
        
    def get_available_paths(self) -> list[str]:
        return list(self.photometry_preprocessors.keys())
    
    def update_copnfig(self, new_config: Dict[str, Any]) -> None:
        self.config.update(new_config)
        self.eeg_preprocessor = self._create_eeg_preprocessor()
        self.photometry_preprocessors = self._create_eeg_preprocessor()


if __name__ == "__main__":
    pass
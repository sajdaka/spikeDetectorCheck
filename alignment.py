from typing import Tuple, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d
import logging

from dataLoad import EEGRecord, PhotometryRecord

logger = logging.getLogger(__name__)

class SignalAlignment:
    
    def __init__(self, target_sampling_rate: float = 1000.0):
        self.target_sampling_rate = target_sampling_rate
        logger.info(f"Signal Alignment initialized")
        
    def align_signals(self,
                      eeg_record: EEGRecord,
                      photometry_record: PhotometryRecord,
                      eeg_channel: int = 9) ->Dict[str, np.ndarray]:
        
        try:
            logger.info(f"Starting signal alignment")
            
            eeg_info, events, eeg_channel_data = self._extract_eeg_info(
                eeg_record, eeg_channel
            )
            
            gcamp, isos, eeg_aligned, time_seconds = self._align_photo_eeg(
                eeg_info, eeg_channel_data, events, photometry_record
            )
            
            logger.info(f"Signal alignment complete")
            
            return {
                'gcamp': gcamp,
                'isos': isos,
                'eeg': eeg_aligned,
                'time': time_seconds,
                'metadata': {
                    'target_sampling_rate': self.target_sampling_rate,
                    'eeg_channel': eeg_channel,
                    'alignment_method': 'cross_correlation'
                }
            }
        except Exception as e:
            logger.error(f"Error during signal alignment: {e}")
            raise
        
    def _extract_eeg_info(self, eeg_record: EEGRecord, channel_index: int) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], np.ndarray]:
        
        try:
            if eeg_record.data.ndim == 1:
                if channel_index == 0:
                    raise ValueError(f'Channel {channel_index} not available')
                eeg_channel_data = eeg_record.data
            else:
                if channel_index >= eeg_record.n_channels:
                    raise ValueError(f" Channel {channel_index} is not available")
                eeg_channel_data = eeg_record.data[channel_index, :]
            
            eeg_info = {
                'metadata': {
                    'sampleRate': eeg_record.sample_rate
                },
                'sampleNumbers': eeg_record.sample_numbers,
                'timestamps': eeg_record.timestamps
            }
            
            events = eeg_record.events
            
            logger.info(f"Extracted EEG channel {channel_index}")
            
            return eeg_info, events, eeg_channel_data
        
        except Exception as e:
            logger.error(f"Error extracting EEG info: {e}")
            raise
        
    def _align_photo_eeg(self,
                         eeg_info: Dict[str, Any],
                         eeg_channel_data: np.ndarray,
                         events: Dict[str, np.ndarray],
                         photometry_record: PhotometryRecord)->Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        try:
            eeg_fs = eeg_info['metadata']['sampleRate']
            logger.info(f"Original EEG sampling rate: {eeg_fs}")
            
            if eeg_fs != self.target_sampling_rate:
                logger.info(f"resampling EEG")
                
                eeg_sync = self._get_oep_ttl(eeg_info, events)
                
                eeg_channel_data = signal.resample(
                    eeg_channel_data,
                    int(len(eeg_channel_data) * self.target_sampling_rate / eeg_fs)
                )
                
                factor = int(eeg_fs /self.target_sampling_rate)
                trim_length = (len(eeg_sync) // factor) * factor
                ttl_signal_trimmed = eeg_sync[:trim_length]
                
                ttl_reshaped = ttl_signal_trimmed.reshape(factor, -1, order='F')
                eeg_sync = np.max(ttl_reshaped, axis=0)
                eeg_fs = self.target_sampling_rate
            else:
                eeg_sync = self._get_oep_ttl(eeg_info, events)
                
            logger.info(f"EEG sync signal length: {len(eeg_sync)}")
            
            photo_raw = photometry_record.photo
            isos_raw = photometry_record.isos
            photo_sync = photometry_record.sync
            
            fs1 = photometry_record.sample_rate
            fs2 = eeg_fs
            
            t1 = np.arange(len(photo_sync)) / fs1
            t2 = np.arange(0, np.max(t1), 1/fs2)
            
            logger.info(f'Time vector lengths: Original - {len(t1)}, Target: {len(t2)}')
            
            photo_sync_interp = interp1d(
                t1,
                photo_sync.astype(float),
                kind='previous',
                bounds_error=False,
                fill_value='extrapolate'
            )
            photo_sync_long = photo_sync_interp(t2)
            
            photo_interp = interp1d(
                t1,
                photo_raw,
                kind='cubic',
                bounds_error=False,
                fill_value='extrapolate',
            )
            photo_long = photo_interp(t2)
            
            isos_interp = interp1d(
                t1,
                isos_raw,
                kind='cubic',
                bounds_error=False,
                fill_value='extrapolate'
            )
            isos_long = isos_interp(t2)
            
            lag = self._align_risetime(photo_sync_long, eeg_sync)
            logger.info(f"Detected lag at: {lag}")
            
            trim = abs(lag)
            
            if lag > 0:
                eeg_aligned = eeg_channel_data[trim:]
                eeg_sync_trim = eeg_sync[trim:]
                photo = photo_long
                isos = isos_long
                time_aligned = t2
                logger.info("EEG leads photometry")
            elif lag < 0:
                eeg_aligned = eeg_channel_data
                photo_sync_long_trim = photo_sync_long[trim:]
                photo = photo_long[trim:]
                isos = isos_long[trim:]
                time_aligned = t2[trim:]
                logger.info("Photometry leads EEG")
            else:
                eeg_aligned = eeg_channel_data
                photo = photo_long
                isos = isos_long
                time_aligned = t2
                logger.info("No lag detected")
            
            logger.info(f'lag time: {lag}')
            
            #TODO: causes threading issue maybe use different library but is displaying the alignment necessary?
            #self._create_alignment_plot(
            #    lag, eeg_sync, photo_sync_long, eeg_channel_data, photo, eeg_aligned
            #)
            
            return photo, isos, eeg_aligned, time_aligned
        
        except Exception as e:
            logger.error(f"Error in alignment process: {e}")
            raise
    
    def _get_oep_ttl(self, eeg_info: Dict[str, Any], events: Dict[str, np.ndarray]) -> np.ndarray:
        
        try:
            TTL = np.zeros(len(eeg_info['sampleNumbers']))
            
            if len(events['sample_number']) == 0:
                print("No TTL events found in recording")
                return TTL
            
            TTL_index = events['sample_number'].copy()
            TTL_index = TTL_index - eeg_info['sampleNumbers'][0]
            
            
            valid = (TTL_index > 0) & (TTL_index <= len(TTL))
            TTL_index_valid = TTL_index[valid].astype(int)
            
            if len(TTL_index) == 0:
                print("No valid TTL indices after filtering")
                return TTL
            

            if 'states' in events and len(events['states']) > 0:
                states_valid = events['states'][valid]
                
                
                # Method 1: Use state transitions to create square waves
                current_state = 0  # Start with TTL off
                
                for i, (idx, state) in enumerate(zip(TTL_index_valid, states_valid)):
                    if state == 1:  # Rising edge
                        current_state = 1
                    elif state == 0:  # Falling edge
                        current_state = 0
                    
                    # Find the next state change or end of data
                    if i < len(TTL_index_valid) - 1:
                        next_idx = TTL_index_valid[i + 1]
                    else:
                        next_idx = len(TTL)
                    
                    # Set TTL values between this event and the next
                    if current_state == 1:
                        TTL[idx:min(next_idx, len(TTL))] = 1
            
            else:
                print("Warning: No state information found, using simple event marking")
                TTL[TTL_index_valid] = 1
                        
            return TTL
        except Exception as e:
            logger.error(f"error in get OEP method: {e}")
            raise
    def _align_risetime(self,
                        x: np.ndarray,
                        y: np.ndarray,
                        state_levels: Tuple[float, float] = (0, 1),
                        max_edges: int = 10) -> int:
        
        try:
            x = np.asarray(x)
            y = np.asarray(y)
            
            #find the rising edges in data
            threshold = (state_levels[1] + state_levels[0]) / 2
            
            x_binary = (x > threshold).astype(int)
            y_binary = (y > threshold).astype(int)
            
            x_edges = np.diff(x_binary)
            y_edges = np.diff(y_binary)
            
            x_rising = np.where(x_edges > 0)[0]
            y_rising = np.where(y_edges > 0)[0]
            
            if len(x_rising) == 0 or len(y_rising) == 0:
                print("No rising edges found in one or either of the signals")
                return 0
            
            n_edges = min(max_edges, len(x_rising), len(y_rising))
            if n_edges == 0:
                return 0
            lag = x_rising[0] - y_rising[0]
            
            return lag
        except Exception as e:
            logger.error(f"error in align risetime: {e}")
            raise
    
    def _create_alignment_plot(self,
                               lag: int,
                               eeg_sync: np.ndarray,
                               photo_sync: np.ndarray,
                               eeg_data: np.ndarray,
                               photo_data: np.ndarray,
                               eeg_aligned: np.ndarray):
        
        try:
            fig, axes = plt.subplots(6, 1, figsize=(14, 10))
            fig.suptitle('Signal Alignment Visualization')
            
            max_plot_samples = 50000
            if len(eeg_sync) > max_plot_samples:
                step = len(eeg_sync) // max_plot_samples
                plot_indices = slice(None, None, step)
            else:
                plot_indices = slice(None)
            
            if lag > 0:
                
                axes[0].plot(eeg_sync[plot_indices])
                axes[0].set_title('Original EEG Sync')
                
                if len(eeg_sync) > abs(lag):
                    axes[1].plot(eeg_sync[abs(lag):][plot_indices])
                axes[1].set_title('Aligned EEG Sync')
                
                axes[2].plot(photo_sync[plot_indices])
                axes[2].set_title('Original Photo Sync')
                
                axes[3].plot(eeg_data[plot_indices])
                axes[3].set_title('Original EEG')
                
                axes[4].plot(eeg_aligned[plot_indices])
                axes[4].set_title('Aligned EEG')
                
                axes[5].plot(photo_data[plot_indices])
                axes[5].set_title('GCaMP (upsampled)')
                
            elif lag < 0:
                axes[0].plot(photo_sync[plot_indices])
                axes[0].set_title('Original Photo Sync')
                
                if len(photo_sync) > abs(lag):
                    axes[1].plot(photo_sync[abs(lag):][plot_indices])
                axes[1].set_title('Aligned Photo Sync')
                
                axes[2].plot(eeg_sync[plot_indices])
                axes[2].set_title('Original EEG Sync')
                
                axes[3].plot(photo_data[plot_indices])
                axes[3].set_title('Original GCaMP (upsampled)')
                
                aligned_gcamp = photo_data
                if len(photo_data) > abs(lag):
                    aligned_photo = photo_data[abs(lag):]
                
                if len(aligned_gcamp) > 0:
                    axes[4].plot(aligned_photo[plot_indices])
                axes[4].set_title('Aligned GCaMP')
                
                axes[5].plot(eeg_data[plot_indices])
                axes[5].set_title('Original EEG')
            else:
                
                axes[0].plot(eeg_sync[plot_indices])
                axes[0].set_title('EEG Sync (No lag detected)')
                
                axes[1].plot(photo_sync[plot_indices])
                axes[1].set_title('Photo Sync (No lag detected)')
                
                axes[2].plot(eeg_data[plot_indices])
                axes[2].set_title('EEG Data')
                
                axes[3].plot(photo_data[plot_indices])
                axes[3].set_title('GCaMP Data')
                
                axes[4].text(0.5, 0.5, 'Signals already aligned', 
                           ha='center', va='center', transform=axes[4].transAxes)
                axes[5].text(0.5, 0.5, f'Lag: {lag} samples', 
                           ha='center', va='center', transform=axes[5].transAxes)
            
            for ax in axes[1:]:
                ax.sharex(axes[0])
            
            plt.tight_layout()
            plt.show()
            logger.info("Alignment visualization is complete")
            
        except Exception as e:
            logger.warning(f"Could not create alignment visualization from error: {e}")
            
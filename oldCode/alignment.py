from multiprocessing import Value
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d
import warnings
from open_ephys.analysis import Session


def alignment_start(session, chx, data_ppd):
    EEG, events, eeg_chx = extract_eeg_info(session, chx)
    gcamp, iso, eeg, time_second = alignPhotoEEG(EEG, eeg_chx, events, data_ppd)
    return gcamp, iso, eeg, time_second




def extract_eeg_info(session, channel_index, processor_name=None):
    recording = session.recordnodes[0].recordings[0]
           
    eeg_chx = recording.continuous[0].get_samples(0, 1220000)
    eeg_chx = eeg_chx[:, channel_index]
    
    EEG = {
        'metadata': {
            'sampleRate': recording.continuous[0].metadata['sample_rate']
        },
        'sampleNumbers': recording.continuous[0].sample_numbers,
        'timestamps': recording.continuous[0].timestamps
    }
                
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
    
    return EEG, events, eeg_chx
    
    



def alignPhotoEEG(EEG, eeg_chx, events, photo_file):
    
    eeg_fs = EEG['metadata']["sampleRate"]
    
    if eeg_fs != 1000:
        eeg_sync = get_OEP_TTL(EEG, events)
        print(eeg_sync)
        
        eeg_chx = signal.resample(eeg_chx, int(len(eeg_chx)))
        
        factor = int(eeg_fs /1000)
        #ensures the length of the time series data is a multiple of factor
        trim_length = (len(eeg_sync) // factor) * factor
        ttl_signal_trimmed = eeg_sync[:trim_length]
        
        #reshapes
        ttl_reshaped = ttl_signal_trimmed.reshape(factor, -1, order='F')
        eeg_sync = np.max(ttl_reshaped, axis=0)
        eeg_fs = 1000
    else:
        eeg_sync = get_OEP_TTL(EEG, events)
        print(eeg_sync)
        
    gcamp_raw = photo_file['analog_1']
    iso_raw = photo_file['analog_2']
    photo_sync = photo_file['digital_1']
    
    fs1 = photo_file['sampling_rate']
    fs2 = eeg_fs
    
    #original time vector
    t1 = np.arange(len(photo_sync)) / fs1
    
    #new upsampled time vector
    t2 = np.arange(0, np.max(t1), 1/fs2)
    
    #interpolate all data used
    photo_sync_interp = interp1d(t1, photo_sync.astype(float), kind='previous',
                                 bounds_error=False, fill_value='extrapolate')
    photo_sync_long = photo_sync_interp(t2)
    
    gcamp_interp = interp1d(t1, gcamp_raw, kind='cubic', bounds_error=False,
                            fill_value='extrapolate')
    gcamp_long = gcamp_interp(t2)
    
    iso_interp = interp1d(t1, iso_raw, kind='cubic',
                          bounds_error=False, fill_value='extrapolate')
    iso_long = iso_interp(t2)
    
    #align signals using cross-correlation
    lag = align_risetime(photo_sync_long, eeg_sync)
    print(lag)
    trim = abs(lag)
    
    if lag>0:
        eeg = eeg_chx[trim:]
        eeg_sync_trim = eeg_sync[trim:]
        gcamp = gcamp_long
        iso = iso_long
        time_aligned = t2
    elif lag < 0:
        eeg = eeg_chx
        photo_sync_long_trim = photo_sync_long[trim:]
        gcamp = gcamp_long[trim:]
        iso = iso_long[trim:]
        time_aligned = t2[trim:]
    else:
        eeg = eeg_chx
        gcamp  = gcamp_long
        iso = iso_long
        time_aligned = t2
        
    fig, axes = plt.subplots(6, 1, figsize=(14, 10))
    fig.suptitle('Signal Alignment')
    
    if lag > 0:
        axes[0].plot(eeg_sync)
        axes[0].set_title('Original EEG Sync')
        
        axes[1].plot(eeg_sync_trim)
        axes[1].set_title('Aligned EEG Sync')
        
        axes[2].plot(photo_sync_long)
        axes[2].set_title('Original Photo Sync')
        
        axes[3].plot(eeg_chx)
        axes[3].set_title('Original EEG')
        
        axes[4].plot(eeg)
        axes[4].set_title('Aligned EEG')
        
        axes[5].plot(gcamp)
        axes[5].set_title('Original GCaMP (upsampled)')
        
    elif lag < 0:
        axes[0].plot(photo_sync_long)
        axes[0].set_title('Original Photo Sync')
        
        axes[1].plot(photo_sync_long_trim)
        axes[1].set_title('Aligned Photo Sync')
        
        axes[2].plot(eeg_sync)
        axes[2].set_title('Original EEG Sync')
        
        axes[3].plot(gcamp_long)
        axes[3].set_title('Original GCaMP (upsampled)')
        
        axes[4].plot(gcamp)
        axes[4].set_title('Aligned GCaMP (upsampled)')
        
        axes[5].plot(eeg)
        axes[5].set_title('Original EEG')
    
    for ax in axes[1:]:
        ax.sharex(axes[0])
    
    plt.tight_layout()
    plt.show()
    
    return gcamp, iso, eeg, time_aligned
        
          
def get_OEP_TTL(data, events):
    TTL = np.zeros(len(data['sampleNumbers']))
    
    if len(events['sample_number']) == 0:
        print("No TTL events found in recording")
        return TTL
    
    TTL_index = events['sample_number'].copy()
    TTL_index = TTL_index - data['sampleNumbers'][0]
    
    
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

#custom risetime alginment for python
def align_risetime(x, y, state_levels=[0, 1], max_edges=10):
    
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

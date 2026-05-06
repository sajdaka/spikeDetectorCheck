
import numpy as np
from typing import Tuple, List
from scipy.signal import find_peaks


def detect_spikes(zscored_signal: np.ndarray, non_zscored_signal: np.ndarray,
                  baseline_start_time: float, baseline_end_time: float,
                  time_seconds: float) -> Tuple[List[float], List[float], List[float], List[float], int]:
    
    effective_thresh = 4.5 #std deviations away for spike detection
    spkdur_samples = (
        20,             
        80
    )   #min and max spike durations
    
    all_peaks = []
    all_heights = []
    all_widths = []
    all_prominences = []
    for polarity in ['positive', 'negative']:
        if polarity == 'negative':
            signal_data = -zscored_signal
        else:
            signal_data = zscored_signal

        half_peaks, half_peak_properties = find_peaks(
            signal_data,
            height=effective_thresh,
            distance=int(spkdur_samples[0]),
            width=(spkdur_samples[0], spkdur_samples[1]),
            prominence=effective_thresh*0.5
        )
        all_peaks.append(half_peaks)
        all_heights.append(half_peak_properties['peak_heights'])
        all_widths.append(half_peak_properties['widths'])
        all_prominences.append(half_peak_properties['prominences'])

    peaks = np.concatenate(all_peaks)
    peak_properties = {
        'peak_heights': np.concatenate(all_heights),
        'widths': np.concatenate(all_widths),
        'prominences': np.concatenate(all_prominences),
    }
    
    filtered = []
    close_samples = 1 #spikes that are too close to the edges by seconds
    seizure_times = [(0,0), (0,0)]
    
    for i, spike in enumerate(peaks):
        if(spike < close_samples or spike > len(zscored_signal) - close_samples):
            continue
        if peak_properties['peak_heights'][i] > 15000000: #huge number can be tuned basically ensures none of the artifacts made it through preprocessing
            continue
        inSeizure = False
        for seizure in seizure_times:
            if spike > seizure[0] and spike < seizure[1]:
                inSeizure = True
        if inSeizure:
            continue
        
        filtered.append(i)
    
    sorted_peak_times = []
    for i in filtered:
        sorted_peak_times.append((peaks[i], i))
        
    sorted_peak_times.sort()
    
    final_filtered = []
    min_separation = 10e-3
    for peak in sorted_peak_times:
        if final_filtered:
            last_spike = final_filtered[-1][0]
            distance = peak[0] - last_spike

            if distance < min_separation:
                continue
            else:
                final_filtered.append(peak)
        else:
            final_filtered.append(peak)
            
    final_peaks = []
    final_peak_amplitudes = []
    final_peak_widths = []
    final_peak_intervals = []
    for valid_peak in final_filtered:
        final_peaks.append(valid_peak[0])
        final_peak_amplitudes.append(peak_properties['peak_heights'][valid_peak[1]])
        final_peak_widths.append(peak_properties['widths'][valid_peak[1]])
        final_peak_intervals.append((valid_peak[0], valid_peak[0] + peak_properties['widths'][valid_peak[1]]))
    return final_peaks, final_peak_amplitudes, final_peak_widths, final_peak_intervals, len(final_peaks)

def baseline_selector(signal: np.ndarray, baseline_start_time: float, baseline_end_time: float,
                      size_of_chunk_seconds: float, size_of_segment_in_chunks: float, fs: float) -> np.ndarray:
    #size of chunk is the smaller portion that makes up the segment
    chunk_duration = size_of_chunk_seconds * fs
    
    baseline_period = signal[int(baseline_start_time * fs) : int(baseline_end_time * fs)]
    chunks = []
    for i in range(int(size_of_segment_in_chunks)):
        chunk_start = i * chunk_duration
        chunk_end = (i + 1) * chunk_duration
        chunk = baseline_period[int(chunk_start):int(chunk_end)]
        chunks.append(np.std(chunk))
        
    chunks_std = np.array(chunks)
    
    mean_std = np.mean(chunks)
    std_of_std = np.std(chunks_std)
    cv = (std_of_std / mean_std) * 100
    min_cv = cv
    min_baseline_start = baseline_start_time
    min_baseline_end = baseline_start_time + (size_of_chunk_seconds * size_of_segment_in_chunks)
    baseline_start_temp = baseline_start_time
    baseline_end_temp = baseline_start_time + (size_of_chunk_seconds * size_of_segment_in_chunks)
    
    while(cv > 20):
        if(baseline_end_temp + size_of_chunk_seconds > len(signal) / fs):
            cv = min_cv
            baseline_start_time = min_baseline_start
            baseline_end_time = min_baseline_end
            break
        if min_cv > cv:
            min_baseline_start = baseline_start_temp
            min_baseline_end = baseline_end_temp
        chunks.pop(0)
        baseline_start_temp += size_of_chunk_seconds
        baseline_end_temp += size_of_chunk_seconds
        new_chunk_start = baseline_end_temp - size_of_chunk_seconds
        added_chunk = signal[int(new_chunk_start * fs): int(baseline_end_temp * fs)]
        chunks.append(np.std(added_chunk))
        chunks_std = np.array(chunks)
        mean_std = np.mean(chunks_std)
        std_of_std = np.std(chunks_std)
        cv = (std_of_std / mean_std) * 100
        
    baseline = signal[int(min_baseline_start * fs): int(min_baseline_end * fs)]
    
    baseline_mean = np.nanmean(baseline)
    baseline_std = np.nanstd(baseline)
    
    return (signal - baseline_mean) / baseline_std
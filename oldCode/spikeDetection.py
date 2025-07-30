import time
import tkinter as tk
from tkinter import messagebox
import numpy as np
from typing import Union, Tuple
from scipy.spatial.distance import cdist
from sklearn import base
from alignment import alignment_start
from photometry import EEG_preprocessing, baselineZscore, MeilingPreprocessing, MeilingDetrended, MeilingDenoising, ChandniPreprocessing
import sys
import os
import json
from open_ephys.analysis import Session
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.signal import find_peaks
from data_import import import_ppd


def main():
    pio.renderers.default = 'browser'
    directory = ""
    directory = os.path.join('./EEGData/Meiling_EEG', sys.argv[1])
    try:
        session = Session(directory)
    except FileNotFoundError:
        print(f"There is no file named: {sys.argv[1]} found in the directory")
        exit()
        
    try:
        with open('seizureOnsets.json', 'r') as file:
            onsets = json.load(file)
    except FileNotFoundError:
        print("The seizure onset file was not found")
        exit()
    except json.JSONDecodeError:
        print("The json file was corrupted or formatted improperly")
        exit()
        
    if sys.argv[1] in onsets:
        onset = onsets[sys.argv[1]]
    else:
        onset = -1
     
    recording = session.recordnodes[0].recordings[0]
    EEG_data = recording.continuous[0].get_samples(start_sample_index=0, end_sample_index=1220000)
    if EEG_data.ndim == 1:
        EEG_data = EEG_data.reshape(1, -1)
        n_channels = 1
    else:
        if EEG_data.shape[0] > EEG_data.shape[1]:
            EEG_data = EEG_data.T

    directory = ""
    directory = os.path.join('./FiberPhotometryData/LC', sys.argv[2])
    try:
        data_ppd = import_ppd(directory)
    except FileNotFoundError:
            print(f"There is no file named {sys.argv[2]} in the directory FiberPhotometryData")
            exit()
    

        

    gui = SpikeDetectorGUI(EEG_data, data_ppd, session, onset)
    return gui.run()


def printGraphs(EEG):
    
    downsample_factor = 10
    n_samples = EEG.shape[1]
    plot_indices = np.arange(0, n_samples, downsample_factor)
    
    fs = 1000
    time_axis = plot_indices/fs
    
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=EEG[2, plot_indices],
            mode='lines',
            line_shape='spline',
            name="Time vs. Raw Photometry"
        ),
        row=1, col=1
    )
    
    spike_results = detect_spikes_for_plot(EEG, fs)
    print(spike_results)
    if spike_results is not None and len(spike_results) > 0:
        
        spike_times = spike_results[:, 1] / fs
        spike_amplitudes = []
        
        for spike_time in spike_times:
            sample_idx = int(spike_time * fs)
            if sample_idx < len(EEG[0]):
                spike_amplitudes.append(EEG[0, sample_idx])
            else:
                spike_amplitudes.append(0)
                
        fig.add_trace(
            go.Scatter(
                x=spike_times,
                y=spike_amplitudes,
                mode='markers',
                marker=dict(
                    color='red',
                    size=10,
                    symbol='x',
                    line=dict(width=2)
                ),
                hovertemplate='<b>Spike Detected</b><br>' +
                            'Time: %{x:.3f}s<br>' +
                            'Amplitude: %{y:.3f}μV<br>' +
                            '<extra></extra>'
            ),
            row=1, col=1
        )

    fig.update_layout(title="Preprocessed EEG Data")
    fig.show()
    
def detect_spikes_for_plot(EEG, fs, channel=0):
    
    default_params = {
        'fs': fs,
        'tmul': 3,
        'absthresh': 0.4,
        'sur_time': 10000,
        'close_to_edge': 0.9,
        'too_high_abs': 1e9,
        'spkdur': (70,200),
        'chx': 8
    }
    
    
    try:
        results = spikeDetection(
            values=EEG,
            fs=default_params['fs'],
            tmul=default_params['tmul'],
            absthresh=default_params['absthresh'],
            sur_time=default_params['sur_time'],
            close_to_edge=default_params['close_to_edge'],
            too_high_abs=default_params['too_high_abs'],
            spkdur=default_params['spkdur'],
            chx=default_params['chx'],
            multiChannelRequirements=False
        )
        return results
    except Exception as e:
        print(f"Error in spike detection: {e}")
        return None   
        
        
        
        
        
class SpikeDetectorGUI:
    def __init__(self, EEG, ppd, session, onset):
        self.EEG = EEG
        self.ppd = ppd
        self.session = session
        self.results = None
        self.onset = onset
         
        self.default_entries = {
            'start_time': 0,
            'end_time': 0,
            'fs': 1000,
            'tmul': 3,
            'absthresh': 0.4,
            'sur_time': 1000,
            'close_to_edge': 0.1,
            'too_high_abs': 1e9,
            'spkdur_min': 70, #determine the real bounds for a spike
            'spkdur_max': 200,
            'chx' : int(8),
            'meiling_preprocessing': False,
            'chandni_preprocessing': False,
        }
        
        self.prompts = {
            'Enter EEG channel number - chx': 'chx',
            'Enter sampling frequency - fs: Hertz(Hz)': 'fs',
            'Enter start time for baseline - Seconds(s)': 'start_time',
            'Enter end time for baseline - Seconds(s)': 'end_time',
            'Enter minimum relative amplitude - tmul: No units': 'tmul',
            'Enter min absolute amplitude - absthresh: Microvolts(μV)': 'absthresh',
            'Enter surround time - sur_time: Seconds(s)': 'sur_time',
            'Enter time surrounding start and end of sample to ignore - close_to_edge: Seconds(s)': 'close_to_edge',
            'Enter amplitude above which to reject as artifact - too_high_abs: Microvolts(μV)': 'too_high_abs',
            'Enter spike duration minimum - spkdur_min: Milliseconds(ms)': 'spkdur_min',
            'Enter spike duration maximum - spkdur_max: Milliseconds(ms)': 'spkdur_max',
            
        }
        
        self.setup_gui()
        
    def setup_gui(self):
        self.ui = tk.Tk()
        self.ui.title("Spike Detection Parameters")
        self.ui.geometry("700x675")
        
        main_frame = tk.Frame(self.ui, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        title_label = tk.Label(main_frame, text="Spike Detection Parameters")
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        self.entries = {}
        
        row = 1
        for prompt, key in self.prompts.items():
            label = tk.Label(main_frame, text=prompt, anchor='w', justify='left')
            label.grid(row=row, column=0, sticky='w', pady=5, padx=(0,10))
            
            entry = tk.Entry(main_frame, width=20)
            entry.grid(row=row, column=1, sticky='ew', pady=5)
            
            entry.insert(0, str(self.default_entries[key]))
            
            self.entries[key] = entry
            
            row += 1
            
            if key == 'end_time':
                separator = tk.Frame(main_frame, height=2, bd=1, relief=tk.SUNKEN)
                separator.grid(row=row, column=0, columnspan=2, sticky='ew', pady=(20, 10))
                row += 1
        
        separator = tk.Frame(main_frame, height=2, bd=1, relief=tk.SUNKEN)
        separator.grid(row=row, column=0, columnspan=2, sticky='ew', pady=(20, 10))
        
        row += 1 
        
        self.meiling_preprocessing = tk.BooleanVar(value=False)
        self.chandni_preprocessing = tk.BooleanVar(value=False)
        
        preprocessing_frame = tk.Frame(main_frame)
        preprocessing_frame.grid(row=row, column=0, columnspan=2, sticky='w', pady=5)
        
        meiling_cb = tk.Checkbutton(preprocessing_frame,
                                    text="Use Meiling's preproccessing pipeline for the photometry data",
                                    variable=self.meiling_preprocessing)
        meiling_cb.grid(row=0, column=0, sticky='w', pady=2)
        
        chandni_cb = tk.Checkbutton(preprocessing_frame,
                                    text="Use Chandni's preproccessing pipeline for the photometry data",
                                    variable=self.chandni_preprocessing)
        chandni_cb.grid(row=1, column=0, sticky='w', pady=2)
        
        row += 1
                        
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        button_frame = tk.Frame(main_frame)
        button_frame.grid(row=row, column=0, columnspan=2, pady=20)
        
        submit_button = tk.Button(button_frame, text="Run Spike Detection",
                                  command=self.collect_inputs, bg='lightblue')
        submit_button.pack(side=tk.LEFT, padx=5)
        
        reset_button = tk.Button(button_frame, text="Reset to Defaults",
                                 command=self.reset_to_defaults, bg='lightgray')
        reset_button.pack(side=tk.LEFT, padx=5)
        
        self.results_label = tk.Label(main_frame, text="Results will appear here",
                                      wraplength=700, justify='left', anchor='nw',
                                      bg='white', relief='sunken', padx=10, pady=10)
        self.results_label.grid(row=row+1, column=0, columnspan=2, pady=10, sticky='ew')
        
    
    def reset_to_defaults(self):
        
        for key, entry in self.entries.items():
            entry.delete(0, tk.END)
            entry.insert(0, str(self.default_entries[key]))
            
        self.chandni_preprocessing.set(False)
        self.meiling_preprocessing.set(False)
            
    def get_value_or_default(self, key):
        
        if key not in self.entries:
            return self.default_entries[key]
        
        value = self.entries[key].get().strip()
        if value == '':
            return self.default_entries[key]
        
        try:
            if key == 'chx':
                return int(float(value))
            if key in ['fs', 'tmul', 'absthresh', 'sur_time', 'close_to_edge', 
                  'too_high_abs', 'start_time', 'end_time', 'spkdur_min', 'spkdur_max']:
                return float(value)
            else:
                return value
        except ValueError:
            messagebox.showwarning("Invalid Input:", f"Invalid value for {key}: '{value}'. Using default: {self.default_entries[key]}")
            return self.default_entries[key]
    
    
    def collect_inputs(self):
        
        try: 
            params = {}
            for key in self.default_entries.keys():
                params[key] = self.get_value_or_default(key)
                
            params['meiling_preprocessing'] = self.meiling_preprocessing.get()
            params['chandni_preprocessing'] = self.chandni_preprocessing.get()
            
                
            self.run_spike_detection(params)
        except Exception as e:
            messagebox.showerror("Error", f"Error in input paramters: {str(e)}")
        
            
    def run_spike_detection(self, params):
        try:
            
            session = self.session
            EEG_sample = self.EEG
            ppd = self.ppd
            
            ppd_aligned, isos_aligned, EEG_aligned, time_seconds = alignment_start(session, params['chx'], ppd)
            
            if len(EEG_aligned) > len(time_seconds):
                dead = len(EEG_aligned) - len(time_seconds)
                EEG_aligned = EEG_aligned[:-dead]
            elif len(EEG_aligned) < len(time_seconds):
                dead = len(time_seconds) - len(EEG_aligned)
                ppd_aligned = ppd_aligned[:-dead]
                isos_aligned = isos_aligned[:-dead]
                time_seconds = time_seconds[:-dead]
            
            EEG_zscored = baselineZscore(EEG_aligned, params['start_time'], params['end_time'], params['fs'])
            
            EEG_filtered = EEG_preprocessing(EEG_zscored, threshold=3)
            
            if params['chandni_preprocessing']:
                ppd_dF_F_precorrected, ppd_dF_F_corrected, z_ppd_dF_F_corrected = ChandniPreprocessing(ppd_aligned, isos_aligned)
            
            else:
                ppd_denoised, isos_denoised = MeilingDenoising(ppd_aligned, isos_aligned, params['fs'])
                ppd_detrended, ppd_expfit, ppd_est_motion, isos_detrended = MeilingDetrended(ppd_denoised, isos_denoised, time_seconds)
                ppd_dF_F_precorrected, ppd_dF_F_corrected, z_ppd_dF_F_corrected = MeilingPreprocessing(ppd_detrended, ppd_expfit, ppd_est_motion)
            
            z_ppd_dF_F_corrected = baselineZscore(ppd_dF_F_corrected, params['start_time'], params['end_time'], params['fs'])
            
            fig = make_subplots(rows=3, cols=2)
            fig.update_layout(title="EEG and photometry run through pre-processing pipeline")
            
            if len(time_seconds) > 10000:
                step = len(time_seconds) // 1000
                time_seconds = time_seconds[::step]
                ppd_aligned = ppd_aligned[::step]
                isos_aligned = isos_aligned[::step]
                EEG_aligned = EEG_aligned[::step]
                z_ppd_dF_F_corrected = z_ppd_dF_F_corrected[::step]
                EEG_filtered = EEG_filtered[::step]
            fig.add_trace(
                go.Scatter(
                    x=time_seconds,
                    y=ppd_aligned,
                    mode='lines',
                    line_shape='spline',
                    name="Time vs. Raw Photometry/Isos"
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=time_seconds,
                    y=isos_aligned,
                    mode='lines',
                    line_shape='spline',
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=time_seconds,
                    y=EEG_aligned,
                    mode='lines',
                    line_shape='spline',
                    name='Time vs. Raw EEG'
                ),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=time_seconds,
                    y=z_ppd_dF_F_corrected,
                    mode='lines',
                    line_shape='spline',
                    name="Time vs. Processed Photometry"
                ),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(
                    x=time_seconds,
                    y=EEG_filtered,
                    mode='lines',
                    line_shape='spline',
                    name="Time vs. Processed EEG"
                ),
                row=2, col=2
            )
            fig.add_vline(x=self.onset/1000, line_dash='dash', line_color='red', row=2, col=2)
            fig.add_vline(x=self.onset/1000, line_dash='dash', line_color='red', row=1, col=2)

            
            fig.show()
            
            
            spkdur = (params['spkdur_min'], params['spkdur_max'])
            
            # Run spike detection
            self.results = spikeDetection(
                values=EEG_filtered,
                fs=params['fs'],
                tmul=params['tmul'],
                absthresh=params['absthresh'],
                sur_time=params['sur_time'],
                close_to_edge=params['close_to_edge'],
                too_high_abs=params['too_high_abs'],
                spkdur=spkdur,
                chx=params['chx'],
                multiChannelRequirements=False
            )
            
            n_spikes = self.results.shape[0]
            if n_spikes > 0:
                channels = np.unique(self.results[:, 0])
                result_text = f"Detected {n_spikes} spikes across {len(channels)} channels\n"
                result_text += f"Channels with spikes: {channels}\n"
                result_text += f"Time range: {params['start_time']}-{params['end_time']} seconds"
            else:
                result_text = "No spikes detected in the specified time range"
            
            self.results_label.config(text=result_text)
            
            
            messagebox.showinfo("Success", f"Spike detection completed!\n{result_text}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error in spike detection: {str(e)}")
            print(f"Error: {e}")
    
    def run(self):
        self.ui.mainloop()
        return self.results
    
def spikeDetection(values: np.ndarray, 
                  fs: float, 
                  tmul: float = 3.0,
                  absthresh: Union[float, np.ndarray] = 0.4,
                  sur_time: float = 1000.0,
                  close_to_edge: float = 0.1,
                  too_high_abs: float = 1e9,
                  spkdur: Tuple[float, float] = (1.0, 2000.0),
                  chx: int = 8,
                  multiChannelRequirements: bool = False) -> np.ndarray:

    spkdur_samples = (spkdur[0] * fs / 1000, spkdur[1] * fs / 1000)
    
    
    
    
    if values.ndim == 2:
        
        if values.shape[0] == 1:
            values = values[0,:]
        else:
            if chx < values.shape[0]:
                values = values[chx, :]
            else:
                raise ValueError(f"Channel {chx} is not available in the array. The EEG only has {values.shape[0]} channels.")
        
    elif values.ndim == 1:
        pass
    else:
        raise ValueError(f"Invalid data dimensions: {values.shape}")
        
    
    nsamples = len(values)
    
    
    hpdata = values - np.nanmean(values)
    
    
    lthresh = np.median(np.abs(hpdata))
    thresh = lthresh * tmul
    
    
    if np.isscalar(absthresh):
        absthresh_val = absthresh
    else:
        absthresh_val = absthresh[0] if len(absthresh) > 0 else 0.4
    
    all_spikes = []
    
    if np.sum(np.isnan(hpdata)) == len(hpdata):
        return np.array([]).reshape(0, 5)
    
    
    # Detect both positive and negative spikes
    for polarity in ['positive', 'negative']:
        if polarity == 'negative':
            signal_data = -hpdata  # Invert for negative spike detection
        else:
            signal_data = hpdata
        
        
        peaks, peak_properties = find_peaks(
            signal_data,
            height=max(thresh, absthresh_val),          
            distance=int(spkdur_samples[0]),  
            width=(spkdur_samples[0]/4, spkdur_samples[1]/2),  
            prominence=absthresh_val/2,   
            rel_height=0.5              
        )
        
        if len(peaks) == 0:
            continue
        
        
        peak_heights = peak_properties['peak_heights']
        peak_widths = peak_properties['widths']
        peak_prominences = peak_properties['prominences']
        
        
        for i, peak_idx in enumerate(peaks):
            peak_time = peak_idx
            peak_amplitude = peak_heights[i]
            peak_width_samples = peak_widths[i]
            peak_prominence = peak_prominences[i]
            
            
            peak_duration_ms = peak_width_samples * 1000 / fs
            
            
            sur_samples = int(sur_time * fs / 1000)
            istart = max(0, peak_time - sur_samples)
            iend = min(len(hpdata), peak_time + sur_samples)
            
            if iend - istart < 10:
                continue
            
            
            alt_thresh = np.median(np.abs(hpdata[istart:iend])) * tmul
            
            
            amplitude_ok = (peak_amplitude > alt_thresh and 
                            peak_amplitude > absthresh_val and
                            peak_prominence > absthresh_val/2)
            duration_ok = (peak_duration_ms >= spkdur[0] and 
                            peak_duration_ms <= spkdur[1])
            not_too_high = peak_amplitude < too_high_abs
            
            # Additional IED-specific criteria
            sharpness_ok = peak_prominence > peak_amplitude * 0.3  # Sharp transients
            
            if amplitude_ok and duration_ok and not_too_high and sharpness_ok:
                
                all_spikes.append([peak_time, peak_width_samples, 
                                        peak_amplitude, peak_prominence])
    

    if len(all_spikes) == 0:
        return np.array([]).reshape(0, 5)
    
    gdf = np.array(all_spikes)
    
    #Remove spikes too clsoe to edges
    close_idx = close_to_edge * fs
    mask = (gdf[:, 1] >= close_idx) & (gdf[:, 1] <= nsamples - close_idx)
    gdf = gdf[mask]
    
    # Remove duplicates and sort by time
    if gdf.size > 0:
        
        gdf = gdf[gdf[:, 1].argsort()]
        
        keep = np.ones(gdf.shape[0], dtype=bool)
        
        for i in range(1, len(gdf)):
            time_diff = abs(gdf[i,1] - gdf[i-1, 1])
            
            if time_diff < 100e-3 * fs:
                if gdf[i, 3] < gdf[i-1, 3]:
                    keep[i] = False
                else:
                    keep[i-1] = False
        
        gdf = gdf[keep]
    
    return gdf



# def find_peaks(s):
#     s = np.asarray(s)
#     if s.ndim != 1:
#         raise ValueError(f"Input must be 1D array, got shape {s.shape}")
#     ds = np.diff(s)
    
#     ds = np.concatenate([[ds[0]], ds])
#     filter_idx = np.where(ds[1:] == 0)[0] + 1
    
#     if len(filter_idx) > 0:
#         for idx in filter_idx:
#             if idx > 0:
#                 ds[idx] = ds[idx-1]
#             else:
#                 nonzero_idx = np.where(ds != 0)[0]
#                 if len(nonzero_idx) > 0:
#                     ds[idx] = ds[nonzero_idx[0]]
#     ds = np.sign(ds)
    
#     ds = np.diff(ds)
    
#     t = np.where(ds > 0)[0]
#     p = np.where(ds < 0)[0]
    
#     return p, t
    
    
    
    

#output eeg plot with flags on targetted spikes

if __name__ == "__main__":
    main()
        
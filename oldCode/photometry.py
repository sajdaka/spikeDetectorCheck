import sys
import pandas as pd
import numpy as np
from numpy.polynomial import Polynomial
import os
import scipy
import plotly
from data_import import import_ppd
from scipy import stats, signal, optimize
from scipy.stats import linregress, theilslopes
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt, savgol_filter
from scipy.interpolate import interp1d

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from sklearn.linear_model import TheilSenRegressor

from open_ephys.analysis import Session

from alignment import alignment_start


def main():
    pio.renderers.default = 'browser'

    # requires three command line inputs (programName, Chadni(0) vs Meiling(1), initial denoise(0) vs detrend(1), photometry fileName, EEG filename)
    if(len(sys.argv) != 2):
        if(len(sys.argv) != 4 and len(sys.argv) != 5):
            print("incorrect usage of program | photometry.py, (Chandni/Meiling)[0/1], initial phase (denoise/detrend)[0/1], [photometryFilename], (optional)[EEGFilename | photometry.py [EEGFilename]")
            exit()
        else:
            EEG_only = 1
        
    Meiling = False
    Chandni = False
    #check the preprocessing selected plan
    if(int(sys.argv[1]) == 0):
        Chandni = True
    elif (int(sys.argv[1]) == 1):
        Meiling = True
    else:
        print("Undefined preprocessing algorithm selected. \n Please choose 0 (Chandni) or 1 (Meiling)")
        exit()

    if(int(sys.argv[2]) != 0 and int(sys.argv[2]) != 1):
        print("Undefined preprocessing initial step. \n Please choose 0 (denoising) or 1 (detrending)")
        exit()
        
    #opening selected photometry file and retrieving data
    directory = ""
    directory = os.path.join('./FiberPhotometryData/GRABne', sys.argv[3])
    try:
            data_ppd = import_ppd(directory)
    except FileNotFoundError:
            print(f"There is no file named {sys.argv[3]} in the directory FiberPhotometryData")
            exit()
        
    #EEG preprocessing read in and parsing
    EEG_preprocessing_check = 1 if len(sys.argv) == 5 else 0
    EEG = []
    EEG_filtered = []
    if(EEG_preprocessing_check):
       
        directory = os.path.join('./EEGData/Meiling_EEG', sys.argv[4])
        try:
            session = Session(directory)
        except FileNotFoundError:
            print(f"There is no file named {sys.argv[4]} in the directory EEGData")
            exit()
            

    
        #specify channel number 
        if(Meiling):
            chEEG = 2
        elif(Chandni):
            chEEG = 2

        ppd_aligned, iso_aligned, EEG, time_second = alignment_start(session, chEEG, data_ppd)
        
        
        EEG_filtered = EEG_preprocessing(EEG)
       

            
    if(Meiling):
        GRABne_raw = ppd_aligned
        Isos_raw = iso_aligned
        sampling_rate = data_ppd['sampling_rate']
        if(int(sys.argv[2]) == 0):
           GRABne_denoised, Isos_denoised = MeilingDenoising(GRABne_raw, Isos_raw, sampling_rate)
           GRABne_detrended, GRABne_expfit, GRABne_est_motion, Isos_detrended = MeilingDetrended(GRABne_denoised, Isos_denoised, time_second)
           GRABne_dF_F_precorrected, GRABne_dF_F_corrected, z_GRABne_dF_F_corrected = MeilingPreprocessing(GRABne_detrended, GRABne_expfit, GRABne_est_motion)
           printPlots(GRABne_raw, Isos_raw, GRABne_denoised, GRABne_dF_F_corrected, z_GRABne_dF_F_corrected, EEG, EEG_filtered, time_second, EEG_preprocessing_check, 1)
        elif(int(sys.argv[2]) == 1):
            GRABne_detrended, GRABne_expfit, GRABne_est_motion, Isos_detrended = MeilingDetrended(GRABne_raw, Isos_raw, time_second)
            GRABne_denoised, Isos_denoised = MeilingDenoising(GRABne_detrended, Isos_detrended, sampling_rate)
            GRABne_dF_F_precorrected, GRABne_dF_F_corrected, z_GRABne_dF_F_corrected = MeilingPreprocessing(GRABne_denoised, GRABne_expfit, GRABne_est_motion)
            printPlots(GRABne_raw, Isos_raw, GRABne_detrended, GRABne_dF_F_corrected, z_GRABne_dF_F_corrected, EEG, EEG_filtered, time_second, EEG_preprocessing_check, 2)
    elif(Chandni):
        GCAMP_raw = ppd_aligned
        Isos_raw = iso_aligned
        sampling_rate = data_ppd['sampling_rate']
        GCAMP_dF_F_preprocessed, GCAMP_dF_F_processed, z_GCAMP_dF_F_processed = ChandniPreprocessing(GCAMP_raw, Isos_raw)
        printPlots(GCAMP_raw, Isos_raw, GCAMP_dF_F_preprocessed, GCAMP_dF_F_processed, z_GCAMP_dF_F_processed, EEG, EEG_filtered, time_second, EEG_preprocessing_check, 3)
       
       
def ChandniPreprocessing(GCAMP_raw, Isos_raw):
    #create gaussian kernel using the standard sigma of 75 *does this need to be a user input*
    #baseline = savgol_filter(Isos_raw, 301, 4)
    
    #GCAMP_raw = GCAMP_raw - (Isos_raw - baseline)
    
    sigma = 75
    kernelSize = sigma * 3
    x =  np.arange(-kernelSize, kernelSize+1)
    gaussianKernel = np.exp(-x**2/ (2* sigma**2))
    gaussianKernel = gaussianKernel / np.sum(gaussianKernel)
    
    GCAMP_filt = filtfilt(gaussianKernel, 1, GCAMP_raw)
    Isos_filt = filtfilt(gaussianKernel, 1, Isos_raw)
    
    GCAMP_dF_F_preprocessed = (GCAMP_raw-Isos_raw)/Isos_raw
    GCAMP_dF_F_processed = (GCAMP_filt-Isos_filt)/Isos_filt
    z_GCAMP_dF_F_processed = stats.zscore(GCAMP_dF_F_processed)
    return GCAMP_dF_F_preprocessed, GCAMP_dF_F_processed, z_GCAMP_dF_F_processed
    
    
    #optional EEG and photometry preprocessing REQUIRES: alignment
    
        
    
def MeilingDenoising(GRABne, Isos, sampling_rate):
    #photobleaching correction
    #lowpass filter for initial generalized denoising
    #baseline = savgol_filter(Isos, 301, 4)
    
    #GRABne = GRABne - (Isos - baseline)
    x, y = butter(4, 10, btype='low', fs=sampling_rate)
    GRABne_denoised = filtfilt(x,y, GRABne)
    Isos_denoised = filtfilt(x,y, Isos)
    return GRABne_denoised, Isos_denoised

def MeilingDetrended(GRABne, Isos, time_second, ):
    #

    
    #double exponential curve for analog 1
    max_sig = np.max(GRABne)
    initial_params = [max_sig/2, max_sig/4, max_sig/4, 3600, 0.1]
    bounds = ([0, 0, 0, 600, 0], [max_sig, max_sig, max_sig, 36000, 1])
    GRABne_params, parm_cov = curve_fit(double_exponential, time_second, GRABne,
                                        p0=initial_params, bounds=bounds, maxfev=1000)
    GRABNE_expfit = double_exponential(time_second, *GRABne_params)
    
    #double exponential curve for analog 2 
    max_sig = np.max(Isos)
    initial_params = [max_sig/2, max_sig/4, max_sig/4, 3600, 0.1]
    bounds = ([0, 0, 0, 600, 0], [max_sig, max_sig, max_sig, 36000, 1])
    Isos_parms, parm_cov = curve_fit(double_exponential, time_second, Isos,
                                     p0=initial_params, bounds=bounds, maxfev=1000)
    Isos_expfit = double_exponential(time_second, *Isos_parms)
    
    #motion correction
    GRABne_detrended = GRABne - GRABNE_expfit
    Isos_detrended = Isos - Isos_expfit
    
    
    slope, intercept, rvalue, pvalue, stderr = linregress(x=Isos_detrended, y=GRABne_detrended)
    #Isos_thielsen = Isos_detrended.reshape(-1, 1)
   
    #TheilSen = TheilSenRegressor().fit(Isos_thielsen, GRABne_detrended)
    #baseline = savgol_filter(Isos_detrended, 301, 4)
    
    #GRABne_detrended = GRABne_detrended - (Isos_detrended - baseline)
    
    GRABne_est_motion = slope * Isos_detrended + intercept
    
    return GRABne_detrended, GRABNE_expfit, GRABne_est_motion, Isos_detrended
    
    
def MeilingPreprocessing(GRABne_cleaned, GRABNE_expfit, GRABne_est_motion):

    GRABne_corrected = GRABne_cleaned - GRABne_est_motion #can be changed to GRABne_detrended/GRABne_est_motion
    GRABne_dF_F_precorrected = GRABne_cleaned/GRABNE_expfit
    
    GRABne_dF_F_corrected = GRABne_corrected/GRABNE_expfit
    z_GRABne_dF_F_corrected = stats.zscore(GRABne_dF_F_corrected)
    return GRABne_dF_F_precorrected, GRABne_dF_F_corrected, z_GRABne_dF_F_corrected
    
    

def double_exponential(t, const, amp_fast, amp_slow, tau_slow, tau_multiplier):
    tau_fast = tau_slow * tau_multiplier
    return const+amp_slow*np.exp(-t/tau_slow) + amp_fast*np.exp(-t/tau_fast)

def EEG_preprocessing(EEG):
    sampling_freq = 1000
    f0 = 60.0 #frequency that will be cleaned from the signal
    
    #iirnotch pass filter
    b, a = signal.iirnotch(f0, Q=30, fs=sampling_freq)
    EEG_notch_filtered = signal.filtfilt(b, a, EEG)
    
    #butterworth filtering
    
    lowcut, highcut = 1, 70
    nyquist = .5 * sampling_freq
    low = lowcut/nyquist
    high = highcut /nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    EEG_filtered = signal.filtfilt(b, a, EEG_notch_filtered)
    return EEG_filtered

def EEG_preprocessing(EEG, threshold):
    sampling_freq = 1000
    f0 = 60.0 #frequency that will be cleaned from the signal
    
    #iirnotch pass filter
    b, a = signal.iirnotch(f0, Q=30, fs=sampling_freq)
    EEG_notch_filtered = signal.filtfilt(b, a, EEG)
    
    EEG_interpolated, _ = EEGInterpolation(EEG_notch_filtered, threshold)
    #butterworth filtering
    
    lowcut, highcut = 1, 70
    nyquist = .5 * sampling_freq
    low = lowcut/nyquist
    high = highcut /nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    EEG_filtered = signal.filtfilt(b, a, EEG_interpolated)
    return EEG_filtered
    
def EEGInterpolation(EEG, threshold):
    
    cleaned_signal = EEG.copy()
    artifact_mask = np.abs(EEG) > threshold
    artifactidx = np.where(artifact_mask)[0]
    n_samples = len(EEG)
    
    if len(artifactidx) == 0:
        return EEG, artifactidx
    
    start = artifactidx[0]
    end = artifactidx[0]
    artifactarea = []
    for i in range(1, len(artifactidx)):
        if artifactidx[i] == end + 1:
            end = artifactidx[i]
        else:
            artifactarea.append((start, end))
            start = artifactidx[i]
            end = artifactidx[i]
            
    artifactarea.append((start, end))
    
    interpolated_areas = []
    
    for start, end in artifactarea:
        
        interp_start = max(0, start - 10)
        interp_end = min(n_samples - 1, end + 10)
        
        artifact_span = np.arange(interp_start, interp_end + 1)
        
        before_start = max(0, interp_start - 20)
        before_end = interp_start - 1
        after_start = interp_end + 1
        after_end = min(n_samples -1, interp_end + 20)
        
        if before_end >= before_start and after_end >= after_start:

            
            time_before = np.arange(before_start, before_end + 1)
            time_after = np.arange(after_start, after_end + 1)
            x_good = np.concatenate([time_before, time_after])
            y_good = np.concatenate([EEG[time_before], EEG[time_after]])
            
            if len(x_good) > 1:
                interpolator = interp1d(x_good, y_good, kind='linear', fill_value='extrapolate')
                
                cleaned_signal[artifact_span] = interpolator(artifact_span)
                interpolated_areas.append((interp_start, interp_end))
        
        return cleaned_signal, interpolated_areas
        
def baselineZscore(data, startBaseline, endBaseline, fs):
    
    startTime = int(startBaseline * fs)
    endTime = int(endBaseline * fs)
    baseline = data[startTime:endTime]
    baseline_mean = baseline.mean()
    baseline_std = baseline.std()
    
    Zscore = (data - baseline_mean) / baseline_std
    return Zscore
    
def fullZscore(data):
    data_mean = data.mean()
    data_std = data.std()
    
    Zscore = (data-data_mean) /data_std
    return Zscore   
    
def printPlots(photoRaw, IsosRaw, photoPrecorrected, photoCorrected, photo_z_score,
               EEG_raw, EEG_filtered, time_seconds, EEG_preprocessing_check, title):
    
    if len(time_seconds) > 10000:
        step = len(time_seconds) // 5000  # Keep ~5000 points
        time_seconds = time_seconds[::step]
        photoRaw = photoRaw[::step]
        IsosRaw = IsosRaw[::step]
        photoPrecorrected = photoPrecorrected[::step]
        photoCorrected = photoCorrected[::step]
        photo_z_score = photo_z_score[::step]
        if EEG_preprocessing_check:
            EEG_raw = EEG_raw[::step]
            EEG_filtered = EEG_filtered[::step]
    
    if(EEG_preprocessing_check):
        fig = make_subplots(rows=4, cols=2)
    else:
        fig = make_subplots(rows=4, cols=1)

    
    fig.add_trace(
        go.Scatter(
            x=time_seconds,
            y=photoRaw,
            mode='lines',
            line_shape='spline',
            name="Time vs. Raw Photometry"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=time_seconds,
            y=IsosRaw,
            mode='lines',
            line_shape='spline',
            name="Time vs. Raw Isos"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=time_seconds,
            y=photoPrecorrected,
            mode='lines',
            line_shape='spline',
            name="Time vs. Intermediate Processed Photometry",
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=time_seconds,
            y=photoCorrected,
            mode='lines',
            line_shape='spline',
            name="Time vs. Preprocessed Photometry",
        ),
        row=3,col=1
    )
    fig.add_trace(
        go.Scatter(
            x=time_seconds,
            y=photo_z_score,
            mode='lines',
            line_shape='spline',
            name="Time vs Z-Score Photometry"
        ),
        row=4,col=1
    )
    if (EEG_preprocessing_check):
        fig.add_trace(
            go.Scatter(
                x=time_seconds,
                y=EEG_raw,
                mode='lines',
                line_shape='spline',
                name="Time vs Raw EEG",
            ),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=time_seconds,
                y=EEG_filtered,
                mode='lines',
                line_shape='spline',
                name="Time vs Processed EEG",
            ),
            row=2, col=2
        )
    if(title == 3):
        fig.update_layout(title=f"Preprocessing pipeline for {sys.argv[3]} using Chandni's process")
    elif(title == 1):
        fig.update_layout(title=f"Preprocessing pipeline for {sys.argv[3]} using Meiling's process, initially denoising")
    elif(title == 2):
        fig.update_layout(title=f"Preprocessing pipeline for {sys.argv[3]} using Meiling's process, initially detrending")
    fig.show()
    
def getEEGFiltered(EEG):
    EEG = EEG_preprocessing(EEG)
    return EEG
    
    
    
if __name__ == "__main__":
    main()
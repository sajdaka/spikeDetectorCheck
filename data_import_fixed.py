import os
import json
import pylab as plt
import numpy as np
from scipy.signal import butter, filtfilt, medfilt, decimate
from scipy.optimize import curve_fit
from scipy.stats import linregress, zscore

def import_ppd(file_path, low_pass=20, high_pass= 0.01):
    
    with open(file_path, 'rb') as f:
        header_sz = int.from_bytes(f.read(2), 'little')
        data_header = f.read(header_sz)
        data = np.frombuffer(f.read(), dtype=np.dtype('<u2'))
        
    header_dict = json.loads(data_header)
    volts_per_division = header_dict['volts_per_division']
    sampling_rate = header_dict['sampling_rate']
    
    analog = data >> 1
    digital = ((data & 1) == 1).astype(int)
    
    if 'n_analog_signals' in header_dict.keys():
        n_analog_signals = header_dict['n_analog_signals']
        n_digital_signals = header_dict['n_digital_signals']
    else:
        n_analog_signals = 2
        n_digital_signals = 2
    
    analog_1 = analog[::]

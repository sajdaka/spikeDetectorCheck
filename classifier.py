from sklearn import linear_model
import numpy as np
import pandas as pd
import csv
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from DataPreprocessing import EEGPreprocessor
from dataLoad import DataLoader

class Classifier:
    
    def __init__(self, spikes: np.ndarray, training_eeg: np.ndarray, test_eeg: np.ndarray, z_score: float, training_data_file: str):
        self.spikes = spikes
        self.training_eeg = training_eeg
        self.test_eeg = test_eeg
        self.z_score = z_score
        self.training_data_file = training_data_file
        
    def read_training_data(self):
        labels = []
        training_data = []
        
        with open(self.training_data_file, 'r') as f:
            reader = csv.DictReader(f)
            labels = [row['Class'] for row in reader]
            training_data = [row['Time'] for row in reader]
            
        self.training_data = np.array(training_data)
        self.labels = np.array(labels)
        
        
    def _find_spike_shape(self, spike_data: np.ndarray, eeg_data: np.ndarray) -> np.ndarray:
        spike_shapes = []
        
        for spike in spike_data:
            spike_shape = []
            for i in range(-100, 100, 20):
                spike_shape.append(np.mean(eeg_data[spike + i: spike +i + 20]))
            spike_shapes.append(spike_shape)
        
        return spike_shapes  
    
    def _find_spike_width(self, spike_data: np.ndarray, eeg_data: np.ndarray) ->np.ndarray:
        spike_widths = []
        
        for spike in spike_data:
            spike_segment = eeg_data[spike - 100: spike + 100]
            midpoint = eeg_data[spike]/2
            first, second = -1
            for i in spike_segment:
                if i == midpoint and first == -1:
                    first = i
                elif i == midpoint and second == -1:
                    second = i
            
            spike_widths.append(second - first)
        
    def _find_spike_zband(self, spike_data: np.ndarray, eeg_data: np.ndarray, z_score: float) -> np.ndarray:
        spike_zbands = []
         
        for spike in spike_data:
            above_z = 0
            spike_segment = eeg_data[spike - 100 : spike + 100]
            for i in spike_segment:
                if i >= z_score:
                    above_z += 1
            spike_zbands.append(above_z/200)
            
        return spike_zbands
            
    def _find_max_slope(self, spike_shapes: np.ndarray) -> np.ndarray:
        spike_max_slope = []
        
        for spike in spike_shapes:
            spike_slopes = []
            for i in enumerate(spike) - 1:
                spike_slopes.append((spike[i+1] - spike[i])/20)
            spike_max_slope.append(np.max(spike_slopes))
            
        return spike_max_slope
    
    def _find_min_slope(self, spike_shapes: np.ndarray) -> np.ndarray:
        spike_min_slope = []
        
        for spike in spike_shapes:
            spike_slopes = []
            for i in enumerate(spike) - 1:
                spike_slopes.append((spike[i+1] - spike[i])/20)
            spike_min_slope.append(np.min(spike_slopes))
            
        return spike_min_slope
    
    def _find_min_to_max(self, spike_data: np.ndarray, eeg_data: np.ndarray) -> np.ndarray:
        min_to_max = []
        
        for spike in spike_data:
            spike_segment = eeg_data[spike - 100: spike + 100]
            val = np.abs(np.max(spike_segment) - np.min(spike_segment))
            min_to_max.append(val)
        
        return min_to_max
        
    def build_features(self, data: np.ndarray, eeg: np.ndarray) -> np.ndarray:
        features = []
        
        spike_shapes = self._find_spike_shape(data, eeg)
        spike_width = self._find_spike_width(data, eeg)
        spike_zband = self._find_spike_zband(data, eeg)
        spike_max_slope = self._find_max_slope(spike_shapes)
        spike_min_slope = self._find_min_slope(spike_shapes)
        min_to_max = self._find_min_to_max(data, eeg)
        
        for i in enumerate(self.training_data):
            features[i] = spike_shapes[i]
            features[i].append(spike_width[i])
            features[i].append(spike_zband[i])
            features[i].append(spike_max_slope[i])
            features[i].append(spike_min_slope[i])
            features[i].append(min_to_max[i])
        
        return features
        
    def init_logistical_regression(self):
        training_features = self.build_features(self.training_data, self.test_eeg)
        self.model = linear_model.LogisticRegression()
        self.model.fit(training_features, self.labels)
        
    def run_logistical_regression(self):
        test_features = self.build_features(self.spikes, self.test_eeg)
        self.predictions = self.model.predict(test_features)
        print(self.predictions)
        self.probabilities = self.model.predict_proba(test_features)
        print(self.probabilities)
    
    def visualize_logistic_regression(self):
            y_prob = self.probabilities[:, 1]
            plt.figure(figsize=(10, 6))
            plt.hist(y_prob, bins=30, edgecolor='black', alpha=0.7)
            plt.xlabel('Predicted Probability of Class 1')
            plt.ylabel('Frequency')
            plt.title('Distribution of Predicted Probabilities')
            plt.show()
    
        
def classifier_start(spikes: np.ndarray, test_eeg: np.ndarray, z_score: float):
    eegLoader = DataLoader()
    raw_eeg = eegLoader.load("2024-06-05_11-35-09_1938_seizure1_GRAB")
    #use the gui preprocessor somehow potentially do that before classification in the master file
    classifier_model = Classifier(spikes, "training eeg", test_eeg, z_score, "training data file")
    classifier_model.read_training_data()
    classifier_model.init_logistical_regression()
    classifier_model.run_logistical_regression()
        

if __name__ is '__main__':
    classifier_start()
        
        
        
        

        
         
        

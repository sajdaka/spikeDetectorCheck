from sklearn import linear_model
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score, train_test_split
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
import csv
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from DataPreprocessing import EEGPreprocessor
from SpikeDetection import BaselineNormalizer, SpikeEvent
from dataLoad import DataLoader, TrainingDataRecord
import logging

logger = logging.getLogger(__name__)

class SpikeClassifier:
    
    def __init__(self, spikes: np.ndarray, training_eeg: np.ndarray, test_eeg: np.ndarray, training_data: TrainingDataRecord, eeg_zscore: float,
                 eeg_training_zscore: float, fs: float = 1000.0, decision_threshold=0.5):
        self.spikes = spikes
        self.spikes_times = [spike.time_samples for spike in spikes]
        self.training_eeg = training_eeg
        self.test_eeg = test_eeg
        self.training_data_times = training_data.timestamps
        self.labels = training_data.spikeIndicators
        self.eeg_zscore = eeg_zscore
        self.eeg_training_zscore = eeg_training_zscore
        self.fs = fs
        self.decision_threshold = decision_threshold
        
    def read_training_data(self):
        labels = []
        training_data = []
        
        with open(self.training_data_file, 'r') as f:
            reader = csv.DictReader(f)
            labels = [row['Class'] for row in reader]
            training_data = [row['Time'] for row in reader]
            
        self.training_data = np.array(training_data)
        self.labels = np.array(labels)
        
    def _get_normalized_spike_segment(self, spike_time: float, eeg_data: np.ndarray, window_before: int = 100, window_after: int = 100) -> np.ndarray:
        start_idx = int(spike_time - window_before)
        end_idx = int(spike_time + window_after)
        
        segment = eeg_data[start_idx:end_idx].copy()
        
        if(np.abs(np.min(segment)) > np.abs(np.max(segment))):
            segment = -segment
            
        return segment
        
        
    def _find_spike_shape(self, spike_data: np.ndarray, eeg_data: np.ndarray) -> np.ndarray:
        spike_shapes = []
        
        for spike in spike_data:
            segment = self._get_normalized_spike_segment(spike, eeg_data)
            
            spike_shape = []
            
            for i in range(0, len(segment), 20):
                if i + 20 <= len(segment):
                    spike_shape.append(np.mean(segment[i:i+20]))
                else:
                    spike_shape.append(np.mean(segment[i:]))
            spike_shapes.append(spike_shape)
        
        return spike_shapes  
    
    def _find_spike_width(self, spike_data: np.ndarray, eeg_data: np.ndarray) -> np.ndarray:
        spike_widths = []
        
        for spike in spike_data:
            segment = self._get_normalized_spike_segment(spike, eeg_data)

            if len(segment) < 200:
                spike_widths.append(100)
                continue
            
            center_start = 50
            center_end = 150
            center_segment = segment[center_start:center_end]
            peak_value = np.max(np.abs(center_segment))
            peak_idx_local = np.argmax(np.abs(center_segment))
            peak_idx = center_start + peak_idx_local
            
            half_max = peak_value /2
            
            left_idx = None
            for i in range(peak_idx, 0, -1):
                if np.abs(segment[i]) <= half_max:
                    left_idx = i
                    break
                
            right_idx = None
            for i in range(peak_idx, len(segment)):
                if np.abs(segment[i]) <= half_max:
                    right_idx = i
                    break
            
            if left_idx is not None and right_idx is not None:
                width = right_idx - left_idx
            else:
                width = 50
            
            spike_widths.append(width)
            
            
        return spike_widths
        
    def _find_spike_zband(self, spike_data: np.ndarray, eeg_data: np.ndarray, z_score_array: np.ndarray) -> np.ndarray:
        spike_zbands = []
         
        for spike in spike_data:
            above_z = 0
            spike_segment = z_score_array[int(spike - 100) : int(spike + 100)]
            for i in spike_segment:
                if i >= 1 or i <= -1:
                    above_z += 1
            spike_zbands.append(above_z/200)
            
        return spike_zbands
            
    def _find_max_slope(self, spike_shapes: np.ndarray) -> np.ndarray:
        spike_max_slope = []
        
        for spike in spike_shapes:
            spike_slopes = []
            for i in range(len(spike) -1 ):
                spike_slopes.append((spike[int(i+1)] - spike[int(i)])/20)
            spike_max_slope.append(np.max(np.abs(spike_slopes)))
            
        return spike_max_slope
    
    def _find_min_slope(self, spike_shapes: np.ndarray) -> np.ndarray:
        spike_min_slope = []
        
        for spike in spike_shapes:
            spike_slopes = []
            for i in range(len(spike) - 1):
                spike_slopes.append((spike[i+1] - spike[i])/20)
            spike_min_slope.append(np.min(np.abs(spike_slopes)))
            
        return spike_min_slope
    
    def _find_min_to_max(self, spike_data: np.ndarray, eeg_data: np.ndarray) -> np.ndarray:
        min_to_max = []
        
        for spike in spike_data:
            spike_segment = self._get_normalized_spike_segment(spike, eeg_data)
            val = np.abs(np.max(spike_segment) - np.min(spike_segment))
            min_to_max.append(val)
        
        return min_to_max
    
    def _find_frequency_features(self, spike_data: np.ndarray, eeg_data: np.ndarray) -> dict:
        features = {'peak_freq': [], 'spectral_entropy': [], 'power_ratio': []}
        
        for spike in spike_data:
            segment = self._get_normalized_spike_segment(spike, eeg_data)
            
            freqs = np.fft.rfftfreq(len(segment), 1/self.fs)
            fft_vals = np.abs(np.fft.rfft(segment))
            
            features['peak_freq'].append(freqs[np.argmax(fft_vals[1:])] + freqs[1])
            
            psd = fft_vals ** 2
            psd_norm = psd / np.sum(psd)
            features ['spectral_entropy'].append(-np.sum(psd_norm * np.log2(psd_norm + 1e-10)))
            
            low_power = np.sum(fft_vals[(freqs >= 1) & (freqs < 10)])
            high_power = np.sum(fft_vals[(freqs >= 15) & (freqs < 40)])
            features['power_ratio'].append(high_power / (low_power + 1e-10))
        
        return features
    
    def _find_symmetry_features(self, spike_data: np.ndarray, eeg_data: np.ndarray) -> dict:
        features = {'symmetry': [], 'sharpness': [], 'rise_fall_ratio': []}
        
        for spike in spike_data:
            segment = eeg_data[int(spike-100): int(spike+100)]
            
            peak_idx = np.argmax(np.abs(segment))
            peak_value = np.abs(segment[peak_idx])
            
            
            left_area = np.sum(np.abs(segment[:peak_idx]))
            right_area = np.sum(np.abs(segment[peak_idx:]))
            features['symmetry'].append((left_area - right_area) / (left_area + right_area + 1e-10))
            
            if peak_idx > 1 and peak_idx < len(segment) - 1:
                second_deriv = segment[peak_idx+1] - 2*segment[peak_idx] + segment[peak_idx-1]
                features['sharpness'].append(np.abs(second_deriv))
            else:
                features['sharpness'].append(0)
            
            half_max = peak_value / 2

            rise_start_idx = None
            for i in range(peak_idx, 0, -1):
                if np.abs(segment[i]) <= half_max:
                    rise_start_idx = i
                    break
                
            if rise_start_idx is not None:
                rise_time = peak_idx - rise_start_idx
            else:
                rise_time = peak_idx
                
            fall_end_idx = None
            for i in range(peak_idx, len(segment)):
                if np.abs(segment[i]) <= half_max:
                    fall_end_idx = i
                    break
                
            if fall_end_idx is not None:
                fall_time = fall_end_idx - peak_idx
            else:
                fall_time = len(segment) - peak_idx
                
            rise_fall_ratio = rise_time / (fall_time + 1e-10)
            features['rise_fall_ratio'].append(rise_fall_ratio)
            
        return features
        
    def build_features(self, data: np.ndarray, eeg: np.ndarray, z_score: float) -> np.ndarray:
        features = [[] for i in range(len(data))]
        
        spike_shapes = self._find_spike_shape(data, eeg)
        spike_width = self._find_spike_width(data, eeg)
        spike_zband = self._find_spike_zband(data, eeg, z_score)
        spike_max_slope = self._find_max_slope(spike_shapes)
        spike_min_slope = self._find_min_slope(spike_shapes)
        min_to_max = self._find_min_to_max(data, eeg)
        frequency_features = self._find_frequency_features(data, eeg)
        symmetry_features = self._find_symmetry_features(data, eeg)
        
        for i, _ in enumerate(data):
            features[i] = spike_shapes[i]
            features[i].append(spike_width[i])
            features[i].append(spike_zband[i])
            features[i].append(spike_max_slope[i])
            features[i].append(spike_min_slope[i])
            features[i].append(min_to_max[i])
            features[i].append(frequency_features['peak_freq'][i])
            features[i].append(frequency_features['spectral_entropy'][i])
            features[i].append(frequency_features['power_ratio'][i])
            features[i].append(symmetry_features['symmetry'][i])
            features[i].append(symmetry_features['sharpness'][i])
            features[i].append(symmetry_features['rise_fall_ratio'][i])
        
        return features
        
    def init_logistical_regression(self):
        training_features = self.build_features(self.training_data_times, self.training_eeg, self.eeg_training_zscore)
        training_features = np.array(training_features)
        
        #for testing and evaluating that my model actually works can remove soon
        X_train, X_val, y_train, y_val = train_test_split(
            training_features, self.labels,
            test_size=0.2, random_state=42, stratify=self.labels
        )
        logger.info(f"Using train/val split:  {len(X_train)}/{len(X_val)}")
        
        class_counts = np.bincount(self.labels)
        logger.info(f"Class distribution: {class_counts}")
        imbalance_ratio = class_counts[0] / class_counts[1] if class_counts[1] > 0 else np.inf
        
        if imbalance_ratio > 3:
            smote = SMOTE(random_state=42)
            training_features, labels = smote.fit_resample(training_features, self.labels)
        else:
            labels = self.labels
            
            
        self.mean = np.mean(training_features, axis=0)
        self.std = np.std(training_features, axis=0) + 1e-10
        training_features = (training_features - self.mean) / self.std
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        models = {
            'LogisticRegression': linear_model.LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=42
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            )
        }
        
        param_grids = {
            'LogisticRegression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            },
            'RandomForest': {
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        }
        
        best_score = 0
        best_model = None
        for model_name, model in models.items():
            logger.info(f"Training {model_name}...")
            
            if model_name in param_grids:
                grid_search = GridSearchCV(
                    model,
                    param_grids[model_name],
                    cv=cv,
                    scoring='f1',
                    n_jobs=-1
                )
                grid_search.fit(training_features, labels)
                if grid_search.best_score_ > best_score:
                    best_score = grid_search.best_score_
                    best_model = grid_search.best_estimator_
                    self.best_model_name = model_name
                logger.info(f"Best params: {grid_search.best_params_}")
                logger.info(f"Best CV score: {grid_search.best_score_:.3f}")
                
            else:
                scores = cross_val_score(model, training_features, labels, cv=cv, scoring='f1')
                avg_score = np.mean(scores)
                
                if avg_score > best_score:
                    self.best_score = avg_score
                    best_model = model.fit(training_features, labels)
                    self.best_model_name = model_name
                    
                logger.info(f"CV score: {avg_score:.3f} (+/- {np.std(scores):.3f})")
        
        self.model = best_model
        logger.info(F"Best model: {self.best_model_name} with score {self.best_score:.3f}")
        plt.figure()
        plt.imshow(training_features[np.argsort(self.labels),:])
        plt.savefig("features.png", format='png')
        plt.close()
        if X_val is not None:
            self.X_val = X_val
            self.y_val = y_val
            self.evaluate_model(X_val, y_val)
        
    def run_logistical_regression(self):
        test_features = self.build_features(self.spikes_times, self.test_eeg, self.eeg_zscore)
        test_features = (test_features - self.mean) / self.std
        self.probabilities = self.model.predict_proba(test_features)
        logger.info(f"probabilities: {self.probabilities}")
        self.predictions = (self.probabilities[:, 1] >= self.decision_threshold).astype(int)
        logger.info(f"predictions: {self.predictions}")
        
    def evaluate_model(self, test_features, test_labels):
        
        test_features_norm = (test_features - self.mean) / self.std
        
        
        probabilities = self.model.predict_proba(test_features_norm)[:, 1]
        predictions = (probabilities >= self.decision_threshold).astype(int)
        logger.info("="*50)
        logger.info("MODEL EVALUATION")
        logger.info("="*50)
        logger.info(classification_report(test_labels, predictions,
                                    target_names=['Artifact', 'True Spike']))

        logger.info("Confusion Matrix:")
        cm = confusion_matrix(test_labels, predictions)
        logger.info(cm)
        logger.info(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
        logger.info(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")

        if len(np.unique(test_labels)) == 2:
            auc = roc_auc_score(test_labels, probabilities)
            logger.info(f"ROC-AUC Score: {auc:.3f}")

            # Plot precision-recall curve
            precision, recall, thresholds = precision_recall_curve(test_labels, probabilities)

            plt.figure(figsize=(10, 6))
            plt.plot(recall, precision)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.grid(True)
            plt.savefig('precision_recall_curve.png', dpi=300, bbox_inches='tight')
            plt.close()  
            
        
    def visualize_logistic_regression(self):
            y_prob = self.probabilities[:, 1]
            plt.figure(figsize=(10, 6))
            plt.hist(y_prob, bins=30, edgecolor='black', alpha=0.7)
            plt.xlabel('Predicted Probability of Class 1')
            plt.ylabel('Frequency')
            plt.title('Distribution of Predicted Probabilities')
            plt.savefig('probabilities_distribution.png', dpi=300, bbox_inches = 'tight')
            plt.close()
            logger.info("Plot saved to probabilities_distribution.png")
            
    def get_predicted_spikes(self) -> np.ndarray:
        final_spikes = []
        for idx, prediction in enumerate(self.predictions):
            if prediction == 1:
                final_spikes.append(self.spikes[idx])
        return final_spikes
    

        
def classifier_start(spikes: np.ndarray, test_eeg: np.ndarray, training_eeg: np.ndarray, training_data: TrainingDataRecord, eeg_zscore: float, eeg_training_zscore: float,
                     decision_threshold: float = 0.5):
    #use the gui preprocessor somehow potentially do that before classification in the master file
    classifier_model = SpikeClassifier(spikes, training_eeg, test_eeg, training_data, eeg_zscore, eeg_training_zscore, decision_threshold=decision_threshold)
    classifier_model.init_logistical_regression()
    
    classifier_model.run_logistical_regression()
    classifier_model.visualize_logistic_regression()
    return classifier_model.get_predicted_spikes()
    
        

if __name__ == '__main__':
    classifier_start()
        
        
        
        

        
         
        

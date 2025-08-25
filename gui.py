

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
from typing import Dict, Any, Optional, List
import threading
import queue
import logging
from dataclasses import asdict
import numpy as np


from config import ConfigManager
from dataLoad import DataManager
from DataPreprocessing import PreprocessingPipeline
from SpikeDetection import SpikeDetector, SpikeDetectionParams
from visualization import InteractivePlotter
from alignment import SignalAlignment

logger = logging.getLogger(__name__)


class ParameterFrame(ttk.Frame):

    
    def __init__(self, parent, config_manager: ConfigManager):
        super().__init__(parent)
        self.config_manager = config_manager
        self.param_vars = {}
        self.setup_ui()
    
    def setup_ui(self):

        notebook = ttk.Notebook(self)
        notebook.pack(fill='both', expand=True, padx=5, pady=5)

        detection_frame = ttk.Frame(notebook)
        notebook.add(detection_frame, text="Detection")
        self.create_detection_params(detection_frame)

        preprocessing_frame = ttk.Frame(notebook)
        notebook.add(preprocessing_frame, text="Preprocessing")
        self.create_preprocessing_params(preprocessing_frame)
        
        paths_frame = ttk.Frame(notebook)
        notebook.add(paths_frame, text="Data Paths")
        self.create_paths_params(paths_frame)
    
    def create_detection_params(self, parent):
        config = self.config_manager.config.detection
        
        params = [
            ('baseline_start_time', 'Baseline Start Time (s)', config.baseline_start_time),
            ('baseline_end_time', 'Baseline End Time (s)', config.baseline_end_time),
            ('fs', 'Sampling Frequency (Hz)', config.fs),
            ('tmul', 'Threshold Multiplier', config.tmul),
            ('absthresh', 'Absolute Threshold (μV)', config.absthresh),
            ('sur_time', 'Surround Time (ms)', config.sur_time),
            ('close_to_edge', 'Edge Exclusion (fraction)', config.close_to_edge),
            ('too_high_abs', 'Artifact Threshold (μV)', config.too_high_abs),
            ('spkdur_min', 'Min Spike Duration (ms)', config.spkdur_min),
            ('spkdur_max', 'Max Spike Duration (ms)', config.spkdur_max)
        ]
        
        row = 0
        for param_name, label, default_value in params:
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', padx=5, pady=2)
            
            var = tk.DoubleVar(value=default_value)
            entry = ttk.Entry(parent, textvariable=var, width=15)
            entry.grid(row=row, column=1, sticky='ew', padx=5, pady=2)
            
            self.param_vars[f'detection.{param_name}'] = var
            row += 1
        
        
        parent.columnconfigure(1, weight=1)
    
    def create_preprocessing_params(self, parent):
        eeg_config = self.config_manager.config.preprocessing
        
        eeg_frame = ttk.LabelFrame(parent, text="EEG Preprocessing")
        eeg_frame.pack(fill='x', padx=5, pady=5)
        
        eeg_params = [
            ('eeg_sampling_freq', 'EEG Sampling Freq (Hz)', eeg_config.eeg_sampling_freq),
            ('notch_frequency', 'Notch Frequency (Hz)', eeg_config.notch_frequency),
            ('notch_quality', 'Notch Quality Factor', eeg_config.notch_quality),
            ('bandpass_low', 'Bandpass Low (Hz)', eeg_config.bandpass_low),
            ('bandpass_high', 'Bandpass High (Hz)', eeg_config.bandpass_high),
            ('interpolation_threshold', 'Artifact Threshold (σ)', eeg_config.interpolation_threshold),
        ]
        
        for row, (param_name, label, default_value) in enumerate(eeg_params):
            ttk.Label(eeg_frame, text=label).grid(row=row, column=0, sticky='w', padx=5, pady=2)
            
            var = tk.DoubleVar(value=default_value)
            entry = ttk.Entry(eeg_frame, textvariable=var, width=15)
            entry.grid(row=row, column=1, sticky='ew', padx=5, pady=2)
            
            self.param_vars[f'preprocessing.{param_name}'] = var
        
        photo_chandni_frame = ttk.LabelFrame(parent, text="Photometry Preprocessing (Chandni)")
        photo_chandni_frame.pack(fill='x', padx=5, pady=5)
        
        photo_chandni_params = [
            ('gaussian_sigma', 'Gaussian Sigma', eeg_config.gaussian_sigma),
        ]
        
        for row, (param_name, label, default_value) in enumerate(photo_chandni_params):
            ttk.Label(photo_chandni_frame, text=label).grid(row=row, column=0, sticky='w', padx=5, pady=2)
            
            var = tk.DoubleVar(value=default_value)
            entry = ttk.Entry(photo_chandni_frame, textvariable=var, width=15)
            entry.grid(row=row, column=1, sticky='ew', padx=5, pady=2)
            
            self.param_vars[f'preprocessing.{param_name}'] = var
        
        eeg_frame.columnconfigure(1, weight=1)
        photo_chandni_frame.columnconfigure(1, weight=1)
        
        photo_meiling_frame = ttk.LabelFrame(parent, text="Photometry Preprocessing (Meiling)")
        photo_meiling_frame.pack(fill='x', padx=5, pady=5)
        
        photo_meiling_params = [
            ('lowpass_cutoff', 'Lowpass Cutoff (Hz)', eeg_config.lowpass_cutoff)
        ]
        
        for row, (param_name, label, default_value) in enumerate (photo_meiling_params):
            ttk.Label(photo_meiling_frame, text=label).grid(row=row, column=0, sticky='w', padx=5, pady=2)
            
            var = tk.DoubleVar(value=default_value)
            entry = ttk.Entry(photo_meiling_frame, textvariable=var, width=15)
            entry.grid(row=row, column=1, sticky='ew', padx=5, pady=2)
            
            self.param_vars[f'preprocessing.{param_name}'] = var
            
        eeg_frame.columnconfigure(1, weight=1)
        photo_meiling_frame.columnconfigure(1, weight=1)
    
    def create_paths_params(self, parent):
        paths_config = self.config_manager.config.data_paths
        
        paths = [
            ('eeg_data_dir', 'EEG Data Directory', paths_config.eeg_data_dir),
            ('photometry_data_dir', 'Photometry Data Directory', paths_config.photometry_data_dir),
            ('output_dir', 'Output Directory', paths_config.output_dir),
            ('seizure_onsets_file', 'Seizure Onsets File', paths_config.seizure_onsets_file),
        ]
        
        for row, (param_name, label, default_value) in enumerate(paths):
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', padx=5, pady=2)
            
            var = tk.StringVar(value=default_value)
            entry = ttk.Entry(parent, textvariable=var, width=40)
            entry.grid(row=row, column=1, sticky='ew', padx=5, pady=2)
            
        
            def make_browse_callback(var, is_dir=True):
                def browse():
                    if is_dir:
                        path = filedialog.askdirectory(initialdir=var.get())
                    else:
                        path = filedialog.askopenfilename(initialdir=Path(var.get()).parent)
                    if path:
                        var.set(path)
                return browse
            
            is_directory = param_name.endswith('_dir')
            browse_btn = ttk.Button(
                parent, 
                text="Browse...", 
                command=make_browse_callback(var, is_directory)
            )
            browse_btn.grid(row=row, column=2, padx=5, pady=2)
            
            self.param_vars[f'data_paths.{param_name}'] = var
        
        parent.columnconfigure(1, weight=1)
    
    def get_parameters(self) -> Dict[str, Any]:
        params = {}
        for key, var in self.param_vars.items():
            params[key] = var.get()
        return params
    
    def reset_to_defaults(self):

        self.config_manager.load_config()
        config = self.config_manager.config
        
        for param_name in ['baseline_end_time', 'baseline_start_time', 'fs', 'tmul', 'absthresh', 'sur_time', 'close_to_edge', 
                          'too_high_abs', 'spkdur_min', 'spkdur_max']:
            if f'detection.{param_name}' in self.param_vars:
                self.param_vars[f'detection.{param_name}'].set(
                    getattr(config.detection, param_name)
                )


class LoggingFrame(ttk.Frame):

    
    def __init__(self, parent):
        super().__init__(parent)
        self.setup_ui()
        self.setup_logging()
    
    def setup_ui(self):
        ttk.Label(self, text="Application Log:").pack(anchor='w')
        
        self.log_text = scrolledtext.ScrolledText(
            self, 
            height=10, 
            width=80,
            wrap=tk.WORD,
            state='disabled'
        )
        self.log_text.pack(fill='both', expand=True, pady=5)

        ttk.Button(self, text="Clear Log", command=self.clear_log).pack(anchor='e', pady=2)
    
    def setup_logging(self):
        self.log_handler = GUILogHandler(self.log_text)
        self.log_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.log_handler.setFormatter(formatter)
        

        logging.getLogger().addHandler(self.log_handler)
    
    def clear_log(self):
        self.log_text.config(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state='disabled')


class GUILogHandler(logging.Handler):

    
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
    
    def emit(self, record):
        try:
            msg = self.format(record)
            self.text_widget.config(state='normal')
            self.text_widget.insert(tk.END, msg + '\n')
            self.text_widget.see(tk.END)
            self.text_widget.config(state='disabled')
            self.text_widget.update()
        except Exception:
            pass  


class SpikeDetectionGUI:
    
    def __init__(self, 
                 config_manager: ConfigManager,
                 data_manager: DataManager,
                 preprocessing_pipeline: PreprocessingPipeline,
                 spike_detector: SpikeDetector,
                 plotter: InteractivePlotter,
                 alignment: SignalAlignment):
        
        self.config_manager = config_manager
        self.data_manager = data_manager
        self.preprocessing_pipeline = preprocessing_pipeline
        self.spike_detector = spike_detector
        self.plotter = plotter
        self.alignment = alignment

        self.current_eeg_file = None
        self.current_photometry_file = None
        self.analysis_results = None
        self.processing_thread = None

        self.root = tk.Tk()
        self.root.title("Spike Detection Toolkit")

        self.root.geometry("1400x1000")
        self.root.minsize(1200, 800)

        self.setup_ui()
        
        logger.info("GUI initialized successfully")
    
    def setup_ui(self):

        self.create_menu()

        main_paned = ttk.PanedWindow(self.root, orient='horizontal')
        main_paned.pack(fill='both', expand=True, padx=5, pady=5)

        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=2)

        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)

        self.setup_control_panel(left_frame)
        
        self.setup_results_panel(right_frame)

        self.setup_status_bar()
    
    def create_menu(self):

        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load EEG Data...", command=self.load_eeg_data)
        file_menu.add_command(label="Load Photometry Data...", command=self.load_photometry_data)
        file_menu.add_separator()
        file_menu.add_command(label="Load Configuration...", command=self.load_config)
        file_menu.add_command(label="Save Configuration...", command=self.save_config)
        file_menu.add_separator()
        file_menu.add_command(label="Export Results...", command=self.export_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Run Spike Detection", command=self.run_analysis)
        analysis_menu.add_command(label="Validate Data", command=self.validate_data)
        analysis_menu.add_separator()
        analysis_menu.add_command(label="Reset Parameters", command=self.reset_parameters)
        
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Show Results Plot", command=self.show_results_plot)
        view_menu.add_command(label="Show Data Validation", command=self.show_data_validation)
        

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        #help_menu.add_command(label="About", command=self.show_about)
        #help_menu.add_command(label="User Guide", command=self.show_user_guide)
    
    def setup_control_panel(self, parent):

        notebook = ttk.Notebook(parent)
        notebook.pack(fill='both', expand=True)
        

        data_frame = ttk.Frame(notebook)
        notebook.add(data_frame, text="Data Input")
        self.setup_data_input(data_frame)
        

        self.param_frame = ParameterFrame(notebook, self.config_manager)
        notebook.add(self.param_frame, text="Parameters")

        analysis_frame = ttk.Frame(notebook)
        notebook.add(analysis_frame, text="Analysis")
        self.setup_analysis_controls(analysis_frame)
    
    def setup_data_input(self, parent):

        eeg_frame = ttk.LabelFrame(parent, text="EEG Data")
        eeg_frame.pack(fill='x', padx=5, pady=5)
        
        self.eeg_file_var = tk.StringVar()
        ttk.Entry(eeg_frame, textvariable=self.eeg_file_var, width=30).pack(side='left', padx=5, pady=5)
        ttk.Button(eeg_frame, text="Browse...", command=self.load_eeg_data).pack(side='right', padx=5, pady=5)

        channel_frame = ttk.Frame(eeg_frame)
        channel_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(channel_frame, text="Channel:").pack(side='left')
        self.channel_var = tk.IntVar(value=self.config_manager.config.default_channel)
        ttk.Spinbox(channel_frame, from_=0, to=32, textvariable=self.channel_var, width=5).pack(side='left', padx=5)
        
        strategy_main_frame = ttk.LabelFrame(parent, text='Processing Strategy')
        strategy_main_frame.pack(fill='x', padx=5, pady=5)
        
        processing_options = [
            "None",
            "Meiling Denoise",
            "Meiling Detrend",
            "Chandni Gaussian Filter"
        ]
        
        step1_frame = ttk.Frame(strategy_main_frame)
        step1_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(step1_frame, text='Step 1:').pack(side='left')
        self.step1_var = tk.StringVar(value='Meiling Denoise')
        step1_combo = ttk.Combobox(step1_frame, textvariable=self.step1_var,
                                   values=processing_options, state='readonly', width=20)
        step1_combo.pack(side='left', padx=5)
        
        step2_frame = ttk.Frame(strategy_main_frame)
        step2_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(step2_frame, text='Step 2:').pack(side='left')
        self.step2_var = tk.StringVar(value="Meiling Detrend")
        step2_combo = ttk.Combobox(step2_frame, textvariable=self.step2_var,
                                   values=processing_options, state='readonly', width=20)
        step2_combo.pack(side='left', padx=5)
        
        step3_frame = ttk.Frame(strategy_main_frame)
        step3_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(step3_frame, text='Step 3:').pack(side='left')
        self.step3_var = tk.StringVar(value="None")
        step3_combo = ttk.Combobox(step3_frame, textvariable=self.step3_var,
                                   values=processing_options, state='readonly', width=20)
        step3_combo.pack(side='left', padx=5)
        
        photo_frame = ttk.LabelFrame(parent, text="Photometry Data (Optional)")
        photo_frame.pack(fill='x', padx=5, pady=5)
        
        self.photo_file_var = tk.StringVar()
        ttk.Entry(photo_frame, textvariable=self.photo_file_var, width=50).pack(side='left', padx=5, pady=5)
        ttk.Button(photo_frame, text="Browse...", command=self.load_photometry_data).pack(side='right', padx=5, pady=5)
    
    def setup_analysis_controls(self, parent):
        ttk.Button(
            parent, 
            text="Run Spike Detection", 
            command=self.run_analysis,
            style='Accent.TButton'
        ).pack(pady=10)
        
        controls_frame = ttk.Frame(parent)
        controls_frame.pack(fill='x', pady=5)
        
        ttk.Button(controls_frame, text="Validate Data", command=self.validate_data).pack(fill='x', pady=2)
        ttk.Button(controls_frame, text="Preview Data", command=self.preview_data).pack(fill='x', pady=2)
        ttk.Button(controls_frame, text="Reset Parameters", command=self.reset_parameters).pack(fill='x', pady=2)
        

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            parent, 
            variable=self.progress_var, 
            mode='determinate'
        )
        self.progress_bar.pack(fill='x', pady=10)
        

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(parent, textvariable=self.status_var).pack(pady=5)
    
    def setup_results_panel(self, parent):

        notebook = ttk.Notebook(parent)
        notebook.pack(fill='both', expand=True)
        

        results_frame = ttk.Frame(notebook)
        notebook.add(results_frame, text="Results")
        self.setup_results_display(results_frame)
        

        self.log_frame = LoggingFrame(notebook)
        notebook.add(self.log_frame, text="Log")
    
    def setup_results_display(self, parent):

        self.results_text = scrolledtext.ScrolledText(
            parent, 
            height=20, 
            width=60,
            wrap=tk.WORD,
            state='disabled'
        )
        self.results_text.pack(fill='both', expand=True, padx=5, pady=5)
        

        button_frame = ttk.Frame(parent)
        button_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(button_frame, text="Show Plot", command=self.show_results_plot).pack(side='left', padx=2)
        ttk.Button(button_frame, text="Export Results", command=self.export_results).pack(side='left', padx=2)
        ttk.Button(button_frame, text="Clear Results", command=self.clear_results).pack(side='left', padx=2)
    
    def setup_status_bar(self):
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.pack(fill='x', side='bottom')
        
        self.status_label = ttk.Label(self.status_bar, text="Ready")
        self.status_label.pack(side='left', padx=5, pady=2)
    
    # Event handlers
    def load_eeg_data(self):
        filename = filedialog.askdirectory(
            title="Select EEG Data Directory",
            initialdir=self.config_manager.config.data_paths.eeg_data_dir
        )
        if filename:
            self.eeg_file_var.set(filename)
            self.current_eeg_file = filename
            logger.info(f"Selected EEG file: {filename}")
    
    def load_photometry_data(self):

        filename = filedialog.askopenfilename(
            title="Select Photometry Data File",
            initialdir=self.config_manager.config.data_paths.photometry_data_dir,
            filetypes=[("PPD files", "*.ppd"), ("All files", "*.*")]
        )
        if filename:
            self.photo_file_var.set(filename)
            self.current_photometry_file = filename
            logger.info(f"Selected photometry file: {filename}")
    
    def load_config(self):

        filename = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")]
        )
        if filename:
            try:
                self.config_manager.load_config(filename)
                self.param_frame.reset_to_defaults()
                messagebox.showinfo("Success", "Configuration loaded successfully")
                logger.info(f"Loaded configuration from {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration: {e}")
    
    def save_config(self):
        filename = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
        )
        if filename:
            try:
                params = self.param_frame.get_parameters()
                # TODO: Apply parameters to config
                
                self.config_manager.save_config(filename)
                messagebox.showinfo("Success", "Configuration saved successfully")
                logger.info(f"Saved configuration to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save configuration: {e}")
    
    def run_analysis(self):
        if not self.current_eeg_file:
            messagebox.showerror("Error", "Please select an EEG data file first")
            return

        if self.processing_thread and self.processing_thread.is_alive():
            messagebox.showwarning("Warning", "Analysis is already running")
            return
        
        try:
            self._update_components_from_gui()
        except Exception as e:
            messagebox.showerror("Parameter Error", f"Error updating parameters: {e}")
            return
        
        
        self.processing_thread = threading.Thread(target=self._run_analysis_thread)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
    def _update_components_from_gui(self):
        
        try:
            gui_params = self.param_frame.get_parameters()
            logger.info("Updating components with current GUI parameters")
            
            
            self._apply_gui_params_to_config(gui_params)
            
            self._reinitialize_preprocessing_pipeline()
            self._reinitialize_spike_detector()
            self._update_plotter_config()
            
            logger.info("All components updated using user's parameters")
            
        except Exception as e:
            logger.errro(f"Error updating components from GUI: {e}")
            raise
    
    def _apply_gui_params_to_config(self, gui_params: Dict[str, Any]):
        config = self.config_manager.config
        
        for key, value in gui_params.items():
            if key.startswith('detection.'):
                param_name = key.replace('detection.', '')
                if hasattr(config.detection, param_name):
                    setattr(config.detection, param_name, value)
                    logger.debug(f"Updated detection.{param_name} = {value}")
                    
        for key, value in gui_params.items():
            if key.startswith('preprocessing.'):
                param_name = key.replace('preprocessing.', '')
                if hasattr(config.preprocessing, param_name):
                    setattr(config.preprocessing, param_name, value)
                    logger.debug(f"Updated preprocessing.{param_name} = {value}")
        
        for key, value in gui_params.items():
            if key.startswith('data_paths.'):
                param_name = key.replace('data_paths.', '')
                if hasattr(config.data_paths, param_name):
                    setattr(config.data_paths, param_name, value)
                    logger.debug(f"Updated data_paths.{param_name} = {value}")
                    
    def _reinitialize_preprocessing_pipeline(self):
        config = self.config_manager.config
        
        preprocessing_config = {
            'eeg': {
                'sample_rate': config.preprocessing.eeg_sampling_freq,
                'notch_frequency': config.preprocessing.notch_frequency,
                'notch_quality': config.preprocessing.notch_quality,
                'bandpass_low': config.preprocessing.bandpass_low,
                'bandpass_high': config.preprocessing.bandpass_high,
                'artifact_threshold': config.preprocessing.interpolation_threshold
            },
            'photometry': {
                'sample_rate': config.detection.fs,
                'lowpass_cutoff': config.preprocessing.lowpass_cutoff,
                'gaussian_sigma': config.preprocessing.gaussian_sigma,
                'savgol_window': config.preprocessing.savgol_window,
                'savgol_polyorder': config.preprocessing.savgol_polyorder
            },
            'baseline_start_time': config.detection.baseline_start_time,
            'baseline_end_time': config.detection.baseline_end_time
        }
        
        self.preprocessing_pipeline = PreprocessingPipeline(preprocessing_config)
        logger.info("Reinitialized Preprocessing Pipeline with user parameters")
        
    def _reinitialize_spike_detector(self):
        config = self.config_manager.config
        
        detection_params = SpikeDetectionParams(
            fs=config.detection.fs,
            tmul=config.detection.tmul,
            absthresh=config.detection.absthresh,
            sur_time=config.detection.sur_time,
            close_to_edge=config.detection.close_to_edge,
            too_high_abs=config.detection.too_high_abs,
            spkdur_min=config.detection.spkdur_min,
            spkdur_max=config.detection.spkdur_max,
            channel=self.channel_var.get(),  # Get current channel from GUI
            baseline_end_time=config.detection.baseline_end_time
        )
        
        self.spike_detector = SpikeDetector(detection_params)
        logger.info("Reinitialized SpikeDetector with user parametes")
        
    def _update_plotter_config(self):
        logger.info("Reinitialized plotter config with user parameters")
        
        
    def _run_analysis_thread(self):
        try:
            self.root.after(0, lambda: self.status_var.set("Loading data..."))
            self.root.after(0, lambda: self.progress_var.set(10))
            
            eeg_data, eeg_record = self.data_manager.load_eeg_data(
                self.current_eeg_file,
                channel=self.channel_var.get()
            )
            
            self.root.after(0, lambda: self.progress_var.set(30))
            
            photometry_record = None
            if self.current_photometry_file:
                photometry_record = self.data_manager.load_photometry_data(
                    self.current_photometry_file
                )
               
            processing_steps= [
                self.step1_var.get(),
                self.step2_var.get(),
                self.step3_var.get()
            ] 
            
            photometry_result = None
            if photometry_record:
               aligned_data = self.alignment.align_signals(eeg_record, photometry_record, self.channel_var.get())
               eeg_data = aligned_data['eeg']
              
               photometry_result = self.preprocessing_pipeline.process_photometry_modular(
                   aligned_data['gcamp'],
                   aligned_data['isos'],
                   processing_steps=processing_steps,
                   time_vector=aligned_data['time']
               )
            
            self.root.after(0, lambda: self.status_var.set("Preprocessing..."))
            self.root.after(0, lambda: self.progress_var.set(50))
            
            
            eeg_result = self.preprocessing_pipeline.process_eeg(eeg_data)
            

            
            self.root.after(0, lambda: self.status_var.set("Detecting spikes..."))
            self.root.after(0, lambda: self.progress_var.set(70))
            
            spikes = self.spike_detector.detect_spikes(eeg_result.data)
            summary = self.spike_detector.get_detection_summary(spikes)
            
            self.root.after(0, lambda: self.progress_var.set(100))
            
            self.analysis_results = {
                'spikes': spikes,
                'summary': summary,
                'eeg_result': eeg_result,
                'photometry_result': photometry_result,
                'eeg_raw': aligned_data['eeg'],
                'photo_raw': aligned_data['gcamp']
            }
            
            self.root.after(0, self._display_results)
            self.root.after(0, lambda: self.status_var.set("Analysis complete"))
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            self.root.after(0, lambda: messagebox.showerror("Analysis Error", str(e)))
            self.root.after(0, lambda: self.status_var.set("Analysis failed"))
        finally:
            self.root.after(0, lambda: self.progress_var.set(0))
    
    def _display_results(self):
        if not self.analysis_results:
            return
        
        summary = self.analysis_results['summary']
        spikes = self.analysis_results['spikes']
        
        # Format results text
        results_text = f"""
SPIKE DETECTION RESULTS
{'='*50}

Total spikes detected: {summary['n_spikes']}
Spike rate: {summary['spike_rate']:.2f} spikes/second
Mean amplitude: {summary['mean_amplitude']:.3f} ± {summary['std_amplitude']:.3f}
Mean width: {summary['mean_width']:.1f} ± {summary['std_width']:.1f} ms
Amplitude range: {summary['amplitude_range'][0]:.3f} to {summary['amplitude_range'][1]:.3f}
Width range: {summary['width_range'][0]:.1f} to {summary['width_range'][1]:.1f} ms

INDIVIDUAL SPIKES:
"""
        
        for i, spike in enumerate(spikes[:20]):  # Show first 20 spikes
            results_text += f"{i+1:2d}. {spike}\n"
        
        if len(spikes) > 20:
            results_text += f"... and {len(spikes) - 20} more spikes\n"
        
        # Update results display
        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, results_text)
        self.results_text.config(state='disabled')
    
    def validate_data(self):
        if not self.current_eeg_file:
            messagebox.showerror("Error", "Please select an EEG data file first")
            return
        
        try:
            eeg_data, eeg_record = self.data_manager.load_eeg_data(
                self.current_eeg_file,
                channel=self.channel_var.get()
            )
            
            eeg_validation = self.data_validator.validate_eeg_data(eeg_record)
            
            validation_text = f"EEG Data Validation:\n"
            validation_text += f"Valid: {'Yes' if eeg_validation['is_valid'] else 'No'}\n"
            
            if eeg_validation['issues']:
                validation_text += f"Issues: {', '.join(eeg_validation['issues'])}\n"
            
            if eeg_validation['warnings']:
                validation_text += f"Warnings: {', '.join(eeg_validation['warnings'])}\n"
            
            validation_text += f"\nStatistics:\n"
            for key, value in eeg_validation['stats'].items():
                validation_text += f"  {key}: {value}\n"
            
            if self.current_photometry_file:
                photometry_record = self.data_manager.load_photometry_data(
                    self.current_photometry_file
                )
                photo_validation = self.data_validator.validate_photometry_data(photometry_record)
                
                validation_text += f"\nPhotometry Data Validation:\n"
                validation_text += f"Valid: {'Yes' if photo_validation['is_valid'] else 'No'}\n"
                
                if photo_validation['issues']:
                    validation_text += f"Issues: {', '.join(photo_validation['issues'])}\n"
                
                if photo_validation['warnings']:
                    validation_text += f"Warnings: {', '.join(photo_validation['warnings'])}\n"
            
            self._show_validation_window(validation_text)
            
        except Exception as e:
            messagebox.showerror("Validation Error", f"Error validating data: {e}")
    
    def _show_validation_window(self, validation_text):
        validation_window = tk.Toplevel(self.root)
        validation_window.title("Data Validation Results")
        validation_window.geometry("600x400")
        
        text_widget = scrolledtext.ScrolledText(validation_window, wrap=tk.WORD)
        text_widget.pack(fill='both', expand=True, padx=10, pady=10)
        text_widget.insert(1.0, validation_text)
        text_widget.config(state='disabled')
        
        ttk.Button(validation_window, text="Close", 
                  command=validation_window.destroy).pack(pady=5)
    
    def preview_data(self):
        if not self.current_eeg_file:
            messagebox.showerror("Error", "Please select an EEG data file first")
            return
        
        try:
            eeg_data, eeg_record = self.data_manager.load_eeg_data(
                self.current_eeg_file,
                channel=self.channel_var.get()
            )
            
            fs = eeg_record.sample_rate
            preview_samples = min(int(10 * fs), len(eeg_data))
            preview_data = eeg_data[:preview_samples]
            
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            
            preview_window = tk.Toplevel(self.root)
            preview_window.title("Data Preview")
            preview_window.geometry("800x600")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            time_axis = np.arange(len(preview_data)) / fs
            ax.plot(time_axis, preview_data)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.set_title(f'EEG Data Preview - Channel {self.channel_var.get()}')
            ax.grid(True, alpha=0.3)
            
            canvas = FigureCanvasTkAgg(fig, preview_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            
            ttk.Button(preview_window, text="Close", 
                      command=preview_window.destroy).pack(pady=5)
            
        except Exception as e:
            messagebox.showerror("Preview Error", f"Error previewing data: {e}")
    
    def reset_parameters(self):
        result = messagebox.askyesno(
            "Reset Parameters", 
            "Are you sure you want to reset all parameters to defaults?"
        )
        if result:
            self.param_frame.reset_to_defaults()
            logger.info("Parameters reset to defaults")
    
    def show_results_plot(self):
        if not self.analysis_results:
            messagebox.showwarning("No analysis results available. Run analysis first.")
            return
        
        try:

            seizure_onset = None
            if self.current_eeg_file:
                filename = Path(self.current_eeg_file).name
                seizure_onset = self.config_manager.get_seizure_onset(filename)
        
            eeg_result = self.analysis_results['eeg_result']
            photometry_result = self.analysis_results['photometry_result']
            eeg_raw = self.analysis_results['eeg_raw']
            photo_raw = self.analysis_results['photo_raw']
            spikes = self.analysis_results['spikes']
            
            fs = self.config_manager.config.detection.fs
            time_vector = np.arange(len(eeg_result.data)) / fs
            
            fig = self.plotter.create_comprehensive_plot(
                eeg_data=eeg_result.data,
                eeg_raw=eeg_raw,
                photometry_data=photometry_result.data,
                photo_raw=photo_raw,
                spikes=spikes,
                time_vector=time_vector,
                seizure_onset=seizure_onset/fs,
                title=f"Analysis: {Path(self.current_eeg_file).name if self.current_eeg_file else 'Unknown'}"
            )
            
            fig.show()
            
        except Exception as e:
            messagebox.showerror("Plot Error", f"Error creating plot: {e}")
    
    def show_data_validation(self):

        self.validate_data()
    
    def export_results(self):

        if not self.analysis_results:
            messagebox.showwarning("Warning", "No analysis results available. Run analysis first.")
            return
        
        output_dir = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir=self.config_manager.config.data_paths.output_dir
        )
        
        if output_dir:
            try:
                self._save_results_to_directory(output_dir)
                messagebox.showinfo("Success", f"Results exported to {output_dir}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Error exporting results: {e}")
    
    def _save_results_to_directory(self, output_dir):
        import json
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.current_eeg_file:
            base_name = Path(self.current_eeg_file).stem
        else:
            base_name = "spike_detection_results"
        
        spikes = self.analysis_results['spikes']
        summary = self.analysis_results['summary']
        
        spikes_data = []
        for spike in spikes:
            spikes_data.append({
                'time_samples': int(spike.time_samples),
                'time_seconds': float(spike.time_seconds),
                'amplitude': float(spike.amplitude),
                'width_ms': float(spike.width_ms),
                'prominence': float(spike.prominence),
                'channel': int(spike.channel)
            })
        
        results = {
            'summary': summary,
            'spikes': spikes_data,
            'configuration': {
                'detection_params': asdict(self.spike_detector.params),
                'processing_strategy': self.strategy_var.get()
            },
            'files': {
                'eeg_file': self.current_eeg_file,
                'photometry_file': self.current_photometry_file
            }
        }
        
        results_file = output_path / f"{base_name}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        eeg_result = self.analysis_results['eeg_result']
        photometry_result = self.analysis_results['photometry_result']
        
        save_data = {
            'eeg_processed': eeg_result.data,
            'eeg_metadata': eeg_result.metadata
        }
        
        if photometry_result:
            save_data.update({
                'photometry_processed': photometry_result.data,
                'photometry_metadata': photometry_result.metadata
            })
        
        signals_file = output_path / f"{base_name}_signals.npz"
        np.savez_compressed(signals_file, **save_data)
        
        logger.info(f"Results saved to {output_path}")
    
    def clear_results(self):
        
        self.analysis_results = None
        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state='disabled')
        self.status_var.set("Ready")
        logger.info("Results cleared")
    

    def run(self) -> int:
        
        try:
            
            self.root.mainloop()
            return 0
            
        except Exception as e:
            logger.error(f"GUI error: {e}")
            return 1


def create_and_run_gui(config_path: Optional[str] = None) -> int:

    try:
        
        from config import ConfigManager
        from dataLoad import DataManager
        from DataPreprocessing import PreprocessingPipeline
        from SpikeDetection  import SpikeDetector, SpikeDetectionParams
        from visualization import InteractivePlotter
        
       
        config_manager = ConfigManager(config_path)
        config = config_manager.config
        
        # Validate configuration
        if not config_manager.validate_config():
            messagebox.showerror("Configuration Error", "Invalid configuration. Please check your settings.")
            return 1
        
        # Initialize components
        data_manager = DataManager()
        
        
        # Initialize preprocessing pipeline
        preprocessing_config = {
            'eeg': {
                'sample_rate': config.preprocessing.eeg_sampling_freq,
                'notch_frequency': config.preprocessing.notch_frequency,
                'notch_quality': config.preprocessing.notch_quality,
                'bandpass_low': config.preprocessing.bandpass_low,
                'bandpass_high': config.preprocessing.bandpass_high,
                'artifact_threshold': config.preprocessing.interpolation_threshold
            },
            'photometry': {
                'sample_rate': config.detection.fs,
                'lowpass_cutoff': config.preprocessing.lowpass_cutoff,
                'gaussian_sigma': config.preprocessing.gaussian_sigma,
                'savgol_window': config.preprocessing.savgol_window,
                'savgol_polyorder': config.preprocessing.savgol_polyorder
            },
            'baseline_start_time': config.detection.baseline_start_time,
            'baseline_end_time': config.detection.baseline_end_time
        }
        
        preprocessing_pipeline = PreprocessingPipeline(preprocessing_config)
        
        
        detection_params = SpikeDetectionParams(
            fs=config.detection.fs,
            tmul=config.detection.tmul,
            absthresh=config.detection.absthresh,
            sur_time=config.detection.sur_time,
            close_to_edge=config.detection.close_to_edge,
            too_high_abs=config.detection.too_high_abs,
            spkdur_min=config.detection.spkdur_min,
            spkdur_max=config.detection.spkdur_max,
            channel=config.default_channel,
            baseline_end_time=config.detection.baseline_end_time,
            baseline_start_time = config.detection.baseline_start_time
        )

        spike_detector = SpikeDetector(detection_params)
        
        plotter = InteractivePlotter(config.visualization)
        
        alignment = SignalAlignment(config.detection.fs)

        gui = SpikeDetectionGUI(
            config_manager=config_manager,
            data_manager=data_manager,
            preprocessing_pipeline=preprocessing_pipeline,
            spike_detector=spike_detector,
            plotter=plotter,
            alignment=alignment
        )
        
        return gui.run()
        
    except Exception as e:
        logger.error(f"Failed to initialize GUI: {e}")
        messagebox.showerror("Initialization Error", f"Failed to initialize application: {e}")
        return 1
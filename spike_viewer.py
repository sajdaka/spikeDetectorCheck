import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from pathlib import Path
import json
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class LoadedSpike:
    spike_number: int
    time_samples: int
    time_seconds: float
    amplitude: float
    width_ms: float
    prominence: float
    channel: str
    
class SpikeViewerGUI:
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Spike Viewer - Load and Review Detected Spikes")
        self.root.geometry("1200x800")
        
        self.spikes: List[LoadedSpike] = []
        self.eeg_data: np.ndarray = None
        self.eeg_zscored: np.ndarray = None
        self.fs: float = None
        self.channel: str = None
        self.current_index: int = 0
        
        self.signals_file: str = None
        self.results_file: str = None
        
        self.setup_ui()
        
    def setup_ui(self):
        
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side='top', fill='x', padx=10, pady=5)

        # Load files section
        load_frame = ttk.LabelFrame(control_frame, text="Load Spike Data")
        load_frame.pack(side='left', fill='x', expand=True, padx=5, pady=5)

        ttk.Button(
            load_frame,
            text="Load Signals NPZ",
            command=self.load_signals_file
        ).pack(side='left', padx=5, pady=5)

        self.signals_label = ttk.Label(load_frame, text="No signals loaded", foreground='gray')
        self.signals_label.pack(side='left', padx=5)

        ttk.Button(
            load_frame,
            text="Load Results JSON",
            command=self.load_results_file
        ).pack(side='left', padx=5, pady=5)

        self.results_label = ttk.Label(load_frame, text="No results loaded", foreground='gray')
        self.results_label.pack(side='left', padx=5)

        # Info panel
        info_frame = ttk.Frame(self.root)
        info_frame.pack(side='top', fill='x', padx=10, pady=5)

        self.info_label = ttk.Label(
            info_frame,
            text="Load spike data to begin viewing",
            font=('Arial', 12, 'bold')
        )
        self.info_label.pack(side='left')

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            info_frame,
            variable=self.progress_var,
            maximum=100,
            length=200
        )
        self.progress_bar.pack(side='right', padx=10)

        # Plot frame
        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side='top', fill='both', expand=True, padx=10, pady=5)

        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()

        # Navigation buttons
        nav_frame = ttk.Frame(self.root)
        nav_frame.pack(side='top', fill='x', padx=10, pady=5)

        ttk.Button(nav_frame, text="<< First", command=self.go_to_first).pack(side='left', padx=2)
        ttk.Button(nav_frame, text="< Previous", command=self.go_to_previous).pack(side='left', padx=2)

        self.jump_var = tk.IntVar(value=1)
        ttk.Label(nav_frame, text="Go to spike:").pack(side='left', padx=5)
        ttk.Entry(nav_frame, textvariable=self.jump_var, width=8).pack(side='left', padx=2)
        ttk.Button(nav_frame, text="Jump", command=self.jump_to_spike).pack(side='left', padx=2)

        ttk.Button(nav_frame, text="Next >", command=self.go_to_next).pack(side='left', padx=2)
        ttk.Button(nav_frame, text="Last >>", command=self.go_to_last).pack(side='left', padx=2)

        # Display mode toggle
        ttk.Label(nav_frame, text="Display:").pack(side='left', padx=(20, 5))
        self.display_mode = tk.StringVar(value='zscored')
        ttk.Radiobutton(
            nav_frame,
            text="Z-scored",
            variable=self.display_mode,
            value='zscored',
            command=self.refresh_display
        ).pack(side='left', padx=2)
        ttk.Radiobutton(
            nav_frame,
            text="Raw (µV)",
            variable=self.display_mode,
            value='raw',
            command=self.refresh_display
        ).pack(side='left', padx=2)

        # Keyboard shortcuts
        self.setup_keyboard_shortcuts()

        # Status bar
        self.status_var = tk.StringVar(value="Ready - Load spike data to begin")
        ttk.Label(self.root, textvariable=self.status_var, relief='sunken').pack(side='bottom', fill='x')
        
    def setup_keyboard_shortcuts(self):
        self.root.bind('<Left>', lambda e: self.go_to_previous())
        self.root.bind('<Right>', lambda e: self.go_to_next())
        self.root.bind('<Home>', lambda e: self.go_to_first())
        self.root.bind('<End>', lambda e: self.go_to_last())
        
    def load_signals_file(self):
        
        filename = filedialog.askopenfilename(
            title="Select Signals NPZ File",
            filetypes=[("NPZ file", "*.npz"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                data = np.load(filename, allow_pickle=True)
                
                self.eeg_data = data['eeg_processed']
                self.eeg_zscored = data['eeg_zscored']
                self.fs = float(data['sampling_rate'])
                self.channel = str(data['channel'])
                
                self.signals_file = filename
                self.signals_label.config(
                    text=f"{Path(filename).name}",
                    foreground='green'
                )
                
                messagebox.showinfo(
                    "Success",
                    f"Loaded EEG data:\n"
                    f"Sampling rate: {self.fs} Hz\n"
                    f"Channel: {self.channel}"
                )
                
                self.check_ready_to_display()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load signals file:\n{e}")
                
    def load_results_file(self):
        
        filename = filedialog.askopenfilename(
            title="Select Results JSON File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    results = json.load(f)
                    
                self.spikes = []
                for i, spike_data in enumerate(results['spikes'], 1):
                    spike = LoadedSpike(
                        spike_number = i,
                        time_samples=spike_data['time_samples'],
                        time_seconds=spike_data['time_seconds'],
                        amplitude=spike_data['amplitude'],
                        width_ms=spike_data['width_ms'],
                        prominence=spike_data['prominence'],
                        channel=spike_data.get('channel', 'Unknown')
                    )
                    self.spikes.append(spike)
                    
                self.results_file = filename
                self.results_label.config(
                    text=f"{Path(filename).name} ({len(self.spikes)} spikes)",
                    foreground='green'
                )
                
                self.progress_bar.config(maximum=len(self.spikes))
                
                messagebox.showinfo(
                    "Success",
                    f"Loaded {len(self.spikes)} spikes"
                )
                
                self.check_ready_to_display()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load results file:\n{e}")
                
    def check_ready_to_display(self):
        if self.eeg_data is not None and len(self.spikes) > 0:
            self.current_index = 0
            self.display_current_spike()
            self.status_var.set("Ready - Use arrow keys or buttons to navigate")
            
    def display_current_spike(self):
        
        if len(self.spikes) == 0 or self.eeg_data is None:
            return
        
        spike = self.spikes[self.current_index]
        
        if self.display_mode.get() == 'zscored':
            signal = self.eeg_zscored
            ylabel = 'EEG Signal (SD)'
            ylim = [-10, 10]
        else:
            signal = self.eeg_data * (10 ** 6)
            ylabel = 'EEG Signal (µV)'
            ylim = [-500, 500]
            
        window_samples = int(1.5 * self.fs)
        start_idx = max(0, spike.time_samples - window_samples)
        end_idx = min(len(signal), spike.time_samples + window_samples)
        
        segment = signal[start_idx:end_idx]
        
        time_axis = (np.arange(len(segment)) - (spike.time_samples - start_idx)) / self.fs * 1000
        
        self.ax.clear()
        self.ax.plot(time_axis, segment, 'b-', linewidth=1)
        
        spike_window_ms = 50
        self.ax.axvspan(-spike_window_ms, spike_window_ms, alpha=0.2, color='yellow')

        
        self.ax.set_xlabel('Time relative to spike (ms)', fontsize=12)
        self.ax.set_ylabel(ylabel, fontsize=12)
        self.ax.set_ylim(ylim)
        self.ax.set_title(
            f'Spike #{spike.spike_number} at {spike.time_seconds:.2f}s\n'
            f'Amplitude: {spike.amplitude:.2f} | Width: {spike.width_ms:.1f}ms | Prominence:{spike.prominence:.2f}',
            fontsize=12,
            fontweight='bold'
        )
        self.ax.grid(True, alpha=0.3)

        self.canvas.draw()

        
        self.info_label.config(
            text=f"Spike {self.current_index + 1}/{len(self.spikes)} | "
                f"Channel: {self.channel}"
        )
        self.progress_var.set(self.current_index + 1)
        self.status_var.set(f"Viewing spike at {spike.time_seconds:.2f}s")
        
    def refresh_display(self):
        if len(self.spikes) > 0 and self.eeg_data is not None:
            self.display_current_spike()
    
    def go_to_next(self):
        if self.current_index < len(self.spikes) - 1:
            self.current_index += 1
            self.display_current_spike()
            
    def go_to_previous(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.display_current_spike()
            
    def go_to_first(self):
        self.current_index = 0
        self.display_current_spike()
        
    def go_to_last(self):
        self.current_index = len(self.spikes) - 1
        self.display_current_spike()
        
    def jump_to_spike(self):
        try:
            target = self.jump_var.get() - 1
            if 0 <= target < len(self.spikes):
                self.current_index = target
                self.display_current_spike()
            else:
                messagebox.showerror("Error", f"Spike number must be between 1 and {len(self.spikes)}")
        except:
            messagebox.showerror("Error", "Invalid spike number")
            
def main():
    root = tk.Tk()
    app = SpikeViewerGUI(root)
    root.mainloop()
    
if __name__ == '__main__':
    main()

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import List, Optional, Dict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

from SpikeDetection import SpikeEvent

logger = logging.getLogger(__name__)


class SpikeLabelingGUI:


    def __init__(self,
                 parent: tk.Tk,
                 detected_spikes: List[SpikeEvent],
                 eeg_data: np.ndarray,
                 fs: float = 1000.0,
                 existing_labels_file: Optional[str] = None,
                 uncertainty_scores: Optional[np.ndarray] = None):
        """
        Initialize spike labeling GUI
        """
        self.parent = parent
        self.detected_spikes = detected_spikes
        self.eeg_data = eeg_data
        self.fs = fs

       
        self.labels = {} 
        self.current_index = 0
        self.history = []  

        
        if uncertainty_scores is not None:
            sorted_indices = np.argsort(uncertainty_scores)
            self.detected_spikes = [detected_spikes[i] for i in sorted_indices]
            logger.info(f"Sorted {len(detected_spikes)} spikes by uncertainty (most uncertain first)")

        
        if existing_labels_file:
            self.load_existing_labels(existing_labels_file)

        
        self.window = tk.Toplevel(parent)
        self.window.title("Spike Labeling Tool")
        self.window.geometry("1200x800")

        
        
        self.setup_ui()

        
        self.display_current_spike()

        
        self.setup_keyboard_shortcuts()

        logger.info(f"Spike labeling GUI initialized with {len(detected_spikes)} spikes")

    def setup_ui(self):
       

       
        info_frame = ttk.Frame(self.window)
        info_frame.pack(side='top', fill='x', padx=10, pady=5)

        self.info_label = ttk.Label(
            info_frame,
            text=f"Spike 1/{len(self.detected_spikes)} | Labeled: 0 | Remaining: {len(self.detected_spikes)}",
            font=('Arial', 12, 'bold')
        )
        self.info_label.pack(side='left')

        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            info_frame,
            variable=self.progress_var,
            maximum=len(self.detected_spikes),
            length=200
        )
        self.progress_bar.pack(side='right', padx=10)

        
        plot_frame = ttk.Frame(self.window)
        plot_frame.pack(side='top', fill='both', expand=True, padx=10, pady=5)

        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=True)

        
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()

        
        button_frame = ttk.Frame(self.window)
        button_frame.pack(side='top', fill='x', padx=10, pady=10)

       
        ttk.Button(
            button_frame,
            text="← ARTIFACT (0) [A or Left Arrow]",
            command=lambda: self.label_spike(0),
            width=30
        ).pack(side='left', padx=5)

        ttk.Button(
            button_frame,
            text="TRUE SPIKE (1) [S or Right Arrow] →",
            command=lambda: self.label_spike(1),
            width=30,
            style='Accent.TButton'
        ).pack(side='left', padx=5)

        ttk.Button(
            button_frame,
            text="Skip [Space]",
            command=self.skip_spike,
            width=15
        ).pack(side='left', padx=5)

        ttk.Button(
            button_frame,
            text="Undo [U]",
            command=self.undo_last,
            width=10
        ).pack(side='left', padx=5)

        
        nav_frame = ttk.Frame(self.window)
        nav_frame.pack(side='top', fill='x', padx=10, pady=5)

        ttk.Button(nav_frame, text="<< First", command=self.go_to_first).pack(side='left', padx=2)
        ttk.Button(nav_frame, text="< Previous", command=self.go_to_previous).pack(side='left', padx=2)

        self.jump_var = tk.IntVar(value=1)
        ttk.Label(nav_frame, text="Go to spike:").pack(side='left', padx=5)
        ttk.Entry(nav_frame, textvariable=self.jump_var, width=8).pack(side='left', padx=2)
        ttk.Button(nav_frame, text="Jump", command=self.jump_to_spike).pack(side='left', padx=2)

        ttk.Button(nav_frame, text="Next >", command=self.go_to_next).pack(side='left', padx=2)
        ttk.Button(nav_frame, text="Last >>", command=self.go_to_last).pack(side='left', padx=2)

        
        action_frame = ttk.Frame(self.window)
        action_frame.pack(side='top', fill='x', padx=10, pady=5)

        ttk.Button(action_frame, text="Save Progress", command=self.save_progress).pack(side='left', padx=5)
        ttk.Button(action_frame, text="Export Labels", command=self.export_labels).pack(side='left', padx=5)
        ttk.Button(action_frame, text="Statistics", command=self.show_statistics).pack(side='left', padx=5)
        ttk.Button(action_frame, text="Close", command=self.close_window).pack(side='right', padx=5)

       
        self.status_var = tk.StringVar(value="Ready to label spikes")
        ttk.Label(self.window, textvariable=self.status_var, relief='sunken').pack(side='bottom', fill='x')

    def setup_keyboard_shortcuts(self):
        
        self.window.bind('<Left>', lambda e: self.label_spike(0))
        self.window.bind('<Right>', lambda e: self.label_spike(1))
        self.window.bind('a', lambda e: self.label_spike(0))
        self.window.bind('A', lambda e: self.label_spike(0))
        self.window.bind('s', lambda e: self.label_spike(1))
        self.window.bind('S', lambda e: self.label_spike(1))
        self.window.bind('<space>', lambda e: self.skip_spike())
        self.window.bind('u', lambda e: self.undo_last())
        self.window.bind('U', lambda e: self.undo_last())
        self.window.bind('<Control-s>', lambda e: self.save_progress())

    def display_current_spike(self):
        
        if self.current_index >= len(self.detected_spikes):
            self.show_completion_message()
            return

        spike = self.detected_spikes[self.current_index]

        
        window_samples = int(1.5 * self.fs)  # 500ms
        start_idx = max(0, int(spike.time_samples - window_samples))
        end_idx = min(len(self.eeg_data), int(spike.time_samples + window_samples))

        segment = self.eeg_data[start_idx:end_idx]
        segment = segment * (10 **6)
        time_axis = (np.arange(len(segment)) - (spike.time_samples - start_idx)) / self.fs * 1000  # ms

        
        self.ax.clear()
        self.ax.plot(time_axis, segment, 'b-', linewidth=1)
        #self.ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Detected spike')

        
        spike_window_ms = 50
        self.ax.axvspan(-spike_window_ms, spike_window_ms, alpha=0.2, color='yellow')

        self.ax.set_xlabel('Time relative to spike (ms)', fontsize=12)
        self.ax.set_ylabel('EEG Signal (µV)', fontsize=12)
        self.ax.set_ylim([-500, 500])
        self.ax.set_title(
            f'Spike at {spike.time_seconds:.2f}s | Amplitude: {spike.amplitude:.2f} | Width: {spike.width_ms:.1f}ms',
            fontsize=14,
            fontweight='bold'
        )
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()

       
        if spike.time_seconds in self.labels:
            label = self.labels[spike.time_seconds]
            label_text = "TRUE SPIKE" if label == 1 else "ARTIFACT"
            color = 'green' if label == 1 else 'orange'
            self.ax.text(
                0.98, 0.98,
                f'Previously labeled: {label_text}',
                transform=self.ax.transAxes,
                fontsize=12,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.5)
            )

        self.canvas.draw()

       
        labeled_count = len(self.labels)
        remaining = len(self.detected_spikes) - labeled_count
        self.info_label.config(
            text=f"Spike {self.current_index + 1}/{len(self.detected_spikes)} | "
                 f"Labeled: {labeled_count} | Remaining: {remaining}"
        )
        self.progress_var.set(labeled_count)

       
        self.status_var.set(f"Viewing spike at {spike.time_seconds:.2f}s")

    def label_spike(self, label: int):
        """Label current spike"""
        if self.current_index >= len(self.detected_spikes):
            return

        spike = self.detected_spikes[self.current_index]

       
        self.history.append({
            'index': self.current_index,
            'spike_time': spike.time_seconds,
            'previous_label': self.labels.get(spike.time_seconds, None)
        })

        
        self.labels[spike.time_seconds] = label

        label_text = "TRUE SPIKE" if label == 1 else "ARTIFACT"
        logger.info(f"Labeled spike at {spike.time_seconds:.2f}s as {label_text}")

        # Auto-save every 10 labels
        if len(self.labels) % 10 == 0:
            self.auto_save()

        
        self.go_to_next()

    def skip_spike(self):
        
        logger.info(f"Skipped spike at {self.detected_spikes[self.current_index].time_seconds:.2f}s")
        self.go_to_next()

    def undo_last(self):
       
        if not self.history:
            messagebox.showinfo("Undo", "No actions to undo")
            return

        last_action = self.history.pop()

        
        if last_action['previous_label'] is None:
            del self.labels[last_action['spike_time']]
        else:
            self.labels[last_action['spike_time']] = last_action['previous_label']

        
        self.current_index = last_action['index']
        self.display_current_spike()

        logger.info(f"Undid last action")

    def go_to_next(self):
        """Navigate to next spike"""
        self.current_index += 1
        if self.current_index >= len(self.detected_spikes):
            self.show_completion_message()
        else:
            self.display_current_spike()

    def go_to_previous(self):
        """Navigate to previous spike"""
        if self.current_index > 0:
            self.current_index -= 1
            self.display_current_spike()

    def go_to_first(self):
        """Navigate to first spike"""
        self.current_index = 0
        self.display_current_spike()

    def go_to_last(self):
        """Navigate to last spike"""
        self.current_index = len(self.detected_spikes) - 1
        self.display_current_spike()

    def jump_to_spike(self):
        """Jump to specific spike number"""
        try:
            target = self.jump_var.get() - 1  
            if 0 <= target < len(self.detected_spikes):
                self.current_index = target
                self.display_current_spike()
            else:
                messagebox.showerror("Error", f"Spike number must be between 1 and {len(self.detected_spikes)}")
        except:
            messagebox.showerror("Error", "Invalid spike number")

    def auto_save(self):
        """Auto-save progress"""
        if len(self.labels) > 0:
            filename = f"./autosaves/spike_labels_autosave_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            self.export_labels_to_file(filename)
            logger.info(f"Auto-saved {len(self.labels)} labels to {filename}")

    def save_progress(self):
        """Manually save progress"""
        if len(self.labels) == 0:
            messagebox.showwarning("Warning", "No labels to save")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
            initialfile=f"spike_labels_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        )

        if filename:
            self.export_labels_to_file(filename)
            messagebox.showinfo("Success", f"Saved {len(self.labels)} labels to {filename}")

    def export_labels(self):
        """Export all labels to Excel file"""
        if len(self.labels) == 0:
            messagebox.showwarning("Warning", "No labels to export")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
            initialfile=f"spike_labels_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        )

        if filename:
            self.export_labels_to_file(filename)
            messagebox.showinfo("Success", f"Exported {len(self.labels)} labels to {filename}")

    def export_labels_to_file(self, filename: str):
        """Export labels to Excel file compatible with TrainingDataLoader"""
        
        times = sorted(self.labels.keys())
        labels = [self.labels[t] for t in times]

        df = pd.DataFrame({
            'Time': times, 
            'Class': labels 
        })

        
        df.to_excel(filename, index=False)
        logger.info(f"Exported {len(self.labels)} labels to {filename}")

    def load_existing_labels(self, filename: str):
        """Load existing labels from file"""
        try:
            df = pd.read_excel(filename)
            for _, row in df.iterrows():
                self.labels[row['Time']] = int(row['Class'])
            logger.info(f"Loaded {len(self.labels)} existing labels from {filename}")
        except Exception as e:
            logger.error(f"Failed to load existing labels: {e}")

    def show_statistics(self):
        """Show labeling statistics"""
        if len(self.labels) == 0:
            messagebox.showinfo("Statistics", "No labels yet")
            return

        artifact_count = sum(1 for label in self.labels.values() if label == 0)
        spike_count = sum(1 for label in self.labels.values() if label == 1)
        total = len(self.labels)
        remaining = len(self.detected_spikes) - total

        stats_text = f"""Labeling Statistics:

Total labeled: {total} / {len(self.detected_spikes)}
Artifacts (0): {artifact_count} ({artifact_count/total*100:.1f}%)
True spikes (1): {spike_count} ({spike_count/total*100:.1f}%)
Remaining: {remaining}

Progress: {total/len(self.detected_spikes)*100:.1f}% complete"""

        messagebox.showinfo("Statistics", stats_text)

    def show_completion_message(self):
        """Show message when all spikes are labeled"""
        labeled_count = len(self.labels)
        total_count = len(self.detected_spikes)

        if labeled_count == total_count:
            message = f"✓ All {total_count} spikes have been labeled!\n\nDon't forget to export your labels."
        else:
            message = f"You've reached the end.\n\nLabeled: {labeled_count}/{total_count} spikes\nRemaining: {total_count - labeled_count}\n\nYou can navigate back to label remaining spikes or export your progress."

        messagebox.showinfo("Labeling Complete", message)

    def close_window(self):
        """Close the labeling window"""
        if len(self.labels) > 0:
            response = messagebox.askyesnocancel(
                "Save Before Closing?",
                f"You have {len(self.labels)} labeled spikes.\n\nSave before closing?"
            )

            if response is None:  # Cancel
                return
            elif response:  
                self.save_progress()

        self.window.destroy()
        logger.info("Spike labeling GUI closed")


if __name__ == "__main__":
    
    import logging
    logging.basicConfig(level=logging.INFO)

    
    class DummySpike:
        def __init__(self, time_s, amplitude, width_ms):
            self.time_samples = int(time_s * 1000)
            self.time_seconds = time_s
            self.amplitude = amplitude
            self.width_ms = width_ms

    dummy_spikes = [DummySpike(t, np.random.randn()*10, np.random.rand()*100 + 50)
                    for t in np.random.uniform(1, 100, 50)]
    dummy_eeg = np.random.randn(100000)

    root = tk.Tk()
    root.withdraw()

    labeler = SpikeLabelingGUI(root, dummy_spikes, dummy_eeg)
    root.mainloop()

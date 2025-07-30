from typing import List, Optional, Dict, Any
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class PlotConfig:
    max_plot_points: int = 10000
    downsample_factor: int = 10
    figure_width: int = 1200
    figure_height: int = 800
    line_width: float = 1.0
    spike_marker_size: int = 10
    spike_marker_color: str = 'red'
    default_renderer: str = 'browser'
    

class InteractivePlotter:
    
    def __init__(self, config: Any = None):
        if hasattr(config, 'max_plot_points'):
            self.config = PlotConfig(
                max_plot_points=config.max_plot_points,
                downsample_factor=config.downsample_factor,
                figure_width=config.figure_width,
                figure_height=config.figue_height,
                line_width=config.line_width,
                spike_marker_size=config.spike_marker_size,
                spike_marker_color=config.spike_marker_color,
                default_render=config.default_renderer
            )
        else:
            self.config = PlotConfig()
        
        pio.renderers.default = self.config.default_renderer
    
    def create_comprehensive_plot(self,
                                  eeg_data: np.ndarray,
                                  photometry_data: Optional[np.ndarray] = None,
                                  spikes: Optional[List] = None,
                                  time_vector: Optional[np.ndarray] = None,
                                  seizure_onset: Optional[int] = None,
                                  title: str = "Spike Detection Ananlysis") -> go.Figure:
        try:
            n_rows = 3 if photometry_data is not None else 2
            
            subplot_titles = ['EEG Signal', 'Processed EEG']
            if photometry_data is not None:
                subplot_titles.append('Photometry Signal')
                
            fig = make_subplots(
                rows=n_rows,
                cols=1,
                subplot_titles=subplot_titles,
                vertical_spacing=0.08,
                shared_xaxes=True
            )
            
            if time_vector is None:
                time_vector = np.arange(len(eeg_data))
                
            if len(time_vector) > self.config.max_plot_points:
                step = len(time_vector) // (self.config.max_plot_points // 2)
                time_vector = time_vector[::step]
                raw_data = raw_data[::step]
                processed_data = processed_data[::step]
                
            fig = make_subplots(
                rows=2,
                cols=1,
                subplot_titles=('Raw Signal', 'Preprocessed Signal'),
                shared_xaxes=True,
                vertical_spacing=0.1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=time_vector,
                    y=raw_data,
                    mode='lines',
                    name='Raw Signal',
                    line=dict(width=self.config.line_width, color='blue')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=time_vector,
                    y=processed_data,
                    mode='lines',
                    name='Processed Signal',
                    line=dict(width=self.config.line_width, color='orange')
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                title=title,
                width=self.config.figure_width,
                height=600,
                showlegend=True
            )
            
            fig.update_xaxes(title_text="Time (s.)", row=2, col=1)
            fig.update_yaxes(title_text="Amplitude", row=1, col=1)
            fig.update_yaxes(title_text="Amplitude", row=2, col=1)
        except Exception as e:
            logger.error(f"Error creating plots: {e}")
            raise
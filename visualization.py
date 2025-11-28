from typing import List, Optional, Dict, Any
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import matplotlib.pyplot as plt
from dataclasses import dataclass
import logging

import config

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
                figure_height=config.figure_height,
                line_width=config.line_width,
                spike_marker_size=config.spike_marker_size,
                spike_marker_color=config.spike_marker_color,
                default_renderer=config.default_renderer
            )
        else:
            self.config = PlotConfig()
        
        pio.renderers.default = self.config.default_renderer
    
    def _get_photometry_segments(self,
                            spikes: List,
                            photometry: np.ndarray,
                            seizure_onset: Optional[int]) -> List:
        
        photometry_segments = []
        for spike in spikes:
            if seizure_onset:
                if (spike.time_seconds <= seizure_onset):
                    photometry_segment = photometry[int(spike.time_seconds-1):int(spike.time_seconds+5)]
                    photometry_segments.append(photometry_segment)
            else:
                photometry_segment = photometry[spike.time_seconds-1:spike.time_seconds+5]
                photometry_segments.append(photometry_segment)
                
        return photometry_segments
    
    
    def create_comprehensive_plot(self,
                                  eeg_data: np.ndarray,
                                  eeg_data_z: np.ndarray,
                                  eeg_raw: np.ndarray,
                                  photometry_data: Optional[np.ndarray] = None,
                                  photometry_data_z: Optional[np.ndarray] = None,
                                  photo_raw: Optional[np.ndarray] = None,
                                  spikes: Optional[List] = None,
                                  time_vector: Optional[np.ndarray] = None,
                                  seizure_onset: Optional[int] = None,
                                  title: str = "Spike Detection Ananlysis") -> go.Figure:
        try:
            
            
            subplot_titles = ['Raw EEG Signal', 'Processed EEG', 'Raw Photometry Signal', 'Processed Photometry Signal',
                              'Processed EEG with Markers on Spikes', 'Spike Segments Overlaid', 'Spike Segments Averaged']

            if photometry_data:   
                fig = make_subplots(
                    rows=5,
                    cols=2,
                    subplot_titles=subplot_titles,
                    vertical_spacing=0.08,
                    specs=[
                        [{}, {}],
                        [{}, {}],
                        [{"colspan": 2}, None],
                        [{}, {}]
                    ]
                )
            else:
                fig = make_subplots(
                    rows=2,
                    cols=2,
                    subplot_titles=['Raw EEG Signal', 'Processed EEG', 'Processed EEG with Markers on Spikes'],
                    vertical_spacing=0.08,
                    specs=[
                        [{}, {}],
                        [{"colspan": 2}, None]
                    ],
                )
            
            if time_vector is None:
                time_vector = np.arange(len(eeg_data))
                
            # if len(time_vector) > self.config.max_plot_points:
            #     step = len(time_vector) // (self.config.max_plot_points // 2)
            #     time_vector_trimed = time_vector[::step]
            #     eeg_data_trimed = eeg_data[::step]
            #     eeg_raw_trimed = eeg_raw[::step]
            #     photometry_data_trimed = photometry_data[::step]
            #     photo_raw_trimed = photo_raw[::step]
                
                

            
            fig.add_trace(
                go.Scatter(
                    x=time_vector,
                    y=eeg_raw,
                    mode='lines',
                    name='Raw EEG Signal',
                    line=dict(width=self.config.line_width, color='blue')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=time_vector,
                    y=eeg_data,
                    mode='lines',
                    name='Processed Photometry Signal',
                    line=dict(width=self.config.line_width, color='orange')
                ),
                row=1, col=2
            )
            if photometry_data:
                fig.add_trace(
                    go.Scatter(
                        x=time_vector,
                        y=photo_raw,
                        mode='lines',
                        name='Raw Photometry',
                        line=dict(width=self.config.line_width, color='yellow')
                    ),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=time_vector,
                        y=photometry_data_z,
                        mode='lines',
                        name='Procesed Photometry',
                        line= dict(width=self.config.line_width, color='green')
                    ),
                    row=2, col=2
                )
                
            if photometry_data:
                eegRow = 3
            else:
                eegRow = 2
            
            fig.add_trace(
                go.Scatter(
                    x=time_vector,
                    y=eeg_data,
                    mode='lines',
                    name='Processed Photometry with Spike indicators',
                    line=dict(width=self.config.line_width, color='black')
                ),
                row=eegRow, col=1
            )
            for spike in spikes:
                fig.add_trace(
                    go.Scatter(
                        x=[spike.time_seconds],
                        y=[spike.amplitude],
                        mode='markers',
                        marker=dict(symbol='x',
                                    size=10,
                                    color='red'),
                        showlegend=False
                    ),
                    row=eegRow, col=1
                )
            if photometry_data:  
                segments = self._get_photometry_segments(spikes, photometry_data_z, seizure_onset)
                segment_x = np.arange(7)
                
                for segment in segments:
                    fig.add_trace(
                        go.Scatter(
                            x=segment_x,
                            y=segment,
                            mode='lines',
                            line=dict(width=self.config.line_width, color='grey'),
                            showlegend=False
                        ),
                        row=4, col=1
                    )
                
                max_segment = max(len(s) for s in segments)
                padded_segments = [np.pad(s, (0, max_segment - len(s)), 'constant') for s in segments]
                segments_mean = np.mean(padded_segments, axis=0)
                
                fig.add_trace(
                    go.Scatter(
                        x=segment_x,
                        y=segments_mean,
                        mode='lines',
                        line=dict(width=self.config.line_width, color='brown')
                    ),
                    row=4, col=2
                )    
            
        
                
            fig.update_layout(
                title=title,
                width=self.config.figure_width,
                height=600,
                showlegend=True
            )
            

            
            if seizure_onset is not None:
                fig.add_vline(x=seizure_onset, row=1, col=2, line_dash='dash', line_color='red')
                fig.add_vline(x=seizure_onset, row=2, col=2, line_dash='dash', line_color='red')
            
            fig.update_xaxes(title_text="Time (s.)", row=2, col=1)
            fig.update_yaxes(title_text="Amplitude", row=1, col=1)
            fig.update_yaxes(title_text="Amplitude", row=2, col=1)
            
            # plt.hist(time_vector, eeg_data_z)
            # plt.show()
            # plt.close()
            
            return fig
        
        except Exception as e:
            logger.error(f"Error creating plots: {e}")
            raise
        

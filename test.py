from data_import import import_ppd
import sys
import os
import numpy as np
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def main():
    directory = ""
    directory = os.path.join('./FiberPhotometryData', sys.argv[1])
    data_ppd = import_ppd(directory)
    time_vector = np.arange(len(data_ppd['analog_1'])) / 100
    
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(
        go.Scatter(
            x=time_vector,
            y=data_ppd['analog_1'],
            mode='lines',
            line_shape='spline',
            name='Example of baseline z-score'
        ),
        row=1, col=1
    )
    fig.add_vline(x=120, line_dash='dash', line_color='red', row=1, col=1)
    fig.add_vline(x=480, line_dash='dash', line_color='red', row=1, col=1)
    
    fig.show()

if __name__ == "__main__":
    main()
    
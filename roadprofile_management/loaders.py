import numpy as np
import os
import plotly.graph_objects as go
import scipy as sp
from plotly.subplots import make_subplots


class Road:
    def __init__(self, folder: str, file_qty: int, file_ini: int = 0, file_extension: str = 'txt', tol: float = 0.2,
                 plot: bool = False):
        self.file_names = [os.path.join(folder, f'{i}.{file_extension}') for i in range(file_ini, file_ini + file_qty)]
        self.data_list = []
        for file_name in self.file_names:
            arr = np.loadtxt(file_name)
            self.data_list.append((arr[arr[:, 2] != 0, :]).copy())
            if len(self.data_list) > 1:
                maxy2 = np.max(self.data_list[-2][:, 1])
                self.data_list[-1][:, 1] = self.data_list[-1][:, 1] + maxy2
                miny1 = np.min(self.data_list[-1][:, 1])
                meanz_maxy2 = np.median(self.data_list[-2][self.data_list[-2][:, 1] > maxy2 - tol, 2])
                meanz_miny1 = np.median(self.data_list[-1][self.data_list[-1][:, 1] < miny1 + tol, 2])
                self.data_list[-1][:, 2] = self.data_list[-1][:, 2] - (meanz_miny1 - meanz_maxy2)
        self.data = np.vstack(self.data_list)
        if plot:
            fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scene'} for _ in range(2)]])
            fig.add_trace(go.Scatter3d(x=self.data[:, 0], y=self.data[:, 1], z=self.data[:, 2],  # Add the first subplot
                                       mode='markers', marker=dict(size=5, color='blue')), row=1, col=1)
            fig.add_trace(go.Scatter3d(x=self.data[:, 0], y=self.data[:, 1], z=self.data[:, 2],
                                       mode='markers', marker=dict(size=5, color='blue')), row=1, col=2)
            fig.update_scenes(aspectmode='data', row=1, col=1)
            fig.update_scenes(aspectmode='cube', row=1, col=2)
            fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))  # Add labels
            fig.show()  # Show the plot

    def get_profile(self, lane: float = 2, tol: float = 0.1, plot: bool = False, resample: bool = True):
        data_ = self.data[(lane - tol < self.data[:, 0]) & (self.data[:, 0] < lane + tol), :]
        # Extract unique values of y from the data
        unique_y = np.unique(data_[:, 1])
        # Initialize an empty array to store the results
        position, height = np.zeros((len(unique_y),)), np.zeros((len(unique_y),))
        # Calculate the mean of z values for each unique y
        for i, y_value in enumerate(unique_y):
            z_values_for_y = data_[data_[:, 1] == y_value][:, 2]
            mean_z_for_y = np.mean(z_values_for_y)
            position[i] = y_value
            height[i] = mean_z_for_y
        if resample:
            interpolation_function = sp.interpolate.interp1d(position, height, kind='cubic',
                                                             bounds_error=False, fill_value="extrapolate")
            position = np.linspace(position[0], position[-1], len(position))
            height = interpolation_function(position)
        if plot:
            # Create a 2x1 subplot layout
            fig = make_subplots(rows=2, cols=1)
            # Add the first subplot
            fig.add_trace(go.Scatter(x=position, y=height, mode='lines+markers', marker=dict(size=5, color='blue')),
                          row=1, col=1)
            # Add the second subplot
            fig.add_trace(go.Scatter(x=position, y=height, mode='lines+markers', marker=dict(size=5, color='blue')),
                          row=2, col=1)
            fig.update_xaxes(scaleanchor="y", scaleratio=1, row=1, col=1)
            fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
            for i in [1, 2]:
                fig.update_xaxes(title_text='horizontal position (m)', row=i, col=1)
                fig.update_yaxes(title_text='road height (m)', row=i, col=1)
            fig.show()
        position_height = np.hstack((position.reshape((-1, 1)), height.reshape((-1, 1))))
        return position_height


class SpeedDescriber:
    def __init__(self, t_accel: float, t_const: float, t_decel: float, max_speed: float):
        self.t_accel = t_accel
        self.t_const = t_const
        self.t_decel = t_decel
        self.max_speed = max_speed
        self.t_start_decel = self.t_accel + self.t_const
        self.T = self.t_accel + self.t_const + self.t_decel

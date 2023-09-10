from typing import Union, Iterable
import numpy as np
import scipy as sp


class Converter:
    def __init__(self, position_height: np.array, new_t_vector: Union[np.ndarray, None] = None):
        assert position_height.shape[1] == 2, AssertionError('two columns are expected')
        if new_t_vector is not None:
            assert isinstance(new_t_vector, np.ndarray)
            assert len(new_t_vector.shape) == 1
        self.position_height = position_height
        self.new_t_vector = new_t_vector

    def position2height(self, position: float):
        return self.position_height[np.argmin(np.abs(self.position_height[:, 0] - position)), 1]

    def position_th2height_th(self, position_th: np.array, smooth_time_window: float = 0.01):
        assert position_th.shape[1] == 2, AssertionError('two columns are expected')
        height_th = np.zeros((position_th.shape[0], 2))
        height_th[:, 0] = position_th[:, 0]  # same time vector
        height_th[:, 1] = np.vectorize(self.position2height)(position_th[:, 1])
        if smooth_time_window != 0:
            assert smooth_time_window > 0
            # Calculate the corresponding number of data points for the window
            sampling_rate = len(position_th[:, 0]) / (position_th[:, 0][-1] - position_th[:, 0][0])
            window_length = int(smooth_time_window * sampling_rate)
            # Ensure that the window length is odd
            if window_length % 2 == 0:
                window_length += 1
            # Perform smoothing
            interpolation_function = sp.interpolate.interp1d(
                height_th[0:-1:window_length, 0], height_th[0:-1:window_length, 1], kind='cubic',
                bounds_error=False, fill_value="extrapolate")
            height_th[:, 1] = interpolation_function(height_th[:, 0])
        return height_th

    def speed_th2position_th(self, speed_th):
        position_th = np.zeros((speed_th.shape[0], 2))
        position_th[:, 0] = speed_th[:, 0]
        position_th[:, 1] = sp.integrate.cumulative_trapezoid(speed_th[:, 1], x=speed_th[:, 0], initial=0.0)
        if self.new_t_vector is not None:
            interp_position_th = np.zeros((self.new_t_vector.shape[0], 2))
            interp_position_th[:, 0] = self.new_t_vector
            interpolation_function = sp.interpolate.interp1d(position_th[:, 0], position_th[:, 1], kind='cubic',
                                                             bounds_error=False, fill_value="extrapolate")
            interp_position_th[:, 1] = interpolation_function(self.new_t_vector)
            return interp_position_th
        return position_th

    def speed_th2height_th(self, speed_th):
        return self.position_th2height_th(self.speed_th2position_th(speed_th))

    def speed_th2roadvertacc_th(self, speed_th, smooth_time_window: float = 0.01):
        assert self.new_t_vector is not None
        roadvertvel_th = np.zeros((self.new_t_vector.shape[0], 2))
        roadvertvel_th[:, 0] = self.new_t_vector
        roadvertvel_th[0:-1, 1] = np.diff(self.speed_th2height_th(speed_th)[:, 1]) / np.diff(self.new_t_vector)
        roadvertvel_th[-1, 1] = roadvertvel_th[-2, 1]
        roadvertacc_th = np.zeros((self.new_t_vector.shape[0], 2))
        roadvertacc_th[:, 0] = self.new_t_vector
        roadvertacc_th[0:-1, 1] = np.diff(roadvertvel_th[:, 1]) / np.diff(roadvertvel_th[:, 0])
        roadvertacc_th[-1, 1] = roadvertacc_th[-2, 1]

        if smooth_time_window != 0:
            assert smooth_time_window > 0
            # Calculate the corresponding number of data points for the window
            sampling_rate = len(roadvertacc_th[:, 0]) / (roadvertacc_th[:, 0][-1] - roadvertacc_th[:, 0][0])
            window_length = int(smooth_time_window * sampling_rate)
            # Ensure that the window length is odd
            if window_length % 2 == 0:
                window_length += 1
            # Perform smoothing
            interpolation_function = sp.interpolate.interp1d(
                roadvertacc_th[0:-1:window_length, 0], roadvertacc_th[0:-1:window_length, 1], kind='cubic',
                bounds_error=False, fill_value="extrapolate")
            roadvertacc_th[:, 1] = interpolation_function(roadvertacc_th[:, 0])

        return roadvertacc_th

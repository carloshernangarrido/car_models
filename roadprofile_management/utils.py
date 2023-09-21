import numpy as np
from matplotlib import pyplot as plt

from roadprofile_management import converters
from roadprofile_management.loaders import Road, SpeedDescriber


def get_roadvertaccheight(loader: Road, speed_descr: SpeedDescriber, t_vector: np.ndarray, plot: bool = False,
                          lane=1.0):
    position_height = loader.get_profile(tol=.2, lane=lane, plot=False)
    converter = converters.Converter(position_height, new_t_vector=t_vector)
    speed_th = np.array([[0, 0],
                         [speed_descr.t_accel, speed_descr.max_speed],
                         [speed_descr.t_start_decel, speed_descr.max_speed],
                         [speed_descr.T, 0]])
    position_th = converter.speed_th2position_th(speed_th)
    if plot:
        fig, ax = plt.subplots(2, 2)
        ax[0, 0].plot(speed_th[:, 0], speed_th[:, 1], marker='.')
        ax[0, 0].set_xlabel('time (s)')
        ax[0, 0].set_ylabel('speed (m/s)')
        ax[1, 0].plot(position_th[:, 0], position_th[:, 1], marker='.')
        ax[1, 0].set_xlabel('time (s)')
        ax[1, 0].set_ylabel('position (m)')
        ax[0, 1].plot(position_height[:, 0], position_height[:, 1], marker='o')
        ax[0, 1].set_xlabel('position (m)')
        ax[0, 1].set_ylabel('height (m)')
        ax[1, 1].plot(converter.speed_th2height_th(speed_th)[:, 0],
                      converter.speed_th2height_th(speed_th)[:, 1], marker='.')
        ax[1, 1].set_xlabel('time (s)')
        ax[1, 1].set_ylabel('height (m)')
        plt.show()
    roadvertacc_th = converter.speed_th2roadvertacc_th(speed_th)
    return roadvertacc_th[:, 1], converter.speed_th2height_th(speed_th)[:, 1]

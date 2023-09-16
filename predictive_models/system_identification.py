from keras.models import Sequential
from keras.layers import Normalization

import numpy as np
import matplotlib.pyplot as plt

from .custom_layers import SOSConv1D


def batch_extractor(time: np.ndarray, response: np.ndarray, batch_length_s: float, ret_time_batches: bool = False,
                    runing: bool = True):
    time_batches = []
    response_batches = []
    if runing:
        batch_length = int(batch_length_s // (time[1] - time[0]))
        n_batches = len(time) - batch_length
    else:
        n_batches = int(time[-1] // batch_length_s)
        batch_length = int(len(time) // n_batches)
    for i_batch in range(n_batches):
        if runing:
            i_ini, i_fin = i_batch, i_batch + batch_length
        else:
            i_ini, i_fin = i_batch * batch_length, (i_batch + 1) * batch_length
        time_batches.append(time[i_ini:i_fin])
        response_batches.append(response[i_ini:i_fin])
    if ret_time_batches:
        return response_batches, time_batches
    else:
        return response_batches


def cardynamics_identification(t_vector, roadvert, carbodyvert, plot: bool = False):
    batch_length_s = 10
    timesteps_skip = 10
    runing = True
    x_train, x_time_batches = batch_extractor(t_vector[0::timesteps_skip], roadvert[0::timesteps_skip],
                                              batch_length_s, runing=runing, ret_time_batches=True)
    y_full_train, y_full_time_batches = batch_extractor(t_vector[0::timesteps_skip], carbodyvert[0::timesteps_skip],
                                                        batch_length_s, runing=runing, ret_time_batches=True)
    x_train, x_time_batches = np.array(x_train), np.array(x_time_batches)
    y_full_train, y_full_time_batches = np.array(y_full_train), np.array(y_full_time_batches)

    kernel_size = 500
    y_train = y_full_train[:, kernel_size - 1:]
    y_time_batches = [_[kernel_size - 1:] for _ in y_full_time_batches]
    # Define a Sequential model
    model = Sequential()
    model.add(Normalization(input_shape=(x_train[0].shape[0], 1)))
    # model.add(Conv1D(filters=1, kernel_size=kernel_size, use_bias=False, activation='linear',
    #                  input_shape=(x_train[0].shape[0], 1)))
    model.add(SOSConv1D(kernel_size=kernel_size, use_bias=True, activation='linear',
                        input_shape=(x_train[0].shape[0], 1)))
    model.compile(optimizer='adam', loss='mean_squared_error')
    # # Print the model summary
    model.summary()
    model.fit(x_train, y_train, epochs=5, validation_split=0.1)
    if plot:
        plt.figure()
        i_kernel = 8
        # plt.plot(x_time_batches[0][0:model.weights[3].shape[0]], np.array(model.weights[3]).reshape((-1,)))
        plt.plot(x_time_batches[0][0:model.weights[i_kernel].shape[0]], np.array(model.weights[i_kernel]).reshape((-1,)))
        samples_to_plot = 3
        for i in np.linspace(0, y_train.shape[0] - 1, samples_to_plot, dtype=int):
            fig, ax = plt.subplots(2, 1, sharex='col')
            ax[0].plot(x_time_batches[i], x_train[i, :], label='input')
            ax[0].legend()
            ax[1].plot(y_time_batches[i], y_train[i, :], label='actual output')
            ax[1].plot(y_time_batches[i], model.predict(x_train[i:i + 1, :]).reshape((-1,)), label='pred. output')
            ax[1].legend()
    #
    # kernel_size = 500
    # y_train = y_full_train[:, kernel_size - 1:]
    # y_time_batches = [_[kernel_size - 1:] for _ in y_full_time_batches]
    # # Define a Sequential model
    # model = Sequential()
    # model.add(Normalization(input_shape=(x_train[0].shape[0], 1)))
    # model.add(Conv1D(filters=1, kernel_size=kernel_size, use_bias=False, activation='linear',
    #                  input_shape=(x_train[0].shape[0], 1)))
    # model.compile(optimizer='adam', loss='mean_squared_error')
    # # # Print the model summary
    # model.summary()
    # model.fit(x_train, y_train, epochs=5, validation_split=0.1)

    ...

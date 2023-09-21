from typing import Union

from keras.models import Sequential
from keras.layers import Normalization
from keras.optimizers import Adam
from keras import regularizers, Input

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.src.layers import Conv1D, Lambda, Layer


def batch_extractor(time: np.ndarray, response: np.ndarray, batch_length_s: float, ret_time_batches: bool = False,
                    runing: bool = True, as_nparray: bool = False):
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
        if as_nparray:
            return np.array(response_batches), np.array(time_batches)
        else:
            return response_batches, time_batches
    else:
        if as_nparray:
            return np.array(response_batches)
        else:
            return response_batches


class CarDynamicsIdentification:
    def __init__(self, t_vector, roadvert, carbodyvert, kernel_length_s: float = 5.0,
                 batch_length_s: int = 10, timesteps_skip: int = 1, plot: bool = False):
        self.plot = plot
        self.timesteps_skip = timesteps_skip
        self.batch_length_s = batch_length_s
        self.carbodyvert = carbodyvert
        self.roadvert = roadvert
        self.t_vector = t_vector
        self.delta_t = timesteps_skip * (t_vector[1] - t_vector[0])
        self.kernel_size = int(kernel_length_s / self.delta_t)
        self.direct_x_train, self.direct_y_train, self.direct_y_pred, self.direct_model = None, None, None, None
        self.inverse_x_train, self.inverse_y_train, self.inverse_y_pred, self.inverse_model = None, None, None, None
        self.inverse_x_scale, self.inverse_y_scale, self.direct_x_scale, self.direct_y_scale = None, None, None, None

    def direct_identification(self, kernel_regularizer=None, plot: Union[bool, None] = None, samples_to_plot: int = 5,
                              fit: bool = True):
        plot = self.plot if plot is None else plot

        x_train, x_time_batches = \
            batch_extractor(self.t_vector[0::self.timesteps_skip], self.roadvert[0::self.timesteps_skip],
                            self.batch_length_s, runing=True, ret_time_batches=True)
        y_full_train, y_full_time_batches = \
            batch_extractor(self.t_vector[0::self.timesteps_skip], self.carbodyvert[0::self.timesteps_skip],
                            self.batch_length_s, runing=True, ret_time_batches=True)

        x_train, x_time_batches = np.array(x_train), np.array(x_time_batches)
        y_full_train, y_full_time_batches = np.array(y_full_train), np.array(y_full_time_batches)

        self.direct_x_scale, self.direct_y_scale = np.max(x_train), np.max(y_full_train)
        x_train, y_full_train = x_train / self.direct_x_scale, y_full_train / self.direct_y_scale

        y_train = y_full_train[:, self.kernel_size - 1:]
        y_time_batches = [_[self.kernel_size - 1:] for _ in y_full_time_batches]

        if fit:
            model = Sequential()
            model.add(Normalization(input_shape=(x_train[0].shape[0], 1)))
            model.add(Conv1D(filters=1, kernel_size=self.kernel_size, use_bias=False, activation='linear',
                             input_shape=(x_train[0].shape[0], 1), kernel_regularizer=kernel_regularizer))
            model.compile(optimizer=Adam(learning_rate=0.1), loss='mean_squared_error')
            model.summary()
            model.fit(x_train, y_train, epochs=20, validation_split=0.1)
            self.direct_model = model
        self.direct_x_train = x_train * self.direct_x_scale
        self.direct_y_train = y_train * self.direct_y_scale
        self.direct_y_pred = self.direct_model.predict(x_train) * self.direct_y_scale
        if plot:
            plt.figure()
            plt.plot(x_time_batches[0][0:self.direct_model.weights[3].shape[0]],
                     np.array(self.direct_model.weights[3]).reshape((-1,)))
            plt.title('impulse response')
            plt.xlabel('time (s)')
            for i in np.linspace(0, y_train.shape[0] - 1, samples_to_plot, dtype=int):
                fig, ax = plt.subplots(2, 1, sharex='col')
                ax[0].plot(x_time_batches[i], x_train[i, :], label='road')
                ax[0].legend()
                ax[1].plot(y_time_batches[i], y_train[i, :], label='carbody (actual)')
                ax[1].plot(y_time_batches[i], self.direct_model.predict(x_train[i:i + 1, :]).reshape((-1,)),
                           label='carbody (predicted)')
                ax[1].legend()
                ax[1].set_xlabel('time (s)')
            plt.show()

    def inverse_identification(self, kernel_regularizer=None, plot: Union[bool, None] = None, samples_to_plot: int = 5,
                               fit: bool = True):
        plot = self.plot if plot is None else plot
        x_train, x_time_batches = \
            batch_extractor(self.t_vector[0::self.timesteps_skip], self.carbodyvert[0::self.timesteps_skip],
                            self.batch_length_s, runing=True, ret_time_batches=True)
        y_full_train, y_full_time_batches = \
            batch_extractor(self.t_vector[0::self.timesteps_skip], self.roadvert[0::self.timesteps_skip],
                            self.batch_length_s, runing=True, ret_time_batches=True)

        x_train, x_time_batches = np.array(x_train), np.array(x_time_batches)
        y_full_train, y_full_time_batches = np.array(y_full_train), np.array(y_full_time_batches)

        self.inverse_x_scale, self.inverse_y_scale = np.max(x_train), np.max(y_full_train)
        x_train, y_full_train = x_train / self.inverse_x_scale, y_full_train / self.inverse_y_scale
        y_train = y_full_train[:, 0:-self.kernel_size + 1]
        y_time_batches = [_[0:-self.kernel_size + 1] for _ in y_full_time_batches]

        if fit:
            model = Sequential()
            model.add(Normalization(input_shape=(x_train[0].shape[0], 1)))
            model.add(Conv1D(filters=1, kernel_size=self.kernel_size, use_bias=False, activation='linear',
                             input_shape=(x_train[0].shape[0], 1), kernel_regularizer=kernel_regularizer))
            model.compile(optimizer=Adam(learning_rate=0.1), loss='mean_squared_error')
            model.summary()
            model.fit(x_train, y_train, epochs=20, validation_split=0.1)
            self.inverse_model = model
        self.inverse_x_train = x_train * self.inverse_x_scale
        self.inverse_y_train = y_train * self.inverse_y_scale
        self.inverse_y_pred = self.inverse_model.predict(x_train) * self.inverse_y_scale
        if plot:
            plt.figure()
            plt.plot(x_time_batches[0][0:self.inverse_model.weights[3].shape[0]],
                     np.array(self.inverse_model.weights[3]).reshape((-1,)))
            plt.title('impulse response')
            plt.xlabel('time (s)')
            for i in np.linspace(0, y_train.shape[0] - 1, samples_to_plot, dtype=int):
                fig, ax = plt.subplots(2, 1, sharex='col')
                ax[0].plot(x_time_batches[i], x_train[i, :], label='carbody')
                ax[0].legend()
                ax[1].plot(y_time_batches[i], y_train[i, :], label='road (actual)')
                ax[1].plot(y_time_batches[i], self.inverse_model.predict(x_train[i:i + 1, :]).reshape((-1,)),
                           label='road (predicted)')
                ax[1].legend()
                ax[1].set_xlabel('time (s)')
            plt.show()


class JointIdentification:
    def __init__(self, x11, x12, x21, x22, time,
                 kernel_length_s: float = 5.0, batch_length_s: int = 10, learning_rate=0.001):
        """
        :param x11: measurement with car 1 on road 1
        :param x12: measurement with car 1 on road 2
        :param x21: measurement with car 2 on road 1
        :param x22: measurement with car 2 on road 2
        """
        self.learning_rate = learning_rate
        self.x11 = x11
        self.x12 = x12
        self.x21 = x21
        self.x22 = x22
        self.time = time
        self.kernel_length_s = kernel_length_s
        self.batch_length_s = batch_length_s
        self.delta_t = time[1] - time[0]
        self.kernel_size = int(kernel_length_s / self.delta_t)
        # To be created in other methods
        self.model = None

        self.x11_train, self.x11_time_batches = batch_extractor(self.time, self.x11, self.batch_length_s, runing=True,
                                                                ret_time_batches=True, as_nparray=True)
        self.x12_train, self.x12_time_batches = batch_extractor(self.time, self.x12, self.batch_length_s, runing=True,
                                                                ret_time_batches=True, as_nparray=True)
        self.x21_train, self.x21_time_batches = batch_extractor(self.time, self.x21, self.batch_length_s, runing=True,
                                                                ret_time_batches=True, as_nparray=True)
        self.x22_train, self.x22_time_batches = batch_extractor(self.time, self.x22, self.batch_length_s, runing=True,
                                                                ret_time_batches=True, as_nparray=True)
        self.y11_train = np.zeros(self.x11_train.shape)
        self.y12_train = np.zeros(self.x12_train.shape)
        self.y21_train = np.zeros(self.x21_train.shape)
        self.y22_train = np.zeros(self.x22_train.shape)

        self.batch_size = self.x11_train.shape[1]
        self.input_shape = (self.batch_size, 1)

    def build_model(self):
        # Create a shared Conv1D layer
        shared_conv_layer_car1 = Conv1D(filters=1, use_bias=False, kernel_size=self.kernel_size,
                                        activation='linear', padding='valid')
        shared_conv_layer_car2 = Conv1D(filters=1, use_bias=False, kernel_size=self.kernel_size,
                                        activation='linear', padding='valid')

        # Define the input tensors
        input11 = Input(shape=self.input_shape)
        input12 = Input(shape=self.input_shape)
        input21 = Input(shape=self.input_shape)
        input22 = Input(shape=self.input_shape)

        # Use Lambda to apply shared Conv1D layers to each element of the input tensor
        conv11 = shared_conv_layer_car1(input11)
        conv12 = shared_conv_layer_car1(input12)
        conv21 = shared_conv_layer_car2(input21)
        conv22 = shared_conv_layer_car2(input22)

        # Create the model
        model = tf.keras.models.Model(inputs=[input11, input12, input21, input22],
                                      outputs=[conv11, conv12, conv21, conv22])

        # Custom loss function for enforcing the "same road"
        def same_road_loss(y_true, y_pred):
            # Extract conv11 and conv21 from y_pred
            conv11 = y_pred[0]
            conv12 = y_pred[1]
            conv21 = y_pred[2]
            conv22 = y_pred[3]
            # Calculate the squared difference between estimations of the same road
            squared_diff_road1 = tf.square(conv11 - conv21)
            squared_diff_road2 = tf.square(conv12 - conv22)
            # Compute the mean squared difference
            return tf.reduce_mean(squared_diff_road1) + tf.reduce_mean(squared_diff_road2)

        # Compile the model and specify loss and optimizer as needed
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=same_road_loss)

        # Print model summary
        model.summary()
        self.model = model

    def fit_model(self):
        self.model.fit([self.x11_train, self.x12_train, self.x21_train, self.x22_train],
                       [self.y11_train, self.y12_train, self.y21_train, self.y22_train], epochs=20, validation_split=0.1)

    def predict(self, x11=None, x12=None, x21=None, x22=None):
        if x11 is not None:
            self.x11 = x11
            self.x11_train, self.x11_time_batches = batch_extractor(self.time, self.x11, self.batch_length_s,
                                                                    runing=True, ret_time_batches=True, as_nparray=True)
        if x12 is not None:
            self.x12 = x12
            self.x12_train, self.x12_time_batches = batch_extractor(self.time, self.x12, self.batch_length_s,
                                                                    runing=True, ret_time_batches=True, as_nparray=True)
        if x21 is not None:
            self.x21 = x21
            self.x21_train, self.x21_time_batches = batch_extractor(self.time, self.x21, self.batch_length_s,
                                                                    runing=True, ret_time_batches=True, as_nparray=True)
        if x22 is not None:
            self.x22 = x22
            self.x22_train, self.x22_time_batches = batch_extractor(self.time, self.x22, self.batch_length_s,
                                                                    runing=True, ret_time_batches=True, as_nparray=True)

        return self.model.predict([self.x11_train, self.x12_train, self.x21_train, self.x22_train])


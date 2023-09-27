from typing import Union
from keras.models import Sequential
from keras.layers import Normalization
from keras.optimizers import Adam
from keras import regularizers, Input
from keras.src.layers import Conv1D, Lambda, Layer, Conv1DTranspose, Dense, Flatten
import tensorflow as tf
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from predictive_models.loss_functions import same_road_loss, SameRoadLossSimple, SameRoadLossCallback, \
    InformedSpeedSameRoadLossSimple
from predictive_models.system_identification import batch_extractor
from predictive_models.custom_layers import DiagonalDense


class EqualizerLearner:
    def __init__(self, x1, x2, time: Union[None, np.ndarray], y,
                 kernel_length_s: float = 5.0, batch_length_s: float = 10,
                 polynomial_kernel_degree: Union[None, int] = None, learning_rate=0.001, epochs: int = 10,
                 amplitude_regularization: float = 1.0, delta_t=None):
        """
            Unsupervised learner of equalizers, assuming x1 and x2 are outputs of filters excited by a common source.
        :param x1:
        :param x2:
        :param time:
        :param y:
        :param kernel_length_s:
        :param batch_length_s:
        :param polynomial_kernel_degree:
        :param learning_rate:
        :param epochs:
        :param amplitude_regularization:
        """
        self.y = y
        self.amplitude_regularization = amplitude_regularization
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.x1 = x1
        self.x2 = x2
        self.time = time
        self.kernel_length_s = kernel_length_s
        self.polynomial_kernel_degree = polynomial_kernel_degree
        self.batch_length_s = batch_length_s
        self.delta_t = time[1] - time[0] if time is not None else delta_t
        self.kernel_size = int(kernel_length_s / self.delta_t)

        self.model = None  # To be created in other methods
        if self.time is not None:
            self.x1_train, self.x1_time_batches = batch_extractor(self.time, self.x1, self.batch_length_s, runing=True,
                                                                  ret_time_batches=True, as_nparray=True)
            self.x2_train, self.x2_time_batches = batch_extractor(self.time, self.x2, self.batch_length_s, runing=True,
                                                                  ret_time_batches=True, as_nparray=True)

            self.y_train, self.y_time_batches = batch_extractor(self.time, self.y, self.batch_length_s, runing=True,
                                                                ret_time_batches=True, as_nparray=True)
            self.batch_size = self.x1_train.shape[1]
            # Frequency domain
            self.freq = np.fft.fftfreq(len(self.x1_time_batches[0]), self.delta_t)
            self.x1_train_fd = np.abs(np.fft.fft(self.x1_train, axis=1)[:, :len(self.freq) // 2])
            self.x2_train_fd = np.abs(np.fft.fft(self.x2_train, axis=1)[:, :len(self.freq) // 2])
            self.y_train_fd = np.abs(np.fft.fft(self.y_train, axis=1)[:, :len(self.freq) // 2])
            self.freq = self.freq[:len(self.freq) // 2]
            self.input_shape = (self.x1_train_fd.shape[1],)
        else:
            self.x1_train, self.x1_time_batches = None, None
            self.x2_train, self.x2_time_batches = None, None
            self.y_train, self.y_time_batches = None, None
            self.batch_size = None
            # Frequency domain
            self.freq = None
            self.x1_train_fd = None
            self.x2_train_fd = None
            self.y_train_fd = None
            self.freq = None
            self.input_shape = None

    def build_model(self):
        # Define the input tensors
        input1 = Input(shape=self.input_shape)
        input2 = Input(shape=self.input_shape)

        # Apply shared Conv1D layers to each element of the input tensor
        deg = self.polynomial_kernel_degree
        diag1 = DiagonalDense(units=self.input_shape[0], kernel_initializer='ones', polynomial_kernel_degree=deg)(input1)
        diag2 = DiagonalDense(units=self.input_shape[0], kernel_initializer='ones', polynomial_kernel_degree=deg)(input2)

        # Create the model
        model = tf.keras.models.Model(inputs=[input1, input2],
                                      outputs=[diag1, diag2])

        # Compile the model and specify loss and optimizer as needed
        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                      loss=SameRoadLossSimple(amplitude_regularization=self.amplitude_regularization))

        # Print model summary
        model.summary()
        self.model = model

    def fit_model(self):
        self.model.fit([self.x1_train_fd, self.x2_train_fd], [0*self.x1_train_fd, 0*self.x2_train_fd],
                       epochs=self.epochs, validation_split=0.1,
                       callbacks=[SameRoadLossCallback(x_val=[self.x1_train_fd, self.x2_train_fd],
                                                       amplitude_regularization=self.amplitude_regularization)])

    def predict(self):
        return self.model.predict([self.x1_train_fd, self.x1_train_fd])


class SpeedInformedEqualizerLearner(EqualizerLearner):
    def __init__(self, x1, x2, s1, s2, p1, p2, t1, t2,
                 kernel_length_s: float = 5.0, batch_length_s: float = 10,
                 polynomial_kernel_degree: Union[None, int] = None, learning_rate=0.001, epochs: int = 10,
                 amplitude_regularization: float = 1.0):
        """
        It assumes that two cars go over the same road, at known speeds (s1 and s2) and known rectified positions
        (p1 and p2), generating responses x1 and x2.
        :param x1: response of car1
        :param x2: response of car2
        :param s1: speed of car 1
        :param s2: speed of car 2
        :param p1: position of car 1
        :param p2: position of car 2
        :param t1: time vector por measurements on car 1
        :param t2: time vector por measurements on car 2
        :param kernel_length_s:
        :param batch_length_s:
        :param polynomial_kernel_degree:
        :param learning_rate:
        :param epochs:
        :param amplitude_regularization:
        """
        assert (t1[1]-t1[0]) == (t2[1]-t2[0])
        super().__init__(x1=x1, x2=x2, time=None, y=None,
                         kernel_length_s=kernel_length_s, batch_length_s=batch_length_s,
                         polynomial_kernel_degree=polynomial_kernel_degree, learning_rate=learning_rate, epochs=epochs,
                         amplitude_regularization=amplitude_regularization, delta_t=t1[1] - t1[0])
        self.s1, self.s2 = s1, s2
        self.p1, self.p2 = p1, p2
        self.t1, self.t2 = t1, t2

        self.x1_train, self.x1_time_batches = batch_extractor(self.t1, self.x1, self.batch_length_s, runing=True,
                                                              ret_time_batches=True, as_nparray=True)
        self.x2_train, self.x2_time_batches = batch_extractor(self.t2, self.x2, self.batch_length_s, runing=True,
                                                              ret_time_batches=True, as_nparray=True)
        self.batch_size = self.x1_train.shape[1]
        # Frequency domain
        self.freq = np.fft.fftfreq(len(self.x1_time_batches[0]), self.delta_t)
        self.x1_train_fd = np.abs(np.fft.fft(self.x1_train, axis=1)[:, :len(self.freq) // 2])
        self.x2_train_fd = np.abs(np.fft.fft(self.x2_train, axis=1)[:, :len(self.freq) // 2])
        self.y_train_fd = np.abs(np.fft.fft(self.y_train, axis=1)[:, :len(self.freq) // 2])
        self.freq = self.freq[:len(self.freq) // 2]
        self.input_shape = (self.x1_train_fd.shape[1],)

    def build_model(self):
        # Define the input tensors
        input_x1 = Input(shape=self.input_shape)
        input_x2 = Input(shape=self.input_shape)
        input_s1 = Input(shape=(1,))
        input_s2 = Input(shape=(1,))

        # Apply shared Conv1D layers to each element of the input tensor
        deg = self.polynomial_kernel_degree
        diag1 = DiagonalDense(units=self.input_shape[0], kernel_initializer='ones',
                              polynomial_kernel_degree=deg)(input_s1)
        diag2 = DiagonalDense(units=self.input_shape[0], kernel_initializer='ones',
                              polynomial_kernel_degree=deg)(input_s2)

        # Create the model
        model = tf.keras.models.Model(inputs=[input_x1, input_x2, input_s1, input_s2],
                                      outputs=[diag1, diag2, input_s1, input_s2])

        # Compile the model and specify loss and optimizer as needed
        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                      loss=InformedSpeedSameRoadLossSimple(amplitude_regularization=self.amplitude_regularization))

        # Print model summary
        model.summary()
        self.model = model

    def fit_model(self):
        ...

    def predict(self):
        ...
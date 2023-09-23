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

from predictive_models.loss_functions import same_road_loss, SameRoadLossSimple, SameRoadLossCallback
from predictive_models.system_identification import batch_extractor
from predictive_models.custom_layers import DiagonalDense


class EqualizerLearner:
    def __init__(self, x1, x2, time, y,
                 kernel_length_s: float = 5.0, batch_length_s: float = 10, learning_rate=0.001, epochs: int = 10,
                 amplitude_regularization: float = 1.0):
        """
        Unsupervised learner of equalizers, assuming x1 and x2 are outputs of filters excited by a common source.
        :param x1:
        :param x2:
        :param time:
        :param kernel_length_s:
        :param batch_length_s:
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
        self.batch_length_s = batch_length_s
        self.delta_t = time[1] - time[0]
        self.kernel_size = int(kernel_length_s / self.delta_t)

        self.model = None  # To be created in other methods

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
        self.input_shape = (self.x1_train_fd.shape[1], )

    def build_model(self):
        # Define the input tensors
        input1 = Input(shape=self.input_shape)
        input2 = Input(shape=self.input_shape)

        # Apply shared Conv1D layers to each element of the input tensor
        diag1 = DiagonalDense(units=self.input_shape[0], kernel_initializer='ones')(input1)
        diag2 = DiagonalDense(units=self.input_shape[0], kernel_initializer='ones')(input2)

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

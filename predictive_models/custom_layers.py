from typing import Union, Callable
import numpy as np
import tensorflow as tf


class SOSConv1D(tf.keras.layers.Layer):
    """
    Convolutional 1D layer enforced to behave as a Second Order Section.
    """
    def __init__(self, kernel_size: int, activation: Union[None, str, Callable], use_bias: bool = False,
                 padding: str = "VALID", stride: int = 1, **kwargs):
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.activation = 'linear' if activation is None else activation
        self.padding = padding
        self.stride = stride
        # trainable parameters
        self.omega, self.phi, self.zeta, self.amplitude, self.bias = None, None, None, None, None
        # generated kernel
        self.kernel = None
        # misc
        self.t = tf.cast(tf.linspace(0, self.kernel_size, self.kernel_size), tf.float32)
        super(SOSConv1D, self).__init__(**kwargs)

    def build(self, input_shape):
        # trainable parameters
        # Define constraints for amplitude, omega, phi, and zeta
        amplitude_constraint = tf.keras.constraints.NonNeg()
        omega_constraint = tf.keras.constraints.MinMaxNorm(min_value=0.0, max_value=np.pi)
        phi_constraint = tf.keras.constraints.MinMaxNorm(min_value=-np.pi/2, max_value=np.pi/2)
        zeta_constraint = tf.keras.constraints.MinMaxNorm(min_value=0.0, max_value=1.0)
        self.omega = self.add_weight(name='omega', shape=(1,), initializer='random_normal', dtype='float32',
                                     trainable=True, constraint=omega_constraint)
        self.phi = self.add_weight(name='phi', shape=(1,), initializer='random_normal', dtype='float32', trainable=True,
                                   constraint=phi_constraint)
        self.zeta = self.add_weight(name='zeta', shape=(1,), initializer='zeros', dtype='float32',
                                    trainable=True, constraint=zeta_constraint)
        self.amplitude = self.add_weight(name='amplitude', shape=(1,), initializer='random_normal', dtype='float32',
                                         trainable=True, constraint=amplitude_constraint)

        # generated kernel
        self.kernel = self.add_weight(name='kernel', shape=[self.kernel_size, 1, 1], initializer='ones', dtype='float32',
                                      trainable=False)
        if self.use_bias:
            self.bias = self.add_weight(name='bias', shape=(1,), initializer='zeros', dtype='float32', trainable=True)

    def call(self, inputs, **kwargs):
        if kwargs['training'] is True:  # Training mode
            ...
        kernel = tf.reshape(self.amplitude * tf.math.cos(self.omega * self.t + self.phi) *
                            tf.math.exp(- self.zeta * self.omega * self.t), (self.kernel_size, 1, 1))
        self.kernel.assign(kernel)
        z = tf.nn.conv1d(input=inputs, filters=tf.reverse(kernel, axis=[0]), stride=self.stride, padding=self.padding)
        if self.use_bias:
            z += self.bias
        if self.activation == 'linear':
            return tf.keras.activations.linear(z)
        else:
            return self.activation(z)

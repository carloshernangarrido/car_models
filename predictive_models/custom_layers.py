from typing import Union, Callable
import numpy as np
import tensorflow as tf


class SOSConv1D(tf.keras.layers.Layer):
    """
    Convolutional 1D layer enforced to behave as a Second Order Section.
    """

    def __init__(self, kernel_size: int, sos_num: int = 1, activation: Union[None, str, Callable] = None,
                 use_bias: bool = False, padding: str = "VALID", stride: int = 1, **kwargs):
        self.kernel_size = kernel_size
        self.sos_num = sos_num
        self.use_bias = use_bias
        self.activation = 'linear' if activation is None else activation
        self.padding = padding
        self.stride = stride
        # trainable parameters
        self.omega = tf.zeros((self.sos_num,))
        self.phi = tf.zeros((self.sos_num,))
        self.zeta = tf.zeros((self.sos_num,))
        self.amplitude = tf.zeros((self.sos_num,))
        self.bias = 0.0
        # generated kernels
        self.kernels = None
        # misc
        self.t = tf.cast(tf.linspace(0, self.kernel_size, self.kernel_size), tf.float32)
        super(SOSConv1D, self).__init__(**kwargs)

    def build(self, input_shape):
        # trainable parameters
        # Define constraints for amplitude, omega, phi, and zeta
        amplitude_constraint = tf.keras.constraints.NonNeg()
        omega_constraint = tf.keras.constraints.MinMaxNorm(min_value=0.0, max_value=np.pi)
        phi_constraint = tf.keras.constraints.MinMaxNorm(min_value=-np.pi / 2, max_value=np.pi / 2)
        zeta_constraint = tf.keras.constraints.MinMaxNorm(min_value=0.0, max_value=1.0)
        self.omega = self.add_weight(name='omega', shape=(self.sos_num,), initializer='random_normal',
                                     dtype='float32',
                                     trainable=True, constraint=omega_constraint)
        self.phi = self.add_weight(name='phi', shape=(self.sos_num,), initializer='random_normal', dtype='float32',
                                   trainable=True,
                                   constraint=phi_constraint)
        self.zeta = self.add_weight(name='zeta', shape=(self.sos_num,), initializer='zeros', dtype='float32',
                                    trainable=True, constraint=zeta_constraint)
        self.amplitude = self.add_weight(name='amplitude', shape=(self.sos_num,), initializer='random_normal',
                                         dtype='float32',
                                         trainable=True, constraint=amplitude_constraint)

        # generated kernel
        self.kernel = self.add_weight(name='kernel', shape=(self.kernel_size,), initializer='ones',
                                      dtype='float32', trainable=False)
        if self.use_bias:
            self.bias = self.add_weight(name='bias', shape=(1,), initializer='zeros', dtype='float32', trainable=True)

    def call(self, inputs, **kwargs):
        if kwargs['training'] is True:  # Training mode
            ...
        kernel = tf.reshape(self.amplitude, (-1, 1)) \
                 * tf.math.cos(
            tf.reshape(self.omega, (-1, 1)) * tf.reshape(self.t, (1, -1)) + tf.reshape(self.phi, (-1, 1))) \
                 * tf.math.exp(
            tf.reshape(-self.zeta, (-1, 1)) * tf.reshape(self.omega, (-1, 1)) * tf.reshape(self.t, (1, -1)))
        kernel = tf.reduce_sum(kernel, axis=0)

        self.kernel.assign(kernel)

        z = tf.nn.conv1d(input=inputs, filters=tf.reshape(tf.reverse(kernel, axis=[0]), (-1, 1, 1)),
                         stride=self.stride, padding=self.padding)
        if self.use_bias:
            z += self.bias
        if self.activation == 'linear':
            return tf.keras.activations.linear(z)
        else:
            return self.activation(z)


class DiagonalDense(tf.keras.layers.Layer):
    def __init__(self, units, kernel_initializer, polynomial_kernel_degree: Union[None, int] = None, **kwargs):
        if polynomial_kernel_degree is not None:
            assert isinstance(polynomial_kernel_degree, int)
            assert polynomial_kernel_degree < units
        self.polynomial_kernel_degree = polynomial_kernel_degree
        self.kernel_initializer = kernel_initializer
        self.kernel = None
        self.kernel_coef = None
        self.kernel_basis = None
        self.units = units
        super(DiagonalDense, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.polynomial_kernel_degree is None:
            self.kernel = self.add_weight("kernel", shape=(self.units,), initializer=self.kernel_initializer,
                                          trainable=True, constraint=tf.keras.constraints.NonNeg())
        else:
            self.kernel_coef = self.add_weight("kernel_coef", shape=(self.polynomial_kernel_degree, 1),
                                               initializer=self.kernel_initializer, trainable=True)
            self.kernel = self.add_weight("kernel", shape=(self.units,), initializer=self.kernel_initializer,
                                          trainable=False, constraint=tf.keras.constraints.NonNeg())
            # Define the degree of the Legendre polynomial
            degree = self.polynomial_kernel_degree  # Replace with your degree value
            length = self.units
            domain = (0, length - 1)
            basis = np.zeros((length, degree))
            for row in range(length):  # Calculate Legendre polynomials for each row
                for col in range(degree):
                    polynomial = np.polynomial.legendre.Legendre.basis(col, domain)
                    # Evaluate the Legendre polynomial at the desired point (0.0 for Legendre polynomials)
                    basis[row, col] = polynomial(row)
            self.kernel_basis = self.add_weight("basis", shape=(length, degree), initializer='ones',
                                                trainable=False)
            self.kernel_basis.assign(basis)

    def call(self, inputs, **kwargs):
        if self.polynomial_kernel_degree is None:
            z = self.kernel * inputs
        else:
            kernel = self.kernel_basis @ self.kernel_coef
            self.kernel.assign(tf.reshape(kernel, (-1,)))
            z = tf.reshape(kernel, (-1,)) * inputs
        return z

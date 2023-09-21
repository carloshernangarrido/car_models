import tensorflow as tf


class MeanLessMeanSquaredError(tf.losses.Loss):
    def call(self, y_true, y_pred):
        y_true = y_true - tf.reduce_mean(y_true)
        y_pred = y_pred - tf.reduce_mean(y_pred)
        return tf.reduce_mean(tf.math.square(y_pred - y_true), axis=-1)


# Define a custom Total Variation regularizer
class TotalVariationRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, x):
        # Calculate the total variation loss for the kernel
        regularization = tf.reduce_sum(tf.image.total_variation(x))
        return self.weight * regularization

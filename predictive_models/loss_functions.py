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


# Custom loss function for enforcing the "same road"
def same_road_loss(y_true, y_pred):
    regularizer_abs_error_diff = 0.1
    # Extract conv11 and conv21 from y_pred
    conv11 = y_pred[0]
    conv12 = y_pred[1]
    conv21 = y_pred[2]
    conv22 = y_pred[3]
    # Calculate the squared difference between estimations of the same road
    mean_squared_diff_road1 = tf.reduce_mean(tf.square(conv11 - conv21))
    mean_squared_diff_road2 = tf.reduce_mean(tf.square(conv12 - conv22))
    error_sum = mean_squared_diff_road1 + mean_squared_diff_road2
    error_abs_diff = tf.abs(mean_squared_diff_road1 - mean_squared_diff_road2)
    # Compute the mean squared difference
    return error_sum + regularizer_abs_error_diff * error_abs_diff


class SameRoadLossSimple:
    def __init__(self, amplitude_regularization: float = 1.0):
        self.amplitude_regularization = amplitude_regularization

    def __call__(self, y_true, y_pred, *args, **kwargs):
        # Compute the mean squared difference between the two predicted outputs
        shape_loss = tf.reduce_mean(tf.square(y_pred[0] - y_pred[1]))
        amplitude_loss = -tf.reduce_min(y_pred)
        return shape_loss + self.amplitude_regularization * amplitude_loss


class SameRoadLossCallback(tf.keras.callbacks.Callback):
    def __init__(self, x_val, amplitude_regularization=1.0):
        super(SameRoadLossCallback, self).__init__()
        self.amplitude_regularization = amplitude_regularization
        self.x_val = x_val

    def on_epoch_end(self, epoch, logs=None):
        # Get the model
        model = self.model

        # Compute the predictions
        y_pred = model.predict(self.x_val)

        # Calculate the loss terms
        shape_loss = tf.reduce_mean(tf.square(y_pred[0] - y_pred[1]))
        amplitude_loss = -tf.reduce_min(y_pred)
        amplitude_loss_term = self.amplitude_regularization * amplitude_loss

        # Calculate the total loss
        total_loss = shape_loss + amplitude_loss_term

        # Print the loss terms
        print(f"Epoch {epoch + 1} - Shape term: {shape_loss:.4f}, Amplitude term: {amplitude_loss_term:.4f}, "
              f"Total Loss: {total_loss:.4f}")


class InformedSpeedSameRoadLossSimple:
    def __init__(self, amplitude_regularization: float = 1.0):
        self.amplitude_regularization = amplitude_regularization

    def __call__(self, y_true, y_pred, *args, **kwargs):
        """
        Compute the mean squared difference between the two predicted outputs accounting for the speeds and assuming
        the same road profile, i.e. the same space history not the same time history
        :param y_true: Not used. This an unsupervised approach.
        :param y_pred: It is assumed to be = [diag1, diag2, input_s1, input_s2]
        :param args:
        :param kwargs:
        :return:
        """
        diag1 = y_pred[0]
        diag2 = y_pred[1]
        s1 = y_true[0]
        s2 = y_true[1]
        s_max = tf.reduce_max([s1, s2])
        size = diag1.shape[0]

        size_s1_smax = tf.squeeze((tf.cast(size * s1 / s_max, dtype=tf.int32)))
        size_s2_smax = tf.squeeze((tf.cast(size * s2 / s_max, dtype=tf.int32)))

        diag1_resized = tf.reshape(tf.image.resize(tf.reshape(diag1[0:size_s1_smax], (-1, 1, 1)), (size, 1)), (-1,))
        diag2_resized = tf.reshape(tf.image.resize(tf.reshape(diag2[0:size_s2_smax], (-1, 1, 1)), (size, 1)), (-1,))

        shape_loss = tf.reduce_mean(tf.square(diag1_resized - diag2_resized))
        amplitude_loss = -tf.reduce_min(y_pred)
        amplitude_loss_term = self.amplitude_regularization * amplitude_loss
        return shape_loss + amplitude_loss_term

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
        amplitude_loss = tf.abs(tf.reduce_mean(tf.square(y_pred[0])) - tf.reduce_mean(tf.square(y_pred[1])))
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
        amplitude_loss = tf.abs(tf.reduce_mean(tf.square(y_pred[0])) - tf.reduce_mean(tf.square(y_pred[1])))

        # Calculate the total loss
        total_loss = shape_loss + self.amplitude_regularization * amplitude_loss

        # Print the loss terms
        print(f"Epoch {epoch + 1} - Shape Loss: {shape_loss:.4f}, Amplitude Loss: {amplitude_loss:.4f}, Total Loss: {total_loss:.4f}")

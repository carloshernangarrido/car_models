import numpy as np
import matplotlib.pyplot as plt


def frf_filter(time, input_, frf_freqs, frf_vals, plot: bool = False):
    delta_t = time[1] - time[0]

    # Perform the Fourier transform on the input signal
    input_fd = np.fft.fft(input_)
    input_freqs = np.fft.fftfreq(len(input_fd), delta_t)

    # Initialize the output frequency domain representation
    frf_new_vals = np.zeros_like(input_freqs)

    # Find the index of the closest frequency in frf_freq for each frequency in the input signal
    for freq in input_freqs:
        if freq >= 0:
            idx = np.argmin(np.abs(frf_freqs - freq))
            frf_new_vals[(np.where(input_freqs == freq))[0][0]] = frf_vals[idx]
        else:
            idx = np.argmin(np.abs(frf_freqs - (-freq)))
            frf_new_vals[(np.where(input_freqs == freq))[0][0]] = frf_vals[idx]

    # Apply the filter in the frequency domain
    output_fd = input_fd * frf_new_vals

    # Perform the inverse Fourier transform to obtain the time domain output
    output = np.real(np.fft.ifft(output_fd))

    if plot:
        plt.figure()
        plt.plot(frf_freqs, frf_vals)
        plt.plot(input_freqs, frf_new_vals)

        fig, ax = plt.subplots(2, 1)
        ax[0].plot(input_freqs, np.abs(input_fd), label='input')
        ax[1].plot(input_freqs, np.angle(input_fd))
        ax[0].plot(input_freqs, np.abs(output_fd), label='output')
        ax[1].plot(input_freqs, np.angle(output_fd))
    return output


def remove_mean_and_scale(road12, window_length=100, scale=2.0):
    """
    Remove the mean and scale the input signal using a moving window.

    Parameters:
    - road12: numpy array, the input signal.
    - window_length: int, the length of the moving window for mean removal.
    - scale: float, reference magnitude to calculate the scaling factor.

    Returns:
    - processed_signal: numpy array, the signal after mean removal and scaling.
    """
    if window_length < 0:
        raise ValueError("Window length must be a positive integer. If it is 0, moving mean is not removed")
    if scale <= 0:
        raise ValueError("scale must be a positive integer.")

    # Remove the mean using a moving window
    if window_length > 0:
        road12_mean_removed = road12 - np.convolve(road12, np.ones(window_length) / window_length, mode='same')
    else:
        road12_mean_removed = road12

    # Scale the signal
    scale_factor = scale / np.max(np.abs(road12_mean_removed))
    processed_signal = road12_mean_removed * scale_factor

    return processed_signal

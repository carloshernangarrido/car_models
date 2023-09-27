import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from utils.filters import remove_mean_and_scale, frf_filter


def modelsol2meas(model, roadvertacc, roadvertheight, timesteps_skip=1):
    wheelvertacc = roadvertacc + model.accelerations(1, append=0)
    carbodyvertacc = roadvertacc + model.accelerations(2, append=0)
    wheelvertheight = roadvertheight + model.displacements(1)
    carbodyvertheight = roadvertheight + model.displacements(2)
    return wheelvertacc[0::timesteps_skip], carbodyvertacc[0::timesteps_skip], \
           wheelvertheight[0::timesteps_skip], carbodyvertheight[0::timesteps_skip]


def plot_modelresults_timedomain(t_vector, roadvertheight, roadvertacc, model, show: bool = True):
    wheelvertacc, carbodyvertacc, wheelvertheight, carbodyvertheight = modelsol2meas(model, roadvertacc, roadvertheight)

    fig, ax = plt.subplots(2, 1, sharex='col')
    ax[0].set_ylabel('acelerations (m/s2)')
    ax[0].plot(t_vector, roadvertacc, label='road')
    ax[0].plot(t_vector, wheelvertacc, label='wheel')
    ax[0].plot(t_vector, carbodyvertacc, label='car body')
    ax[0].legend()
    ax[1].set_ylabel('height (m)')
    ax[1].plot(t_vector, roadvertheight, label='road')
    ax[1].plot(t_vector, wheelvertheight, label='wheel')
    ax[1].plot(t_vector, carbodyvertheight, label='car body')
    ax[1].legend()
    if show:
        plt.show()


def plot_modelresults_frequencydomain(t_vector, roadvertheight, roadvertacc, model, f_lim: float = 100,
                                      show: bool = True):
    wheelvertacc, carbodyvertacc, wheelvertheight, carbodyvertheight = modelsol2meas(model, roadvertacc, roadvertheight)

    freq = np.fft.fftfreq(len(t_vector), t_vector[1] - t_vector[0])
    roadvertheight_freq = np.fft.fft(roadvertheight)[:len(freq)//2]
    roadvertacc_freq = np.fft.fft(roadvertacc)[:len(freq)//2]

    wheelvertacc_freq = np.fft.fft(wheelvertacc)[:len(freq)//2]
    carbodyvertacc_freq = np.fft.fft(carbodyvertacc)[:len(freq)//2]
    wheelvertheight_freq = np.fft.fft(wheelvertheight)[:len(freq)//2]
    carbodyvertheight_freq = np.fft.fft(carbodyvertheight)[:len(freq)//2]

    freq = freq[:len(freq)//2]

    fig, ax = plt.subplots(2, 1, sharex='col')
    ax[0].set_ylabel('acelerations (m/s2)')
    ax[0].plot(freq, np.abs(roadvertacc_freq), label='road')
    ax[0].plot(freq, np.abs(wheelvertacc_freq), label='wheel')
    ax[0].plot(freq, np.abs(carbodyvertacc_freq), label='car body')
    ax[0].legend()
    ax[1].set_ylabel('height (m)')
    ax[1].plot(freq, np.abs(roadvertheight_freq), label='road')
    ax[1].plot(freq, np.abs(wheelvertheight_freq), label='wheel')
    ax[1].plot(freq, np.abs(carbodyvertheight_freq), label='car body')
    ax[1].set_xlim((0, f_lim))
    ax[1].set_yscale('log')
    ax[1].set_xlabel('frequency (Hz)')

    ax[1].legend()
    if show:
        plt.show()


def plot_modelresults_timefrequencydomain(t_vector, roadvertheight, roadvertacc, model, f_lim: float = 100,
                                          batch_length_s: float = 10, show: bool = True, return_batches: bool = False,
                                          separate_plot: bool = False, plot_wheel: bool = False,
                                          smoothing_window_Hz: float = 0.0):
    wheelvertacc, carbodyvertacc, wheelvertheight, carbodyvertheight = modelsol2meas(model, roadvertacc, roadvertheight)

    n_batches = int(t_vector[-1] // batch_length_s)
    batch_length = int(len(t_vector) // n_batches)

    batches = []
    for i_batch in range(n_batches):
        i_ini, i_fin = i_batch * batch_length, (i_batch+1)*batch_length
        freq = np.fft.fftfreq(len(t_vector[i_ini:i_fin]), t_vector[1] - t_vector[0])
        freq = freq[:len(freq) // 2]
        winlen = 1 if smoothing_window_Hz == 0 else int(smoothing_window_Hz//(freq[1]-freq[0]))
        poly_order = 0 if smoothing_window_Hz == 0 else 3
        batches.append({'t_vector': t_vector[i_ini:i_fin],
                        'freq': freq,
                        'roadvertacc': roadvertacc[i_ini:i_fin],
                        'wheelvertacc': wheelvertacc[i_ini:i_fin],
                        'carbodyvertacc': carbodyvertacc[i_ini:i_fin],
                        'roadvertheight': roadvertheight[i_ini:i_fin],
                        'wheelvertheight': wheelvertheight[i_ini:i_fin],
                        'carbodyvertheight': carbodyvertheight[i_ini:i_fin],
                        'roadvertacc_freq': sp.signal.savgol_filter(np.fft.fft(roadvertacc[i_ini:i_fin])[:len(freq)], winlen, poly_order),
                        'wheelvertacc_freq': sp.signal.savgol_filter(np.fft.fft(wheelvertacc[i_ini:i_fin])[:len(freq)], winlen, poly_order),
                        'carbodyvertacc_freq': sp.signal.savgol_filter(np.fft.fft(carbodyvertacc[i_ini:i_fin])[:len(freq)], winlen, poly_order),
                        'roadvertheight_freq': sp.signal.savgol_filter(np.fft.fft(roadvertheight[i_ini:i_fin])[:len(freq)], winlen, poly_order),
                        'wheelvertheight_freq': sp.signal.savgol_filter(np.fft.fft(wheelvertheight[i_ini:i_fin])[:len(freq)], winlen, poly_order),
                        'carbodyvertheight_freq': sp.signal.savgol_filter(np.fft.fft(carbodyvertheight[i_ini:i_fin])[:len(freq)], winlen, poly_order)})
        if separate_plot:
            fig, ax = plt.subplots(2, 1, sharex='col')
            ax[0].set_ylabel('acelerations (m/s2)')
            ax[0].plot(freq, np.abs(batches[-1]['roadvertacc_freq']), label='road')
            if plot_wheel:
                ax[0].plot(freq, np.abs(batches[-1]['wheelvertacc_freq']), label='wheel')
            ax[0].plot(freq, np.abs(batches[-1]['carbodyvertacc_freq']), label='car body')
            ax[0].legend()
            ax[1].set_ylabel('height (m)')
            ax[1].plot(freq, np.abs(batches[-1]['roadvertheight_freq']), label='road')
            if plot_wheel:
                ax[1].plot(freq, np.abs(batches[-1]['wheelvertheight_freq']), label='wheel')
            ax[1].plot(freq, np.abs(batches[-1]['carbodyvertheight_freq']), label='car body')
            ax[1].set_xlim((0, f_lim))
            ax[1].set_xlabel('frequency (Hz)')
            ax[1].legend()
            ax[1].set_yscale('log')

    # Create a 3D figure.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for batch_index, batch in enumerate(batches):  # Plot the 3D lines (waterfall) for each batch.
        freq = batch['freq']
        endtime = (np.full_like(freq, batch_index)+1) * batch_length_s
        ax.plot(freq, endtime, np.abs(batch['roadvertacc_freq']), label=f'road' if batch_index == 0 else None, color='b', linewidth=2)
        if plot_wheel:
            ax.plot(freq, endtime, np.abs(batch['wheelvertacc_freq']), label=f'wheel' if batch_index == 0 else None, color='r', linewidth=2)
        ax.plot(freq, endtime, np.abs(batch['carbodyvertacc_freq']), label=f'car body' if batch_index == 0 else None, color='g', linewidth=2)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_xlim((0, f_lim))
    ax.set_ylabel('End time')
    ax.set_zlabel('Magnitude of vertical acceleration')
    ax.legend()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for batch_index, batch in enumerate(batches):  # Plot the 3D lines (waterfall) for each batch.
        freq = batch['freq']
        endtime = (np.full_like(freq, batch_index)+1) * batch_length_s
        ax.plot(freq, endtime, np.abs(batch['roadvertheight_freq']), label=f'road' if batch_index == 0 else None, color='b', linewidth=2)
        if plot_wheel:
            ax.plot(freq, endtime, np.abs(batch['wheelvertheight_freq']), label=f'wheel' if batch_index == 0 else None, color='r', linewidth=2)
        ax.plot(freq, endtime, np.abs(batch['carbodyvertheight_freq']), label=f'car body' if batch_index == 0 else None, color='g', linewidth=2)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_xlim((0, f_lim))
    ax.set_ylabel('End time')
    ax.set_zlabel('Magnitude of vertical height')

    ax.legend()

    if show:
        plt.show()
    if return_batches:
        return batches


def plot_comparison_td(time_road12, road12, carbodyvertacc1, carbodyvertacc2,
                       carbodyvertacc1_filt, carbodyvertacc2_filt,
                       int2_acc1, int2_acc2, int2_acc1_filt, int2_acc2_filt):
    fig, ax = plt.subplots(2, 2, sharex='all')
    ax[0, 0].set_title('measured car accelerations')
    ax[0, 0].plot(time_road12, carbodyvertacc1, label='car1')
    ax[0, 0].plot(time_road12, carbodyvertacc2, label='car2', alpha=0.5)
    ax[0, 0].plot(time_road12, remove_mean_and_scale(np.diff(np.diff(road12, append=0), append=0),
                                                     window_length=0, scale=np.max(np.abs(carbodyvertacc1))),
                  label='road ', color='g')
    ax[1, 0].set_title('equalized car accelerations')
    ax[1, 0].plot(time_road12, carbodyvertacc1_filt, label='car1')
    ax[1, 0].plot(time_road12, carbodyvertacc2_filt, label='car2', alpha=0.5)
    ax[1, 0].plot(time_road12, remove_mean_and_scale(np.diff(np.diff(road12, append=0), append=0),
                                                     window_length=0, scale=np.max(np.abs(carbodyvertacc1_filt))),
                  label='road ', color='g')
    ax[1, 0].legend()
    ax[0, 1].set_title('displacements from measured accelerations')
    ax[0, 1].plot(time_road12, int2_acc1, label='car1')
    ax[0, 1].plot(time_road12, int2_acc2, label='car2', alpha=0.5)
    ax[0, 1].plot(time_road12, remove_mean_and_scale(road12, window_length=100, scale=np.max(int2_acc1)),
                  label='scaled road', color='g')
    ax[1, 1].set_title('displacements from equalized accelerations')
    ax[1, 1].plot(time_road12, int2_acc1_filt, label='car1')
    ax[1, 1].plot(time_road12, int2_acc2_filt, label='car2', alpha=0.5)
    ax[1, 1].plot(time_road12, remove_mean_and_scale(road12, window_length=100, scale=np.max(int2_acc1_filt)),
                  label='scaled road', color='g')
    ax[1, 1].legend()


def plot_eq_fd_td_results(eq_l, time_road12=None, carbodyvertacc1=None, carbodyvertacc2=None, road12=None):
    # kernels
    fig, ax = plt.subplots(2, 3, sharex='all')
    ax[1, 0].set_title('equalizers')
    kernel1 = next((var for var in eq_l.model.layers[2].variables if 'kernel:' in var.name), None)
    kernel2 = next((var for var in eq_l.model.layers[3].variables if 'kernel:' in var.name), None)
    ax[1, 0].plot(eq_l.freq, kernel1, label='for car1')
    ax[1, 0].plot(eq_l.freq, kernel2, label='for car2')
    ax[1, 0].set_xlabel('frequency (Hz)')
    ax[1, 0].legend()

    # Spectrograms Linear scale
    f = np.linspace(1, kernel1.shape[0], kernel1.shape[0])  # Valores de x de 1 a 49 # Crear una matriz que represente la hiperbola y = 1/x
    diff_fd = f ** 2
    acc_spectrogram_road = eq_l.y_train_fd * diff_fd
    vmin, vmax = np.min(acc_spectrogram_road), np.max(acc_spectrogram_road)
    ax[0, 0].imshow(acc_spectrogram_road, extent=[0, eq_l.freq[-1], 0, eq_l.time[-1]], aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    ax[0, 0].set_ylabel('time (s)')
    # cbar_road = fig_road.colorbar(cax_road)
    ax[0, 0].set_title('acc. spectrogram of the actual road')
    acc_car1, acc_car2 = eq_l.x1_train_fd, eq_l.x2_train_fd
    vmin, vmax = np.min(np.min((acc_car1, acc_car2))), np.max(np.max((acc_car1, acc_car2)))

    ax[0, 1].imshow(acc_car1, extent=[0, eq_l.freq[-1], 0, eq_l.time[-1]], aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    ax[0, 1].set_title('acc. at car 1')
    ax[0, 2].imshow(acc_car2, extent=[0, eq_l.freq[-1], 0, eq_l.time[-1]], aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    ax[0, 2].set_title('acc. at car 2')
    y_pred = eq_l.predict()

    vmin, vmax = np.min(np.min(y_pred)), np.max(np.max(y_pred))
    ax[1, 1].imshow(y_pred[0], extent=[0, eq_l.freq[-1], 0, eq_l.time[-1]], aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    ax[1, 1].set_title('equalized acc. at car 1')
    ax[1, 2].imshow(y_pred[1], extent=[0, eq_l.freq[-1], 0, eq_l.time[-1]], aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    ax[1, 2].set_title('equalized acc. at car 2')
    plt.show()

    # Plot time domain
    carbodyvertacc1_filt = frf_filter(time=time_road12, input_=carbodyvertacc1, frf_freqs=eq_l.freq, frf_vals=kernel1)
    carbodyvertacc2_filt = frf_filter(time=time_road12, input_=carbodyvertacc2, frf_freqs=eq_l.freq, frf_vals=kernel2)
    frf_int2 = np.zeros_like(eq_l.freq)
    frf_int2[1:] = 1 / (2 * np.pi * eq_l.freq[1:]) ** 2
    int2_acc1 = frf_filter(time=time_road12, input_=carbodyvertacc1, frf_freqs=eq_l.freq, frf_vals=frf_int2)
    int2_acc2 = frf_filter(time=time_road12, input_=carbodyvertacc2, frf_freqs=eq_l.freq, frf_vals=frf_int2)
    int2_acc1_filt = frf_filter(time=time_road12, input_=carbodyvertacc1_filt, frf_freqs=eq_l.freq, frf_vals=frf_int2)
    int2_acc2_filt = frf_filter(time=time_road12, input_=carbodyvertacc2_filt, frf_freqs=eq_l.freq, frf_vals=frf_int2)
    plot_comparison_td(time_road12, road12, carbodyvertacc1, carbodyvertacc2,
                       carbodyvertacc1_filt, carbodyvertacc2_filt,
                       int2_acc1, int2_acc2, int2_acc1_filt, int2_acc2_filt)
    plt.show()

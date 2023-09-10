import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


def modelsol2meas(model, roadvertacc, roadvertheight):
    wheelvertacc = roadvertacc + model.accelerations(1, append=0)
    carbodyvertacc = roadvertacc + model.accelerations(2, append=0)
    wheelvertheight = roadvertheight + model.displacements(1)
    carbodyvertheight = roadvertheight + model.displacements(2)
    return wheelvertacc, carbodyvertacc, wheelvertheight, carbodyvertheight


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
    ax[1].set_xlabel('frequency (Hz)')

    ax[1].legend()
    if show:
        plt.show()


def plot_modelresults_timefrequencydomain(t_vector, roadvertheight, roadvertacc, model, f_lim: float = 100,
                                          batch_length_s: float = 10, show: bool = True, return_batches: bool = False):
    wheelvertacc, carbodyvertacc, wheelvertheight, carbodyvertheight = modelsol2meas(model, roadvertacc, roadvertheight)

    batch_length = int(t_vector[-1]//batch_length_s)

    batches = []
    for i_batch in range(batch_length):
        i_ini, i_fin = i_batch * batch_length, (i_batch+1)*batch_length
        freq = np.fft.fftfreq(len(t_vector[i_ini:i_fin]), t_vector[1] - t_vector[0])
        freq = freq[:len(freq) // 2]
        batches.append({'t_vector': t_vector[i_ini:i_fin],
                        'freq': freq,
                        'roadvertacc': roadvertacc[i_ini:i_fin],
                        'wheelvertacc': wheelvertacc[i_ini:i_fin],
                        'carbodyvertacc': carbodyvertacc[i_ini:i_fin],
                        'roadvertheight': roadvertheight[i_ini:i_fin],
                        'wheelvertheight': wheelvertheight[i_ini:i_fin],
                        'carbodyvertheight': carbodyvertheight[i_ini:i_fin],
                        'roadvertacc_freq': np.fft.fft(roadvertacc[i_ini:i_fin])[:len(freq)],
                        'wheelvertacc_freq': np.fft.fft(wheelvertacc[i_ini:i_fin])[:len(freq)],
                        'carbodyvertacc_freq': np.fft.fft(carbodyvertacc[i_ini:i_fin])[:len(freq)],
                        'roadvertheight_freq': np.fft.fft(roadvertheight[i_ini:i_fin])[:len(freq)],
                        'wheelvertheight_freq': np.fft.fft(wheelvertheight[i_ini:i_fin])[:len(freq)],
                        'carbodyvertheight_freq': np.fft.fft(carbodyvertheight[i_ini:i_fin])[:len(freq)]})

        fig, ax = plt.subplots(2, 1, sharex='col')
        ax[0].set_ylabel('acelerations (m/s2)')
        ax[0].plot(freq, np.abs(batches[-1]['roadvertacc_freq']), label='road')
        ax[0].plot(freq, np.abs(batches[-1]['wheelvertacc_freq']), label='wheel')
        ax[0].plot(freq, np.abs(batches[-1]['carbodyvertacc_freq']), label='car body')
        ax[0].legend()
        ax[1].set_ylabel('height (m)')
        ax[1].plot(freq, np.abs(batches[-1]['roadvertheight_freq']), label='road')
        ax[1].plot(freq, np.abs(batches[-1]['wheelvertheight_freq']), label='wheel')
        ax[1].plot(freq, np.abs(batches[-1]['carbodyvertheight_freq']), label='car body')
        ax[1].set_xlim((0, f_lim))
        ax[1].set_xlabel('frequency (Hz)')
        ax[1].legend()

    if show:
        plt.show()
    if return_batches:
        return batches

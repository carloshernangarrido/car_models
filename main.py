import itertools
import pickle
from copy import copy

import matplotlib.pyplot as plt
import numpy as np

from predictive_models.frequency_domain import EqualizerLearner
from predictive_models.system_identification import CarDynamicsIdentification, JointIdentification, \
    JointIdentificationSimple
from roadprofile_management import loaders
from roadprofile_management.utils import get_roadvertaccheight
from models.chain_like import Mesh, Constraint, Model, Load
from visualization.ploting_results import plot_modelresults_timedomain, plot_modelresults_frequencydomain, \
    plot_modelresults_timefrequencydomain, modelsol2meas


if __name__ == "__main__":
    from input_data import k_s, c_s, m_s, k_t, c_t, m_u, roaddata_folder, speed_descr, delta_t, file_qty, \
        timesteps_skip, generate_dataset

    if generate_dataset:
        # Road excitation preparing
        t_vector = np.arange(0, speed_descr.T, delta_t)
        loader = loaders.Road(roaddata_folder, file_ini=0, file_qty=file_qty, plot=False, tol=1)
        roadvertaccs, roadvertheights = [], []
        for lane in [1.0, 2.0, 3.0]:
            roadvertacc, roadvertheight = get_roadvertaccheight(loader, speed_descr, t_vector, lane=lane, plot=False)
            if lane == 3.0:  # just to make road 2 more different
                roadvertaccs.append(roadvertacc[::-1])
                roadvertheights.append(roadvertheight[::-1])
            else:
                roadvertaccs.append(roadvertacc)
                roadvertheights.append(roadvertheight)

        # Simulations
        models = []
        for m_s_var in [1.5, 1.0, 0.5]:
            model_roads = []
            for roadvertacc_ in roadvertaccs:
                m_s_ = m_s_var * m_s
                load_u = Load(dof_s=1, force=m_u * roadvertacc_, t=t_vector)
                load_s = Load(dof_s=2, force=m_s_ * roadvertacc_, t=t_vector)
                mesh = Mesh(n_dof=3, length=1)
                mesh.fill_elements('k', [k_t, k_s])
                mesh.fill_elements('c', [c_t, c_s])
                const = Constraint(dof_s=0)
                model = Model(mesh=mesh, constraints=const, lumped_masses=[1, m_u, m_s_], loads=[load_u, load_s],
                              options={'t_vector': t_vector, 'method': 'RK23'})
                model.linearize()
                model.lsim()
                model_roads.append(model.deepcopy())
            models.append(model_roads)


        # Inverse identification of two cars without knowing the road, just it is the same road
        # Training dataset
        _, carbodyvertacc11, _, _ = \
            modelsol2meas(models[1][1], roadvertaccs[1], roadvertheights[1], timesteps_skip=timesteps_skip)
        _, carbodyvertacc12, _, _ = \
            modelsol2meas(models[1][2], roadvertaccs[2], roadvertheights[2], timesteps_skip=timesteps_skip)
        _, carbodyvertacc21, _, _ = \
            modelsol2meas(models[2][1], roadvertaccs[1], roadvertheights[1], timesteps_skip=timesteps_skip)
        _, carbodyvertacc22, _, _ = \
            modelsol2meas(models[2][2], roadvertaccs[2], roadvertheights[2], timesteps_skip=timesteps_skip)
        time = t_vector[0::timesteps_skip]

        with open("dataset.pkl", "wb") as file:
            pickle.dump([time, carbodyvertacc11, carbodyvertacc12, carbodyvertacc21, carbodyvertacc22, roadvertheights],
                        file)
    else:
        with open("dataset.pkl", "rb") as file:
            time, carbodyvertacc11, carbodyvertacc12, carbodyvertacc21, carbodyvertacc22, roadvertheights = \
                pickle.load(file)

    plt.figure()
    plt.plot(time, carbodyvertacc11, label='car1 road1')
    plt.plot(time, carbodyvertacc12, label='car1 road2')
    plt.plot(time, carbodyvertacc21, label='car2 road1')
    plt.plot(time, carbodyvertacc22, label='car2 road2')
    plt.legend()
    plt.show()

    carbodyvertacc1 = np.hstack((carbodyvertacc11, carbodyvertacc12))
    carbodyvertacc2 = np.hstack((carbodyvertacc21, carbodyvertacc22))
    road12 = np.hstack((roadvertheights[1][0::timesteps_skip], roadvertheights[2][0::timesteps_skip]))
    time_road12 = np.hstack((time, time + time[-1]))
    fig, ax = plt.subplots(2, 1, sharex='col')
    ax[0].plot(time_road12, road12, label='road')
    ax[1].plot(time_road12, carbodyvertacc1, label='carbodyvertacc1')
    ax[1].plot(time_road12, carbodyvertacc2, alpha=0.5, label='carbodyvertacc2')
    ax[1].set_xlabel('time (s)')
    plt.legend()
    plt.show()

    eq_l = EqualizerLearner(x1=carbodyvertacc1,
                            x2=carbodyvertacc2, batch_length_s=1.0,
                            y=road12, amplitude_regularization=1000.0,
                            time=time_road12, learning_rate=0.0005, epochs=50)  # x_car
    eq_l.build_model()
    eq_l.fit_model()

    # kernels
    plt.figure()
    plt.title('equalizers')
    plt.plot(eq_l.freq, eq_l.model.weights[0], label='for car1')
    plt.plot(eq_l.freq, eq_l.model.weights[1], label='for car2')
    plt.xlabel('frequency (Hz)')
    plt.legend()

    # Spectrograms
    f = np.linspace(1, 49, 49)  # Valores de x de 1 a 49 # Crear una matriz que represente la hiperbola y = 1/x
    diff_fd = f**2
    acc_spectrogram_road = eq_l.y_train_fd * diff_fd
    acc_spectrogram_road_log = np.log(acc_spectrogram_road)
    vmin, vmax = np.min(acc_spectrogram_road_log), np.max(acc_spectrogram_road_log)
    fig_road = plt.figure()
    cax_road = plt.imshow(acc_spectrogram_road_log, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    cbar_road = fig_road.colorbar(cax_road)
    plt.title('acc. spectrogram of the actual road')

    acc_car1_log, acc_car2_log = np.log(eq_l.x1_train_fd), np.log(eq_l.x2_train_fd)
    vmin, vmax = np.min(np.min((acc_car1_log, acc_car2_log))), np.max(np.max((acc_car1_log, acc_car2_log)))
    fig_, ax = plt.subplots(1, 2)
    ax[0].imshow(acc_car1_log, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    ax[0].set_title('acc. at car 1')
    ax[1].imshow(acc_car2_log, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    ax[1].set_title('acc. at car 2')

    # Plot prediction
    y_pred_log = np.log(eq_l.predict())
    fig_, ax = plt.subplots(1, 2)
    vmin, vmax = np.min(np.min(y_pred_log)), np.max(np.max(y_pred_log))
    ax[0].imshow(y_pred_log[0], aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    ax[0].set_title('equalized acc. at car 1')
    ax[1].imshow(y_pred_log[1], aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    ax[1].set_title('equalized acc. at car 2')
    plt.show()

    a = 0
    # # Inverse identification of cars knowing the road
    # _, carbodyvertacc1, _, _ = \
    #     modelsol2meas(models[1][1], roadvertaccs[0], roadvertheights[0], timesteps_skip=timesteps_skip)
    # true_car1 = CarDynamicsIdentification(t_vector[0::timesteps_skip], roadvertheights[1][0::timesteps_skip],
    #                                       carbodyvertacc1, normalize=False, learning_rate=0.0005, epochs=50,
    #                                       kernel_length_s=0.5, batch_length_s=1.0)
    # true_car1.inverse_identification(plot=True)
    # _, carbodyvertacc2, _, _ = \
    #     modelsol2meas(models[2][1], roadvertaccs[0], roadvertheights[0], timesteps_skip=timesteps_skip)
    # true_car2 = CarDynamicsIdentification(t_vector[0::timesteps_skip], roadvertheights[1][0::timesteps_skip],
    #                                       carbodyvertacc2, normalize=False, learning_rate=0.0005, epochs=50,
    #                                       kernel_length_s=0.5, batch_length_s=1.0)
    # true_car2.inverse_identification(plot=True)
    #
    # y_pred = [true_car1.inverse_y_pred, true_car2.inverse_y_pred]
    # # Plots!
    # roads = [np.zeros((y_pred_.shape[1] + y_pred_.shape[0],)) for y_pred_ in y_pred]
    # y_pred_cols = y_pred[0].shape[1]
    # road_length = y_pred[0].shape[1] + y_pred[0].shape[0] - 1
    # for i_road, y_pred_ in zip(range(len(y_pred)), y_pred):
    #     for i in range(0, road_length - y_pred_cols, y_pred_cols):
    #         roads[i_road][i:i + y_pred_cols] = y_pred_[i, :, 0]
    # fig, ax = plt.subplots(2, 1)
    # ax[0].plot(roads[0], label='car1 road 1')
    # ax[0].plot(roads[1], label='car2 road 1')
    # ax[0].plot(roadvertheights[1][0::timesteps_skip], label='road 1')
    # ax[1].plot(roadvertheights[2][0::timesteps_skip], label='road 2')
    # ax[0].legend()
    # ax[1].legend()
    # plt.show()

    # ji = JointIdentification(x11=carbodyvertacc11, x12=carbodyvertacc12, x21=carbodyvertacc21, x22=carbodyvertacc22,
    #                          time=time, learning_rate=0.0001, epochs=50)  # x_car_road
    # ji.build_model()
    # ji.fit_model()
    # y_pred = ji.predict()
    #
    # # Plots!
    # roads = [np.zeros((y_pred_.shape[1] + y_pred_.shape[0],)) for y_pred_ in y_pred]
    # y_pred_cols = y_pred[0].shape[1]
    # road_length = y_pred[0].shape[1] + y_pred[0].shape[0] - 1
    # for i_road, y_pred_ in zip(range(len(y_pred)), y_pred):
    #     for i in range(0, road_length - y_pred_cols, y_pred_cols):
    #         roads[i_road][i:i + y_pred_cols] = y_pred_[i, :, 0]
    # fig, ax = plt.subplots(2, 1)
    # ax[0].plot(roads[0], label='car1 road 1')
    # ax[1].plot(roads[1], label='car1 road 2')
    # ax[0].plot(roads[2], label='car2 road 1')
    # ax[1].plot(roads[3], label='car2 road 2')
    # ax[0].plot(roadvertheights[1][0::timesteps_skip], label='road 1')
    # ax[1].plot(roadvertheights[2][0::timesteps_skip], label='road 2')
    # ax[0].legend()
    # ax[1].legend()
    # plt.show()
    #
    # plt.figure()
    # plt.title('blind identified kernes')
    # plt.plot(np.reshape(ji.model.weights[0], (-1,)), label='car1')
    # plt.plot(np.reshape(ji.model.weights[1], (-1,)), label='car2')
    # plt.legend()
    # plt.ylim([-.005, .005])
    # plt.show()
    #
    # plt.figure()
    # plt.title('true kernels')
    # plt.plot(np.array(true_car1.inverse_model.weights[3]).reshape((-1,)), label='car1')
    # plt.plot(np.array(true_car2.inverse_model.weights[3]).reshape((-1,)), label='car2')
    # plt.ylim([-.005, .005])
    # plt.legend()
    # plt.show()
    #
    # #  # JointIdentificationSimple
    # ji_s = JointIdentificationSimple(x1=np.hstack((carbodyvertacc11, carbodyvertacc12)),
    #                                  x2=np.hstack((carbodyvertacc21, carbodyvertacc22)),
    #                                  time=np.hstack((time, time+time[-1])), learning_rate=0.0001, epochs=50,
    #                                  amplitude_regularization=100.)  # x_car
    # ji_s.build_model()
    # ji_s.fit_model()
    # y_pred_s = ji_s.predict()
    # roads_s = [np.zeros((y_pred_.shape[1] + y_pred_.shape[0],)) for y_pred_ in y_pred_s]
    # y_pred_cols_s = y_pred_s[0].shape[1]
    # road_length_s = y_pred_s[0].shape[1] + y_pred_s[0].shape[0] - 1
    # for i_road, y_pred_ in zip(range(len(y_pred_s)), y_pred_s):
    #     for i in range(0, road_length_s - y_pred_cols_s, y_pred_cols_s):
    #         roads_s[i_road][i:i + y_pred_cols_s] = y_pred_[i, :, 0]
    #
    # plt.figure()
    # plt.title('blind identified kernels (simple)')
    # plt.plot(np.reshape(ji_s.model.weights[0], (-1,)), label='car1')
    # plt.plot(np.reshape(ji_s.model.weights[1], (-1,)), label='car2')
    # plt.legend()
    # plt.ylim([-.005, .005])
    # plt.show()
    #
    # fig, ax = plt.subplots(2, 1)
    # ax[0].plot(ji_s.x1, label='accelerations of car1')
    # ax[0].plot(ji_s.x2, alpha=0.5, label='accelerations of car2')
    # ax[1].plot(roads_s[0], label='prediction with car1')
    # ax[1].plot(roads_s[1], alpha=0.5, label='prediction with car2')
    # road1_length = len(roadvertheights[1][0::timesteps_skip])
    # ax[1].plot(range(road1_length), roadvertheights[1][0::timesteps_skip], label='road1')
    # ax[1].plot(np.array(range(road1_length)) + road1_length, roadvertheights[2][0::timesteps_skip], label='road2')
    # ax[0].legend()
    # ax[1].legend()
    # plt.show()

    # parameter_variations = [0.8, 1.0, 1.2]  # equal, +20%
    # models = []
    # for var in itertools.product(parameter_variations, repeat=5):
    #     params = np.array(var) * np.array([k_s, c_s, m_s, k_t, m_u])
    #     k_s_, c_s_, m_s_, k_t_, m_u_ = params
    #
    #     load_u = Load(dof_s=1, force=m_u_ * roadvertacc, t=t_vector)
    #     load_s = Load(dof_s=2, force=m_s_ * roadvertacc, t=t_vector)
    #     mesh = Mesh(n_dof=3, length=1)
    #     mesh.fill_elements('k', [k_t_, k_s_])
    #     mesh.fill_elements('c', [c_t, c_s_])
    #     const = Constraint(dof_s=0)
    #     model = Model(mesh=mesh, constraints=const, lumped_masses=[1, m_u, m_s], loads=[load_u, load_s],
    #                   options={'t_vector': t_vector, 'method': 'RK23'})
    #     model.linearize()
    #     model.lsim()
    #     models.append(model.deepcopy())
    #
    # wheelvertacc, carbodyvertacc, wheelvertheight, carbodyvertheight = modelsol2meas(model, roadvertacc, roadvertheight)
    # car_id = CarDynamicsIdentification(t_vector, roadvertheight, carbodyvertacc)
    # car_id.direct_identification(plot=True)
    # car_id.inverse_identification(plot=True)
    #
    # plot_modelresults_timedomain(t_vector, roadvertheight, roadvertacc, model, show=False)
    # plot_modelresults_frequencydomain(t_vector, roadvertheight, roadvertacc, model, show=False)
    # batches = plot_modelresults_timefrequencydomain(t_vector, roadvertheight, roadvertacc, model, batch_length_s=10,
    #                                                 show=False, return_batches=True,
    #                                                 smoothing_window_Hz=10)
    # plt.show()
    ...

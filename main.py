import itertools
from copy import copy

import matplotlib.pyplot as plt
import numpy as np

from predictive_models.system_identification import CarDynamicsIdentification, JointIdentification
from roadprofile_management import loaders
from roadprofile_management.utils import get_roadvertaccheight
from models.chain_like import Mesh, Constraint, Model, Load
from visualization.ploting_results import plot_modelresults_timedomain, plot_modelresults_frequencydomain, \
    plot_modelresults_timefrequencydomain, modelsol2meas


if __name__ == "__main__":
    from input_data import k_s, c_s, m_s, k_t, c_t, m_u, roaddata_folder, speed_descr, delta_t, file_qty

    t_vector = np.arange(0, speed_descr.T, delta_t)
    loader = loaders.Road(roaddata_folder, file_ini=0, file_qty=file_qty, plot=False, tol=1)
    roadvertaccs, roadvertheights = [], []
    for lane in [1.0, 2.0, 3.0]:
        roadvertacc, roadvertheight = get_roadvertaccheight(loader, speed_descr, t_vector, lane=lane, plot=False)
        roadvertaccs.append(roadvertacc)
        roadvertheights.append(roadvertheight)

    models = []
    for m_s_var in [0.5, 1.0, 1.5]:
        model_roads = []
        for roadvertacc_ in roadvertaccs:
            m_s_ = m_s_var * m_s
            load_u = Load(dof_s=1, force=m_u * roadvertacc_, t=t_vector)
            load_s = Load(dof_s=2, force=m_s_ * roadvertacc_, t=t_vector)
            mesh = Mesh(n_dof=3, length=1)
            mesh.fill_elements('k', [k_t, k_s])
            mesh.fill_elements('c', [c_t, c_s])
            const = Constraint(dof_s=0)
            model = Model(mesh=mesh, constraints=const, lumped_masses=[1, m_u, m_s], loads=[load_u, load_s],
                          options={'t_vector': t_vector, 'method': 'RK23'})
            model.linearize()
            model.lsim()
            model_roads.append(model.deepcopy())
        models.append(model_roads)
    ...
    print(models)
    print('')



    ## initial guess of system model
    timesteps_skip = 10
    # wheelvertacc, carbodyvertacc, wheelvertheight, carbodyvertheight = \
    #     modelsol2meas(models[0][0], roadvertaccs[0], roadvertheights[0], timesteps_skip=timesteps_skip)
    # init_model = CarDynamicsIdentification(t_vector[0::timesteps_skip],
    #                                        roadvertheights[0][0::timesteps_skip], carbodyvertacc)
    # init_model.inverse_identification(plot=True)

    _, carbodyvertacc11, _, _ = \
        modelsol2meas(models[1][1], roadvertaccs[1], roadvertheights[1], timesteps_skip=timesteps_skip)
    _, carbodyvertacc12, _, _ = \
        modelsol2meas(models[1][2], roadvertaccs[2], roadvertheights[2], timesteps_skip=timesteps_skip)
    _, carbodyvertacc21, _, _ = \
        modelsol2meas(models[2][1], roadvertaccs[1], roadvertheights[1], timesteps_skip=timesteps_skip)
    _, carbodyvertacc22, _, _ = \
        modelsol2meas(models[2][2], roadvertaccs[2], roadvertheights[2], timesteps_skip=timesteps_skip)
    time = t_vector[0::timesteps_skip]
    plt.figure()
    plt.plot(time, carbodyvertacc11, label='car1 road1')
    plt.plot(time, carbodyvertacc12, label='car1 road2')
    plt.plot(time, carbodyvertacc21, label='car2 road1')
    plt.plot(time, carbodyvertacc22, label='car2 road2')
    plt.legend()
    plt.show()
    ji = JointIdentification(x11=carbodyvertacc11, x12=carbodyvertacc12, x21=carbodyvertacc21, x22=carbodyvertacc22,
                             time=time, learning_rate=0.001)  # x_car_road
    ji.build_model()
    ji.fit_model()
    y_pred = ji.predict()
    roads = [np.zeros((y_pred_.shape[1] + y_pred_.shape[0],)) for y_pred_ in y_pred]
    y_pred_cols = y_pred[0].shape[1]
    road_length = y_pred[0].shape[1] + y_pred[0].shape[0] - 1
    for i_road, y_pred_ in zip(range(len(y_pred)), y_pred):
        for i in range(0, road_length - y_pred_cols, y_pred_cols):
            roads[i_road][i:i + y_pred_cols] = y_pred_[i, :, 0]
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(roads[0], label='car1 road 1')
    ax[1].plot(roads[1], label='car1 road 2')
    ax[0].plot(roads[2], label='car2 road 1')
    ax[1].plot(roads[3], label='car2 road 2')
    ax[0].plot(roadvertheights[1][0::timesteps_skip], label='road 1')
    ax[1].plot(roadvertheights[2][0::timesteps_skip], label='road 2')
    ax[0].legend()
    ax[1].legend()
    plt.show()

    plt.figure()
    plt.plot(np.reshape(ji.model.weights[0], (-1,)))
    plt.plot(np.reshape(ji.model.weights[1], (-1,)))




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

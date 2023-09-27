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
from utils.filters import remove_mean_and_scale, frf_filter
from visualization.ploting_results import plot_modelresults_timedomain, plot_modelresults_frequencydomain, \
    plot_modelresults_timefrequencydomain, modelsol2meas, plot_comparison_td, plot_eq_fd_td_results

if __name__ == "__main__":
    from input_data import k_s, c_s, m_s, k_t, c_t, m_u, roaddata_folder, speed_descr, delta_t, file_qty, \
        file_ini, timesteps_skip, generate_dataset

    if generate_dataset:
        # Road excitation preparing
        t_vector = np.arange(0, speed_descr.T, delta_t)
        loader = loaders.Road(roaddata_folder, file_ini=file_ini, file_qty=file_qty, plot=False, tol=1)
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
        for m_s_var in [1.5, 1.0, 0.50]:
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
                            x2=carbodyvertacc2, batch_length_s=1.0, polynomial_kernel_degree=None,
                            y=road12, amplitude_regularization=1000,
                            time=time_road12, learning_rate=0.0005, epochs=100)  # x_car
    eq_l.build_model()
    eq_l.fit_model()

    plot_eq_fd_td_results(eq_l=eq_l, time_road12=time_road12, carbodyvertacc1=carbodyvertacc1,
                          carbodyvertacc2=carbodyvertacc2, road12=road12)
    a = 0

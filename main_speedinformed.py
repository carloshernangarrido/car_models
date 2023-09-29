import itertools
import joblib
from copy import copy

import matplotlib.pyplot as plt
import numpy as np

from predictive_models.frequency_domain import EqualizerLearner, SpeedInformedEqualizerLearner
from predictive_models.system_identification import CarDynamicsIdentification, JointIdentification, \
    JointIdentificationSimple
from roadprofile_management import loaders
from roadprofile_management.utils import get_roadvertaccheight
from models.chain_like import Mesh, Constraint, Model, Load
from utils.filters import remove_mean_and_scale, frf_filter
from visualization.ploting_results import plot_modelresults_timedomain, plot_modelresults_frequencydomain, \
    plot_modelresults_timefrequencydomain, modelsol2meas, plot_comparison_td, \
    plot_eq_fd_td_results, plot_sieq_fd_td_results


if __name__ == "__main__":
    from input_data_speedinformed import k_s, c_s, m_s, k_t, c_t, m_u, roaddata_folder, speed_describers, \
        delta_t, file_qty, file_ini, timesteps_skip, generate_dataset

    if generate_dataset:
        dataset = {'times': [], 'roadvertaccs': [], 'roadvertheights': [], 'carbodyvertaccs': [], 'car_models': [],
                   'speeds': [], 'positions': []}
        for m_s_var in [1.0, 1.00]:
            # Road excitation preparing
            speed_describer = speed_describers.pop()
            t_vector = np.arange(0, speed_describer.T, delta_t)
            loader = loaders.Road(roaddata_folder, file_ini=file_ini, file_qty=file_qty, plot=False, tol=1)
            roadvertacc, roadvertheight, speed, position = \
                get_roadvertaccheight(loader, speed_describer, t_vector, lane=2.0, plot=False,
                                      ret_speed_and_pos=True)

            # Simulations
            m_s_ = m_s_var * m_s
            load_u = Load(dof_s=1, force=m_u * roadvertacc, t=t_vector)
            load_s = Load(dof_s=2, force=m_s_ * roadvertacc, t=t_vector)
            mesh = Mesh(n_dof=3, length=1)
            mesh.fill_elements('k', [k_t, k_s])
            mesh.fill_elements('c', [c_t, c_s])
            const = Constraint(dof_s=0)
            model = Model(mesh=mesh, constraints=const, lumped_masses=[1, m_u, m_s_], loads=[load_u, load_s],
                          options={'t_vector': t_vector, 'method': 'RK23'})
            dataset['car_models'].append(model.deepcopy())
            model.linearize()
            model.lsim()

            _, carbodyvertacc, _, _ = \
                modelsol2meas(model, roadvertacc, roadvertheight, timesteps_skip=timesteps_skip)  # Training dataset

            dataset['roadvertaccs'].append(roadvertacc[0::timesteps_skip])
            dataset['roadvertheights'].append(roadvertheight[0::timesteps_skip])
            dataset['carbodyvertaccs'].append(carbodyvertacc)
            dataset['times'].append(t_vector[0::timesteps_skip])
            dataset['speeds'].append(speed[0::timesteps_skip])
            dataset['positions'].append(position[0::timesteps_skip])

        with open("dataset_speedinformed.pkl", "wb") as file:
            joblib.dump(dataset, file)
    else:
        with open("dataset_speedinformed.pkl", "rb") as file:
            dataset = joblib.load(file)

    fig, ax = plt.subplots(5, 2, sharex='all')
    for i_car in range(2):
        ax[0, i_car].set_title(f'car {i_car + 1}')
        ax[0, i_car].plot(dataset['times'][i_car], dataset['positions'][i_car])
        ax[1, i_car].plot(dataset['times'][i_car], dataset['speeds'][i_car])
        ax[2, i_car].plot(dataset['times'][i_car], dataset['roadvertheights'][i_car])
        ax[3, i_car].plot(dataset['times'][i_car], dataset['roadvertaccs'][i_car])
        ax[4, i_car].plot(dataset['times'][i_car], dataset['carbodyvertaccs'][i_car])
        ax[4, i_car].set_xlabel('time (s)')
    ax[0, 0].set_ylabel('position')
    ax[1, 0].set_ylabel('speed')
    ax[2, 0].set_ylabel('road vertical height')
    ax[3, 0].set_ylabel('road vertical acc.')
    ax[4, 0].set_ylabel('car body vertical acc.')
    plt.show()

    sieql = SpeedInformedEqualizerLearner(x1=dataset['carbodyvertaccs'][0], x2=dataset['carbodyvertaccs'][1],
                                          s1=dataset['speeds'][0], s2=dataset['speeds'][1],
                                          p1=dataset['positions'][0], p2=dataset['positions'][1],
                                          t1=dataset['times'][0], t2=dataset['times'][1],
                                          road1=dataset['roadvertheights'][0],
                                          batch_length_s=1.0, polynomial_kernel_degree=None,
                                          amplitude_regularization=1000.0, learning_rate=0.0001, epochs=50,
                                          train_batch_size=10)
    sieql.build_model()
    sieql.fit_model()
    sieql.predict()
    plot_sieq_fd_td_results(sieql=sieql, fvalid=25)

    a = 0

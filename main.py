import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

from roadprofile_management import loaders
from roadprofile_management.utils import get_roadvertacc
from models.chain_like import Mesh, Constraint, Model, Load

if __name__ == "__main__":
    from input_data import k_s, c_s, m_s, k_t, c_t, m_u, roaddata_folder, speed_descr, delta_t

    t_vector = np.arange(0, speed_descr.T, delta_t)
    loader = loaders.Road(roaddata_folder, file_ini=0, file_qty=100, plot=False, tol=1)
    roadvertacc, roadvertheight = get_roadvertacc(loader, speed_descr, t_vector, plot=False)

    load_u = Load(dof_s=1, force=m_u * roadvertacc, t=t_vector)
    load_s = Load(dof_s=2, force=m_s * roadvertacc, t=t_vector)
    mesh = Mesh(n_dof=3, length=1)
    mesh.fill_elements('k', [k_t, k_s])
    mesh.fill_elements('c', [c_t, c_s])
    const = Constraint(dof_s=0)
    model = Model(mesh=mesh, constraints=const, lumped_masses=[1, m_u, m_s], loads=[load_u, load_s],
                  options={'t_vector': t_vector, 'method': 'RK23'})
    model.linearize()
    model.lsim()
    # model.solve()

    fig, ax = plt.subplots(2, 1, sharex='col')
    ax[0].set_ylabel('acelerations (m/s2)')
    ax[0].plot(t_vector, roadvertacc, label='road')
    ax[0].plot(t_vector, roadvertacc + model.accelerations(1, append=0), label='wheel')
    ax[0].plot(t_vector, roadvertacc + model.accelerations(2, append=0), label='car body')
    ax[0].legend()

    ax[1].set_ylabel('height (m)')
    ax[1].plot(t_vector, roadvertheight, label='road')
    ax[1].plot(t_vector, roadvertheight + model.displacements(1), label='wheel')
    ax[1].plot(t_vector, roadvertheight + model.displacements(2), label='car body')
    ax[1].legend()
    plt.show()
    ...

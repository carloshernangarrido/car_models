def estimate_car(t_vector, roadvertheight, carbodyvertacc, plot=False):
    car = CarDynamicsIdentification(t_vector, roadvertheight, carbodyvertacc)
    car.inverse_identification(plot=plot, fit=True)
    return car


def estimate_road(car, carbodyvertacc, plot=False):
    car.carbodyvert = carbodyvertacc
    car.inverse_identification(plot=plot, fit=False)
    road = np.zeros((car.inverse_y_pred.shape[1] + car.inverse_y_pred.shape[0],))
    y_pred_cols = car.inverse_y_pred.shape[1]
    road_length = car.inverse_y_pred.shape[1] + car.inverse_y_pred.shape[0] - 1
    for i in range(0, road_length - y_pred_cols, y_pred_cols):
        road[i:i + y_pred_cols] = car.inverse_y_pred[i, :, 0]
    return road


fig, ax = plt.subplots(2, 2)
ax[0, 0].plot(roadvertheights[1][0::timesteps_skip], marker='.')
ax[0, 1].plot(roadvertheights[2][0::timesteps_skip], marker='.')
plt.show()
n_iter = 3
car1_iters, car2_iters = [], []
car1_iters.append(init_model)
car2_iters.append(init_model)
road1_iters, road2_iters = [], []
ks = init_model.kernel_size
for i_iter in range(n_iter):
    # estimate road1 with car1
    _, carbodyvertacc, _, _ = \
        modelsol2meas(models[1][1], roadvertaccs[1], roadvertheights[1], timesteps_skip=timesteps_skip)
    road1_car1 = estimate_road(car1_iters[-1], carbodyvertacc, plot=False)
    # estimate road1 with car2
    _, carbodyvertacc, _, _ = \
        modelsol2meas(models[2][1], roadvertaccs[1], roadvertheights[1], timesteps_skip=timesteps_skip)
    road1_car2 = estimate_road(car2_iters[-1], carbodyvertacc, plot=False)
    # estimate road1
    road1 = (road1_car1 + road1_car2) / 2
    road1_iters.append(road1)

    # estimate road2 with car1
    _, carbodyvertacc, _, _ = \
        modelsol2meas(models[1][2], roadvertaccs[2], roadvertheights[2], timesteps_skip=timesteps_skip)
    road2_car1 = estimate_road(car1_iters[-1], carbodyvertacc, plot=False)
    # estimate road2 with car2
    _, carbodyvertacc, _, _ = \
        modelsol2meas(models[2][2], roadvertaccs[2], roadvertheights[2], timesteps_skip=timesteps_skip)
    road2_car2 = estimate_road(car2_iters[-1], carbodyvertacc, plot=False)
    # estimate road2
    road2 = (road2_car1 + road2_car2) / 2
    road2_iters.append(road2)

    # estimate car1 with road1
    _, carbodyvertacc, _, _ = \
        modelsol2meas(models[1][1], roadvertaccs[1], roadvertheights[1], timesteps_skip=timesteps_skip)
    car1_road1 = estimate_car((t_vector[0::timesteps_skip])[ks - 1:], road1_iters[-1], carbodyvertacc[ks - 1:])
    # estimate car1 with road2
    _, carbodyvertacc, _, _ = \
        modelsol2meas(models[1][2], roadvertaccs[2], roadvertheights[2], timesteps_skip=timesteps_skip)
    car1_road2 = estimate_car((t_vector[0::timesteps_skip])[ks - 1:], road2_iters[-1], carbodyvertacc[ks - 1:])
    # estimate car1
    car1 = copy(car1_road1)
    car1.inverse_model.weights[3] = (car1_road1.inverse_model.weights[3] + car1_road2.inverse_model.weights[3]) / 2
    car1_iters.append(car1)

    # estimate car2 with road1
    _, carbodyvertacc, _, _ = \
        modelsol2meas(models[2][1], roadvertaccs[1], roadvertheights[1], timesteps_skip=timesteps_skip)
    car2_road1 = estimate_car((t_vector[0::timesteps_skip])[ks - 1:], road1_iters[-1], carbodyvertacc[ks - 1:])
    # estimate car2 with road2
    _, carbodyvertacc, _, _ = \
        modelsol2meas(models[2][2], roadvertaccs[2], roadvertheights[2], timesteps_skip=timesteps_skip)
    car2_road2 = estimate_car((t_vector[0::timesteps_skip])[ks - 1:], road2_iters[-1], carbodyvertacc[ks - 1:])
    # estimate car2
    car2 = copy(car2_road1)
    car2.inverse_model.weights[3] = (car2_road1.inverse_model.weights[3] + car2_road2.inverse_model.weights[3]) / 2
    car2_iters.append(car2)

    ax[0, 0].plot(road1)
    ax[0, 1].plot(road2)
    ax[1, 0].plot(np.reshape(car1.inverse_model.weights[3], (-1,)))
    ax[1, 1].plot(np.reshape(car2.inverse_model.weights[3], (-1,)))
    plt.show()




    # # Spectrograms Log scale
    # f = np.linspace(1, 49, 49)  # Valores de x de 1 a 49 # Crear una matriz que represente la hiperbola y = 1/x
    # diff_fd = f**2
    # acc_spectrogram_road = eq_l.y_train_fd * diff_fd
    # acc_spectrogram_road_log = np.log(acc_spectrogram_road)
    # vmin, vmax = np.min(acc_spectrogram_road_log), np.max(acc_spectrogram_road_log)
    # fig_road = plt.figure()
    # cax_road = plt.imshow(acc_spectrogram_road_log, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    # cbar_road = fig_road.colorbar(cax_road)
    # plt.title('acc. spectrogram of the actual road')
    #
    # acc_car1_log, acc_car2_log = np.log(eq_l.x1_train_fd), np.log(eq_l.x2_train_fd)
    # vmin, vmax = np.min(np.min((acc_car1_log, acc_car2_log))), np.max(np.max((acc_car1_log, acc_car2_log)))
    # fig_, ax = plt.subplots(1, 2)
    # ax[0].imshow(acc_car1_log, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    # ax[0].set_title('acc. at car 1')
    # ax[1].imshow(acc_car2_log, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    # ax[1].set_title('acc. at car 2')
    #
    # # Plot prediction
    # y_pred_log = np.log(eq_l.predict())
    # fig_, ax = plt.subplots(1, 2)
    # vmin, vmax = np.min(np.min(y_pred_log)), np.max(np.max(y_pred_log))
    # ax[0].imshow(y_pred_log[0], aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    # ax[0].set_title('equalized acc. at car 1')
    # ax[1].imshow(y_pred_log[1], aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    # ax[1].set_title('equalized acc. at car 2')
    # plt.show()

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

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

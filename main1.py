from source import *


def generate_MCC_training_data(mode, timestep, t_end,path):
    motordict = read_motordict(path)
    stator_connection = 'directPhase'

    motor_model = MotorModel(motordict, timestep, stator_connection, solver='newton')
    tuner = RelaxationTuner()
    data_logger = HistoryDataLogger(motor_model)

    steps_total = int(t_end // timestep)  # Total number of steps to simulate

    # data_logger.pre_allocate(steps_total)

    for n in tqdm(range(steps_total)):
        # I. Generate the input

        # I.A Load torque
        # The IMMEC function smooth_runup is called.
        # Here, it runs up to 3.7 Nm between 1.5 seconds and 1.7 seconds
        T_l = smooth_runup(3.7, n * timestep, 1.5, 1.7)

        # I.B Applied voltage
        # 400 V_RMS symmetrical line voltages are used
        v_a = 400 / np.sqrt(3) * np.sqrt(2) * np.sin(2 * np.pi * 50 * n * timestep)
        v_b = 400 / np.sqrt(3) * np.sqrt(2) * np.sin(2 * np.pi * 50 * n * timestep - 2 * np.pi / 3)
        v_c = 400 / np.sqrt(3) * np.sqrt(2) * np.sin(2 * np.pi * 50 * n * timestep - 4 * np.pi / 3)
        v_abc = np.array([v_a, v_b, v_c])
        v_abc = smooth_runup(v_abc, n * timestep, 0.0, 1.5)

        # I.C Rotor eccentricity
        # In this demo, the rotor is placed in a centric position
        ecc = np.zeros(2)

        # I.D The inputs are concatenated to a single vector
        inputs = np.concatenate([v_abc, [T_l], ecc])

        # II. Log the motor model values, time, and inputs

        data_logger.log(n * timestep, inputs)

        # III. Step the motor model
        if mode == 'linear':
            motor_model.step(inputs)
        else:
            # A step is initialised as unsolved
            tuner.solved = False

            while not tuner.solved:
                try:
                    # Apply the relaxation factor of the tuner to the motor model
                    motor_model.relaxation_factor = tuner.relaxation
                    # Attempt to step the motor model
                    motor_model.step(inputs)
                    # When succesful, increase the tuner relaxation factor
                    tuner.step()
                # When unsuccesful, decrease the tuner relaxation factor
                except NoConvergenceException:
                    tuner.jump()

    # get x train data

    data_logger.postprocess()
    x_train = reference_abc_to_dq0(data_logger.quantities['i_st'])

    # get u data: potentials_st, i_st, omega_rot, gamma_rot, and the intergals.
    I =
    V =

    u_data = np.hstack((data_logger.quantities['v_applied'],I,V, data_logger.quantities['gamma_rot'], data_logger.quantities['omega_rot']))

    #data_logger.postprocess()
    #data_logger.plot('all') #don't plot 'everything'


    # use quantities to get desired information
    #data_logger.quantities['v_applied'];data_logger.quantities['i_st']; data_logger.quantities['omega_rot']; data_logger.quantities['gamma_rot']

    #todo, format output probably reshape needed
    #todo, implement sindy c !
    return x_train, u_train


# Generate training data
# From MCC model: find dx/dt

dt = 1e-4
path = 'C:/Users/emmav/PycharmProjects/SINDY_project/Cantoni.pkl'

x_train = generate_MCC_training_data(mode='linear', timestep=dt, t_end = 1e-3, path = path)
# Fit the model

poly_order = 5
threshold = 0.005
optimizer = ps.SR3(thresholder="l1", threshold=threshold)  # l1 optimizer

model = ps.SINDy(
    optimizer=optimizer,
    feature_library=ps.PolynomialLibrary(degree=poly_order),
)

model.fit(x_train, t=dt) #here control array should be added

model.print()
# Simulate and plot the results
'''
x_sim = model.simulate(x0_train, t_train)
plot_kws = dict(linewidth=2)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].plot(t_train, x_train[:, 0], "r", label="$x_0$", **plot_kws)
axs[0].plot(t_train, x_train[:, 1], "b", label="$x_1$", alpha=0.4, **plot_kws)
axs[0].plot(t_train, x_sim[:, 0], "k--", label="model", **plot_kws)
axs[0].plot(t_train, x_sim[:, 1], "k--")
axs[0].legend()
axs[0].set(xlabel="t", ylabel="$x_k$")

axs[1].plot(x_train[:, 0], x_train[:, 1], "r", label="$x_k$", **plot_kws)
axs[1].plot(x_sim[:, 0], x_sim[:, 1], "k--", label="model", **plot_kws)
axs[1].legend()
axs[1].set(xlabel="$x_1$", ylabel="$x_2$")
plt.show()
'''
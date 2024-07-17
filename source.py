from tqdm import tqdm
from immec import *
import matplotlib.pyplot as plt
import numpy as np
import pysindy as ps


def reference_abc_to_dq0(coord_array):
    # coord_array is a Nx3 array with N the number of samples

    # Clarke transformation: power invariant
    T = np.sqrt(2 / 3) * np.array([[1, -0.5, -0.5],
                                   [0, np.sqrt(3) / 2, -np.sqrt(3) / 2],
                                   [1 / np.sqrt(2), 1 / np.sqrt(2), 1 / np.sqrt(2)]])
    return np.dot(T, coord_array.T).T


def reference_dq0_to_abc(coord_array):
    # coord_array is a Nx3 array with N the number of samples

    # Clarke inverse transformation: power invariant
    T = np.sqrt(2 / 3) * np.array([[1, -0.5, -0.5],
                                   [0, np.sqrt(3) / 2, -np.sqrt(3) / 2],
                                   [1 / np.sqrt(2), 1 / np.sqrt(2), 1 / np.sqrt(2)]]).T
    return np.dot(T, coord_array.T).T

def show_data_keys(path_to_data_logger):
    with open(path_to_data_logger, 'rb') as file:
        data_logger = pkl.load(file)
    print(data_logger.keys())
    return

def check_training_data(path_to_data_logger, keys_to_plot_list=None):
    if keys_to_plot_list is None:
        keys_to_plot_list = ['i_st']
    with open(path_to_data_logger, 'rb') as file:
        data_logger = pkl.load(file)
    for key in keys_to_plot_list:
        plt.figure()
        plt.plot(data_logger[key])
    plt.show()
    return


def get_immec_training_data(path_to_data_logger, timestep=1e-4, use_estimate_for_v=False, useOldData=False):
    # useOldData TO BE REMOVED, for backward compatibility
    with open(path_to_data_logger, 'rb') as file:
        data_logger = pkl.load(file)

    # get x train data
    x_train = reference_abc_to_dq0(data_logger['i_st'])

    # todo v_applied is not in correct reference
    if use_estimate_for_v:
        v_stator = reference_abc_to_dq0(v_abc_estimate(data_logger))
    elif useOldData:
        v_stator = data_logger['v_applied']  # TO BE REMOVED ONLY WORKS ON OLD DATA WITH no wye circuit
    else:
        v_stator = reference_abc_to_dq0(v_abc_exact(data_logger))

    # get u data: potentials_st, i_st, omega_rot, gamma_rot, and the intergals.
    I = np.cumsum(x_train,
                  0) * timestep  # those are caluculated using forward euler. Note that x_train*timestep instead of cumsum * timestep yield same results
    V = np.cumsum(v_stator, 0) * timestep
    # CONSIDER USING RUNGE KUTTA METHOD FOR THIS - see ABU-SEIF ET AL.

    u_data = np.hstack((v_stator, I, V, data_logger['gamma_rot'] % (2 * np.pi),
                        data_logger['omega_rot']))

    return x_train, u_data


def create_and_save_immec_data(timestep, t_end, path_to_motor, save_path, V=400, mode='linear'):
    # V should always be below 400, minimal V is 40 (means 5hz f)
    motordict = read_motordict(path_to_motor)
    stator_connection = 'wye'

    motor_model = MotorModel(motordict, timestep, stator_connection, solver='newton')
    tuner = RelaxationTuner()
    data_logger = HistoryDataLogger(motor_model)

    steps_total = int(t_end // timestep)  # Total number of steps to simulate

    Vf_ratio = 400 / 50

    # data_logger.pre_allocate(steps_total)

    for n in tqdm(range(steps_total)):
        # I. Generate the input

        # I.A Load torque
        # The IMMEC function smooth_runup is called.
        # Here, it runs up to 3.7 Nm between 1.5 seconds and 1.7 seconds
        # T_l = smooth_runup(3.7, n * timestep, 1.5, 1.7)
        # Here, no torque is applied
        T_l = 0

        # I.B Applied voltage
        # 400 V_RMS symmetrical line voltages are used
        v_u = V * np.sqrt(2) * np.sin(2 * np.pi * V / Vf_ratio * n * timestep)
        v_v = V * np.sqrt(2) * np.sin(2 * np.pi * V / Vf_ratio * n * timestep - 2 * np.pi / 3)
        v_w = V * np.sqrt(2) * np.sin(2 * np.pi * V / Vf_ratio * n * timestep - 4 * np.pi / 3)
        v_uvw = np.array([v_u, v_v, v_w])

        v_uvw = smooth_runup(v_uvw, n * timestep, 0.0, 1.5)

        # I.C Rotor eccentricity
        # In this demo, the rotor is placed in a centric position
        ecc = np.zeros(2)

        # I.D The inputs are concatenated to a single vector
        inputs = np.concatenate([v_uvw, [T_l], ecc])

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

        data_logger.save_history(save_path)
    return


def v_abc_exact(data_logger):
    data_logger.keys()
    # input the entire data_logger

    v_abc = None

    return v_abc


def v_abc_estimate(data_logger):
    # actually, only data_logger['v_applied'] is needed here
    # outputs Nx3 array
    T = np.array([[1, -1, 0],
                  [1, 2, 0],
                  [-2, -1, 0]])
    return 1/3 * np.dot(T,data_logger['v_applied'].T).T


if __name__ == '__main__':
    def f(t, offset): return np.sin(2 * np.pi * t + offset)


    # test reference_abc_to_dq0 : for a balanced system
    array = np.array([[f(0, 0), f(0, 2 / 3 * np.pi), f(0, 4 / 3 * np.pi)],
                      [f(0.2, 0), f(0.2, 2 / 3 * np.pi), f(0.2, 4 / 3 * np.pi)]])
    print(reference_abc_to_dq0(array))

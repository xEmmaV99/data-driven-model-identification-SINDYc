import sklearn.utils
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


def get_immec_training_data(path_to_data_logger, timestep=1e-4, use_estimate_for_v=False, motorinfo_path=None,
                            useOldData=False):
    # useOldData TO BE REMOVED, for backward compatibility
    with open(path_to_data_logger, 'rb') as file:
        data_logger = pkl.load(file)

    # get x  data
    x_data = reference_abc_to_dq0(data_logger['i_st'])

    if use_estimate_for_v:
        v_stator = reference_abc_to_dq0(v_abc_estimate(data_logger))
    elif useOldData:
        v_stator = data_logger['v_applied']  # TO BE REMOVED ONLY WORKS ON OLD DATA WITH no wye circuit
    else:
        v_stator = reference_abc_to_dq0(v_abc_exact(data_logger, path_to_motor_info=motorinfo_path))

    # get u data: potentials_st, i_st, omega_rot, gamma_rot, and the intergals.
    I = np.cumsum(x_data,
                  0) * timestep  # those are caluculated using forward euler. Note that x_train*timestep instead of cumsum * timestep yield same results
    V = np.cumsum(v_stator, 0) * timestep
    # CONSIDER USING RUNGE KUTTA METHOD FOR THIS - see ABU-SEIF ET AL.

    u_data = np.hstack((v_stator, I, V, data_logger['gamma_rot'] % (2 * np.pi),
                        data_logger['omega_rot']))

    # lastly, shuffle the data using sklearn.utils.shuffle to shuffle consistently
    t = data_logger['time']
    #x_data, u_data, t = sklearn.utils.shuffle(x_data,u_data,t)

    shuffle = False
    if shuffle:
        train_idx = sklearn.utils.shuffle(np.arange(x_data.shape[0])) # 80% training
        train_idx = np.sort(train_idx[:int(0.8*x_data.shape[0])])
        x_train = x_data[train_idx,:]
        u_train = u_data[train_idx,:]
        x_valid = x_data[[True if k in train_idx  else False for k in range(len(x_data))],:]
        u_valid = u_data[[True if k in train_idx  else False for k in range(len(x_data))],:]

        t_train = t[train_idx,:]
        t_valid = t[[True if k in train_idx  else False for k in range(len(x_data))],:]

    else:
        cutoff = int(.8* x_data.shape[0]);
        x_train = x_data[:cutoff, :]
        u_train = u_data[:cutoff, :]
        x_valid = x_data[cutoff:, :]
        u_valid = u_data[cutoff:, :]

        t_train = t[:cutoff, :]
        t_valid = t[cutoff:, :]
    '''plt.figure()
    plt.plot(t_train, x_train)
    plt.plot(t_valid, x_valid, "k")
    plt.show()'''
    return x_train, u_train, t_train, x_valid, u_valid, t_valid

def save_motor_data(motor_path, save_path):
    motordict = read_motordict(motor_path)
    motor_model = MotorModel(motordict, 1e-4, "wye", solver='newton')

    dictionary = {}
    dictionary['stator_leakage_inductance'] = motor_model.stator_leakage_inductance
    dictionary['N_abc_T'] = motor_model.N_abc_T
    dictionary['R_st'] = motor_model.R_st
    with open(save_path + '/MOTORDATA.pkl', 'wb') as file:
        pkl.dump(dictionary, file)
    return

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

        data_logger.save_history(save_path)  # debug here for data_logger.model.R_st
    return


def v_abc_exact(data_logger, path_to_motor_info):
    print(data_logger.keys())

    with open(path_to_motor_info, 'rb') as file:
        motorinfo = pkl.load(file)

    # input the entire data_logger
    R = motorinfo['R_st']
    Nt = motorinfo['N_abc_T']  # model.N_abc
    L_s = motorinfo['stator_leakage_inductance']  # model.stator_leakage_inductance

    dt = data_logger['time'][-1] - data_logger['time'][-2]
    dphi = 1 / dt * np.diff(data_logger['flux_st_yoke'].T)  # forward euler, flux_st_yoke, this array is one shorter
    di = 1 / dt * np.diff(data_logger['i_st'].T)  # forward euler, flux_st_yoke, this array is one shorter

    dphi = np.append(dphi, dphi[:,-1:], axis = 1)
    di = np.append(di, di[:,-1:], axis = 1)
    print("We assume dphi is the same for the last value of v_abc (using approx)")
    v_abc = np.dot(R, data_logger['i_st'].T) + np.dot(Nt, dphi) + L_s * di

    return v_abc.T


def v_abc_estimate(data_logger):
    # actually, only data_logger['v_applied'] is needed here
    # outputs Nx3 array
    T = np.array([[1, -1, 0],
                  [1, 2, 0],
                  [-2, -1, 0]])
    return 1 / 3 * np.dot(T, data_logger['v_applied'].T).T


if __name__ == '__main__':
    def f(t, offset): return np.sin(2 * np.pi * t + offset)


    # test reference_abc_to_dq0 : for a balanced system
    array = np.array([[f(0, 0), f(0, 2 / 3 * np.pi), f(0, 4 / 3 * np.pi)],
                      [f(0.2, 0), f(0.2, 2 / 3 * np.pi), f(0.2, 4 / 3 * np.pi)]])
    print(reference_abc_to_dq0(array))

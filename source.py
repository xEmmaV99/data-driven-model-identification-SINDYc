import copy
import multiprocessing

import sklearn.utils
from tqdm import tqdm
from immec import *
import matplotlib.pyplot as plt
import numpy as np
import pysindy as ps
import seaborn as sns
import os
from libs import *  # import custom library functions
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import optuna



def reference_abc_to_dq0(coord_array: np.array):
    """
    Changes reference system from abc to dq0. Note that if dimension is (N, 3, k)
     then for each k, the (N, 3) will be transformed
    :param coord_array: array with shape (N, 3) with N the number of samples
    :return: transformed array with shape (N, 3)
    """
    # Clarke transformation: power invariant
    T = np.sqrt(2 / 3) * np.array(
        [[1, -0.5, -0.5], [0, np.sqrt(3) / 2, -np.sqrt(3) / 2], [1 / np.sqrt(2), 1 / np.sqrt(2), 1 / np.sqrt(2)]]
    )
    return np.tensordot(T, coord_array.swapaxes(0, 1), axes=([1], [0])).swapaxes(0, 1)


def reference_dq0_to_abc(coord_array: np.array):
    """
    Changes reference system from dq0 to abc
    :param coord_array: array with shape (N, 3) with N the number of samples
    :return: transformed array with shape (N, 3)
    """
    # Clarke inverse transformation: power invariant
    T = (
            np.sqrt(2 / 3)
            * np.array(
        [[1, -0.5, -0.5], [0, np.sqrt(3) / 2, -np.sqrt(3) / 2], [1 / np.sqrt(2), 1 / np.sqrt(2), 1 / np.sqrt(2)]]
    ).T
    )
    return np.dot(T, coord_array.swapaxes(0, 1), axes=([1], [0])).swapaxes(0, 1)


def show_data_keys(path_to_data_logger: str):
    """
    Shows the keys of the data logger from a .pkl file
    :param path_to_data_logger: str with the path to the .pkl file
    :return:
    """
    with open(path_to_data_logger, "rb") as file:
        data_logger = pkl.load(file)
    print(data_logger.keys())
    return


def check_training_data(path_to_data_logger: str, keys_to_plot_list: list = None):
    """
    Plots the data from the data logger, optional list of keys to plot
    :param path_to_data_logger: str with the path to the .pkl file
    :param keys_to_plot_list: list of str with the keys to plot
    :return:
    """
    if keys_to_plot_list is None:
        keys_to_plot_list = ["i_st"]
    with open(path_to_data_logger, "rb") as file:
        data_logger = pkl.load(file)
    for key in keys_to_plot_list:
        plt.figure()
        plt.plot(data_logger[key])
    plt.show()
    return


""" TO BE REMOVED
def get_immec_training_data(path_to_data_logger, timestep=1e-4, use_estimate_for_v=False, motorinfo_path=None,
                            useOldData=False):
    # useOldData TO BE REMOVED, for backward compatibility
    # works for ONE datafile
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

    u_data = np.hstack((v_stator, I, V, data_logger['gamma_rot'] % (2 * np.pi),
                        data_logger['omega_rot']))

    # lastly, shuffle the data using sklearn.utils.shuffle to shuffle consistently
    t = data_logger['time']
    # x_data, u_data, t = sklearn.utils.shuffle(x_data,u_data,t)

    shuffle = True  # debug
    if shuffle:
        cutoff = int(.8 * x_data.shape[0])
        train_idx = sklearn.utils.shuffle(np.arange(x_data.shape[0]))  # 80% training
        train_idx = np.sort(train_idx[:cutoff])  # leave them unsorted
        x_train = x_data[train_idx, :]
        u_train = u_data[train_idx, :]
        x_valid = x_data[[False if k in train_idx else True for k in range(len(x_data))], :]
        u_valid = u_data[[False if k in train_idx else True for k in range(len(x_data))], :]

        t_train = t[train_idx, :]
        t_valid = t[[False if k in train_idx else True for k in range(len(x_data))], :]

    else:
        cutoff = int(.5 * x_data.shape[0])
        x_train = x_data[:cutoff, :]
        u_train = u_data[:cutoff, :]
        x_valid = x_data[cutoff:, :]
        u_valid = u_data[cutoff:, :]

        t_train = t[:cutoff, :]
        t_valid = t[cutoff:, :]
    plot = True  # debug
    if plot:
        plt.figure()
        plt.scatter(t_train, x_train[:, 0], marker=".")
        plt.scatter(t_valid, x_valid[:, 0], marker=".")
        plt.legend(["traindata", "validationdata"])
        plt.show()
    return x_train, u_train, t_train, x_valid, u_valid, t_valid"""


def save_simulation_data(motor_path: str, save_path: str, extra_dict: dict = None):
    """
    Saves useful simulation data in a dictionary to a .pkl file
    :param motor_path: path to motor data file
    :param save_path: path to save the file
    :param extra_dict: dict with extra things to save
    :return:
    """
    motordict = read_motordict(motor_path)
    motor_model = MotorModel(motordict, 5e-5, "wye", solver="newton")

    dictionary = {
        "stator_leakage_inductance": motor_model.stator_leakage_inductance,
        "N_abc_T": motor_model.N_abc_T,
        "R_st": motor_model.R_st,
    }

    for key in extra_dict.keys():  # add extra things
        dictionary[key] = extra_dict[key]

    with open(save_path + "/SIMULATION_DATA.pkl", "wb") as file:
        pkl.dump(dictionary, file)
    return


def create_and_save_immec_data(
        timestep: float,
        t_end: float,
        path_to_motor: str,
        save_path: str,
        V: float = 400,
        mode: str = "linear",
        solving_tolerance: float = 1e-4,
):
    """
    Creates and saves data for the IMMEC project, TO BE REMOVED
    :param timestep:
    :param t_end:
    :param path_to_motor:
    :param save_path:
    :param V:
    :param mode:
    :param solving_tolerance:
    :return:
    """
    # V should always be below 400, minimal V is 40 (means 5hz f)
    motordict = read_motordict(path_to_motor)
    stator_connection = "wye"

    if mode == "linear":
        motor_model = MotorModel(motordict, timestep, stator_connection)
    else:
        motor_model = MotorModel(
            motordict, timestep, stator_connection, solver="newton", solving_tolerance=solving_tolerance
        )

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
        T_l = smooth_runup(3.7, n * timestep, 1.5, 1.7)
        # Here, no torque is applied
        # T_l = 0

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
        if mode == "linear":
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


def create_immec_data(
        timestep: float,
        t_end: float,
        path_to_motor: str,
        V: float = 400,
        mode: str = "linear",
        solving_tolerance: float = 1e-4,
        load: float = 3.7,
        ecc: np.array = np.zeros(2),
):
    """
    Creates data from the IMMEC model
    :param timestep: timestep of the simulation
    :param t_end: end of the simulation
    :param path_to_motor: path to the motor data file
    :param V: Maximum voltage applied to the motor
    :param mode: 'linear' for no nonlinear effects when simulating
    :param solving_tolerance:  tolerance used for the approximation of non linearities, only used when mode is not "linear"
    :param load: Load applied to the motor in Nm, at time 1.5s to 1.7s
    :param ecc: eccentricity of the motor, in percentage of the airgap
    :return: data_logger object (HistoryDataLogger) containing the data
    """
    # V should always be below 400, minimal V is 40 (means 5hz f)
    motordict = read_motordict(path_to_motor)
    stator_connection = "wye"

    if mode == "linear":
        motor_model = MotorModel(motordict, timestep, stator_connection)
    else:
        motor_model = MotorModel(
            motordict, timestep, stator_connection, solver="newton", solving_tolerance=solving_tolerance
        )

    tuner = RelaxationTuner()
    data_logger = HistoryDataLogger(motor_model)

    steps_total = int(t_end // timestep)  # Total number of steps to simulate

    Vf_ratio = 400 / 50

    # data_logger.pre_allocate(steps_total)

    # initial params
    start_load = 0.0
    end_load = load
    start_time = 0.0
    close_to_steady_state = False
    dt_load = 0.2  # first applied load for 1 second
    Vfmode = "chirp"
    print('Mode: ', Vfmode)
    for n in tqdm(range(steps_total)):
        # I. Generate the input

        # I.A Load torque todo: change this!

        # The IMMEC function smooth_runup is called.
        # Here, it runs up to 3.7 Nm (load) between 1.5 seconds and 1.7 seconds
        # T_l = smooth_runup(load, n * timestep, 1.5, 1.7)  # n*timestep is current time
        # Here, no torque is applied
        # T_l = 0
        if close_to_steady_state:
            # start_load = end_load #continuous
            start_load = change_load(start_load, end_load, n * timestep, start_time, start_time + dt_load)
            end_load = int(np.random.randint(0, 370) * (V / 400.0)) / 100  # choose new load
            print("New applied load: ", end_load, "Nm")
            start_time = n * timestep  # apply now
            close_to_steady_state = False  # change back
            dt_load = .2  # apply faster

        T_l = change_load(start_load, end_load, n * timestep, start_time, start_time + dt_load)
        # print("I think the problem is the .2 seconds, the motor cannot deal with the load, applied to fast")

        # I.B Applied voltage
        # 400 V_RMS symmetrical line voltages are used
        if Vfmode == "constant_freq":
            V_amp = V
            f_amp = V / Vf_ratio * n * timestep
        elif Vfmode == "chirp_linear":
            V_amp = linear_runup(V, n * timestep, 1.5)
            f_amp = linear_runup_freq(V / Vf_ratio, n * timestep, 1.5)
        elif Vfmode == "chirp":
            V_amp = smooth_runup(V, n * timestep, 0.0, 1.5)
            f_amp = chirp_freq(V / Vf_ratio, n * timestep, 1.5)

        v_u = V_amp * np.sqrt(2) * np.sin(2 * np.pi * f_amp)
        v_v = V_amp * np.sqrt(2) * np.sin(2 * np.pi * f_amp - 2 * np.pi / 3)
        v_w = V_amp * np.sqrt(2) * np.sin(2 * np.pi * f_amp - 4 * np.pi / 3)
        v_uvw = np.array([v_u, v_v, v_w])

        if Vfmode == "constant_freq":
            v_uvw = smooth_runup(v_uvw, n * timestep, 0.0, 1.5)  # change amplitude of voltage

        # I.C Rotor eccentricity
        ecc = ecc * motordict["d_air"]

        # I.D The inputs are concatenated to a single vector
        inputs = np.concatenate([v_uvw, [T_l], ecc])

        # II. Log the motor model values, time, and inputs
        data_logger.log(n * timestep, inputs)

        # III. Step the motor model
        if mode == "linear":
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

        # check if the steady state is reached, every .1 seconds
        if n % int(0.1 / timestep) == 0:
            close_to_steady_state = check_steady_state(
                T_em=data_logger.quantities["T_em"],
                speed=data_logger.quantities["omega_rot"],
                nmbr_of_steps=int(0.15 / timestep),
            )

    return data_logger


"""
x = np.arange(0,2,0.0001)
y=np.array([0,0,0])
Vf_ratio = 400/5 # should be 400/50 but this is more visible
for x_val in x:
    V_amp = smooth_runup(400, x_val, 0.0, 1.5)
    v_u = V_amp * np.sqrt(2) * np.sin(2 * np.pi * V_amp / Vf_ratio * x_val)
    v_v = V_amp * np.sqrt(2) * np.sin(2 * np.pi * V_amp / Vf_ratio * x_val - 2 * np.pi / 3)
    v_w = V_amp * np.sqrt(2) * np.sin(2 * np.pi * V_amp / Vf_ratio * x_val - 4 * np.pi / 3)
    v_uvw = np.array([v_u, v_v, v_w])
    y = np.vstack((y,v_uvw))

plt.plot(y)
plt.show()
"""


def linear_runup_freq(values, time: float, end_time: float, start_time: float = 0.0):
    f_0 = 0
    c = (values - f_0) / (end_time - start_time)
    if time < start_time:
        return values * time
    elif start_time <= time < end_time:
        return 0.5 * c * (time) ** 2 + f_0 * time
    else:
        phi_add = 0.5 * c * (end_time) ** 2 + f_0 * end_time
        return values * (time - end_time) + phi_add


def linear_runup(values, time: float, end_time: float, start_time: float = 0.0):
    if time < start_time:
        return np.zeros_like(values)
    elif start_time <= time < end_time:
        return values * (time - start_time) / (end_time - start_time)
    else:
        return values


def chirp_freq(values, time: float, end_time: float, start_time: float = 0.0):
    duration = end_time - start_time
    if time < start_time:
        return values * time
    elif start_time <= time < end_time:
        return 1 / 2 * (time - np.sin(np.pi * time / duration) * duration / np.pi) * values

    else:
        phi_add = 1 / 2 * (end_time - np.sin(np.pi * end_time / duration) * duration / np.pi) * values
        return values * (time - end_time) + phi_add


def check_steady_state(T_em, speed, nmbr_of_steps):
    # steady state  is when T_em and speed is constant
    T_em = T_em[-nmbr_of_steps:]
    speed = speed[-nmbr_of_steps:]

    meanT = np.mean(T_em)
    meanS = np.mean(speed)

    # all points should be within 5% of the mean
    if np.all(np.abs(T_em - meanT) < 0.05 * meanT) and np.all(np.abs(speed - meanS) < 0.05 * meanS):
        return True
    return False


def change_load(start_load, end_load, time: float, start_time: float, end_time: float):
    """
    Function used to change the load continuously with 1 - cos(t) type of run-up and run-down.
    :param start_load: The value of the initial load
    :param end_load: The value of the applied load
    :param time: The continuous (current) time value
    :param start_time: The time at which run-up starts
    :param end_time: The time at which run-up is completed
    :return: The values subjected to run-up
    """
    # Return start_load before the starting time
    if time < start_time:
        return start_load

    if start_load < end_load:
        # Use 1 - cos(t) run-up between the start time and the end time
        if time < end_time:
            duration = end_time - start_time
            return start_load + (end_load - start_load) * 0.5 * (1 - np.cos(np.pi / duration * (time - start_time)))
    elif start_load > end_load:
        # Use cos(t) run-down between the start time and the end time
        if time < end_time:
            duration = end_time - start_time
            return end_load + (start_load - end_load) * 0.5 * (1 + np.cos(np.pi / duration * (time - start_time)))
    # Return the value(s) after the end time
    return end_load


"""
x  = np.arange(0,10,0.01)
y = []
for x_val in x:
    y.append(change_load(0,10,x_val,0,10))
x2 = np.arange(10,20,0.01)
for x_val in x2:
    y.append(change_load(10,5,x_val,13,15))

plt.plot(y)
plt.show()
"""


def v_abc_exact(data_logger: dict, path_to_motor_info: str):
    """
    This function calculates the exact abc voltages from the line voltages
    :param data_logger: HistoryDataLogger object or dict, containing the data
    :param path_to_motor_info: str, path to the motor data file
    :return: np.array with shape (N, 3), abc voltages
    """
    with open(path_to_motor_info, "rb") as file:
        motorinfo = pkl.load(file)

    # input the entire data_logger
    R = motorinfo["R_st"]
    Nt = motorinfo["N_abc_T"]  # model.N_abc
    L_s = motorinfo["stator_leakage_inductance"]  # model.stator_leakage_inductance

    if np.ndim(data_logger['time']) > 2:
        print("Also 4'th order now")  # DEBUG
        t = data_logger['time'][:, 0, 0]
        dphi = ps.FiniteDifference(order=4, axis=0)._differentiate(data_logger["flux_st_yoke"], t)
        di = ps.FiniteDifference(order=4, axis=0)._differentiate(data_logger["i_st"], t)
    else:
        t = data_logger['time'][:, 0]
        dphi = ps.FiniteDifference(order=4, axis=0)._differentiate(data_logger["flux_st_yoke"], t)
        di = ps.FiniteDifference(order=4, axis=0)._differentiate(data_logger["i_st"], t)

    '''
    dphi = (
            1 / dt * np.diff(data_logger["flux_st_yoke"].swapaxes(0, 1), axis=1)
    )  # forward euler, flux_st_yoke, this array is one shorter
    di = (
            1 / dt * np.diff(data_logger["i_st"].swapaxes(0, 1), axis=1)
    )  # forward euler, flux_st_yoke, !!! this array is one shorter !!
    '''

    ist = data_logger["i_st"]
    v_abc = np.tensordot(R, ist.swapaxes(0, 1), axes=([1], [0])) + np.tensordot(Nt, dphi.swapaxes(0, 1),
                                                                                axes=([1], [0])) + L_s * di.swapaxes(0,
                                                                                                                     1)

    return v_abc.swapaxes(0, 1)


def v_abc_estimate(data_logger: dict):
    """
    This function estimates the abc voltages from the line voltages, using a transformation
    :param data_logger: HistoryDataLogger object containing the data
    :return: np.array with shape (N, 3), abc voltages
    """
    # actually, only data_logger['v_applied'] is needed here
    # outputs Nx3 array
    T = np.array([[1, -1, 0], [1, 2, 0], [-2, -1, 0]])
    return 1 / 3 * np.dot(T, data_logger["v_applied"].T).T


def read_V_from_directory(path_to_directory: str):
    """
    This function reads the voltage from the directory name, when multiple simulations are in seperate files. TO BE REMOVED
    :param path_to_directory:
    :return:
    """
    # most datafiles have the name IMMEC_history_{voltage}V_1.0sec.pkl, so this function reads the voltage from the
    # directory name, return an array with all voltages from files in this directory
    files = os.listdir(path_to_directory)
    V = []
    for file in files:
        if file.endswith(".pkl") and file.startswith("IMMEC_history"):
            V.append(int(file.split("_")[2][:-1]))
    return np.array(V)


def read_V_load_from_simulationdata(path_to_simulation_data: str):
    """
    This function reads the voltage and load from the simulation data
    :param path_to_simulation_data: str, path to the simulation data file
    :return: np.array, np.array of the voltage and load
    """
    # read the load torque from the simulation data
    with open(path_to_simulation_data, "rb") as file:
        data = pkl.load(file)
    return data["V"], data["load"]


def calculate_xdot(x: np.array, t: np.array):
    """
    Calculate the time derivative of the state x at time t, by using pySINDy
    :param x: array of shape (N , 3)
    :param t: array of shape (N , 1) or (N, )
    :return: array of shape (N , 3) with the time derivative of x, pySINDy does not remove one value.
    """
    # print('debug, 4th order of xdot')
    if np.ndim(t) == 3:
        print("Assume all t_vec are equal")
        t_ = t[:, 0, 0].reshape(t.shape[0])
        return ps.FiniteDifference(order=4, axis=0)._differentiate(x, t_)
        # t should have shape (N, )
    return ps.FiniteDifference(order=4, axis=0)._differentiate(x, t.reshape(t.shape[0]))  # Default order is two.


def save_plot_data(
        save_name: str, xydata: list, title: str, xlab, ylab, legend=None, plot_now=False, specs=None, sindy_model=None
):
    # xydata contains the data to plot, but if multiple axis should be plotted, xy data should be a list of arrays
    # if it is only one x,y then [np.array([x,y])] should be the input
    # create the dictionary to save as is
    pltdata = {
        "title": title,
        "xlab": xlab,
        "ylab": ylab,
        "legend": legend,
        "plots": {},
        "specs": specs,
        "model": sindy_model,
    }
    for i, xy_array in enumerate(xydata):
        pltdata["plots"][str(i)] = xy_array
    cwd = os.getcwd()
    save_path = os.path.join(cwd, "plot_data\\", save_name + ".pkl")

    with open(save_path, "wb") as file:
        pkl.dump(pltdata, file)
    if plot_now:
        plot_data(save_path)
    return save_path


def plot_data(path="plotdata.pkl", show=True, figure=True, limits=None):
    suppres_title = False
    if type(path) == str:
        paths = [path]
    else:
        paths = path
        suppres_title = True

    linetypes = ["-", "--", ":"]
    for j, path in enumerate(paths):
        with open(path, "rb") as file:
            data = pkl.load(file)

        if type(data["ylab"]) != str:  # multiple axis
            print("Multiple axis plot detected.")
            print("loglog ax1 and semilogx ax2.")
            # if subplot exist, dont' create a new subplot
            figure = plt.fignum_exists(1)  # check if figure exists
            if not figure:
                fig, ax1 = plt.subplots()

            ax1.set_xlabel(data["xlab"])

            ax1.set_ylabel(data["ylab"][0], color="r")
            ax1.loglog(data["plots"]["0"][:, 0], data["plots"]["0"][:, 1:], "r" + linetypes[j])

            ax2 = ax1.twinx()
            ax2.set_ylabel(data["ylab"][1], color="b")
            ax2.semilogx(data["plots"]["1"][:, 0], data["plots"]["1"][:, 1:], "b" + linetypes[j])

            if not suppres_title:
                plt.title(data["title"])
            if limits == None:
                fig.tight_layout()
            else:
                ax1.set_ylim(limits[0])
                ax2.set_ylim(limits[1])

        else:  # only one axis
            plt.figure()
            plt.xlabel(data["xlab"]), plt.ylabel(data["ylab"])
            specs = data["specs"]
            for idx in data["plots"]:
                if specs[int(idx)] is not None:
                    plt.plot(data["plots"][idx][:, 0], data["plots"][idx][:, 1:], specs[int(idx)])
                else:
                    plt.plot(data["plots"][idx][:, 0], data["plots"][idx][:, 1:])

            plt.legend(data["legend"])
            plt.title(data["title"])
    if show:
        plt.show()
    return


def plot_coefs(model):
    coefs = model.coefficients()
    featurenames = model.feature_names()
    # plot coefs of the model, based on the code provided by pysindy:
    # https://pysindy.readthedocs.io/en/latest/examples/7_plasma_examples/example.html
    input_features = [rf"$\dot x_{k}$" for k in range(coefs.shape[0])]
    if featurenames == None:
        input_names = [rf"$x_{k}$" for k in range(coefs.shape[1])]
    else:
        input_names = featurenames

    with sns.axes_style(style="white", rc={"axes.facecolor": (0, 0, 0, 0)}):
        fig, ax = plt.subplots(1, 1)
        max_magnitude = np.max(np.abs(coefs))
        heatmap_args = {
            "xticklabels": input_features,
            "yticklabels": input_names,
            "center": 0.0,
            # "cmap": "RdBu_r",
            "cmap": sns.color_palette("vlag", n_colors=20),
            "ax": ax,
            "linewidths": 0.1,
            "linecolor": "whitesmoke",
        }

        sns.heatmap(coefs[:, : len(input_names)].T, **heatmap_args)

        ax.tick_params(axis="y", rotation=0)
    return


def plot_coefs2(model, normalize_values=False, show=False, log=False):
    xticknames = model.get_feature_names()
    for i in range(len(xticknames)):
        xticknames[i] = xticknames[i]
    plt.figure(figsize=(len(xticknames), 4))
    colors = ["b", "r", "k"]
    coefs = copy.deepcopy(model.coefficients())

    if normalize_values:
        raise NotImplementedError("This function is not implemented yet.")  # todo

    if log:
        plt.yscale("log", base=10)
        coefs = np.abs(coefs)

    for i in range(2):
        values = coefs[i, :].T
        values[values == 0] = np.nan  # don't plot zero
        plt.scatter(
            np.arange(0, len(xticknames), 1),
            values,
            color=colors[i],
            label=r"Equation for $\dot{" + xticknames[i + 1].strip("$") + "}$",
        )

    plt.grid(True)
    plt.xticks(range(len(xticknames)), xticknames, rotation=90)

    plt.legend()
    if show:
        plt.show()
    return


def save_model(model, name, libstr):
    print("Saving model")
    path = "C:/Users/emmav/PycharmProjects/SINDY_project/models/" + name + ".pkl"
    x = model.n_features_in_ - model.n_control_features_
    u = model.n_control_features_

    lib = {
        "coefs": model.coefficients(),
        "features": model.feature_names,
        "library": libstr,  # todo; custom lirary from libs
        "shapes": [(1, x), (1, u), (1, x)],
    }
    with open(path, "wb") as file:
        pkl.dump(lib, file)


def load_model(name):
    path = "C:/Users/emmav/PycharmProjects/SINDY_project/models/" + name + ".pkl"
    with open(path, "rb") as file:
        model_data = pkl.load(file)

    # initialize pysindy model

    lib = get_custom_library_funcs(model_data["library"])

    new_model = ps.SINDy(optimizer=None, feature_names=model_data["features"], feature_library=lib)

    x_shape, u_shape, xdot_shape = model_data["shapes"]
    new_model.fit(np.zeros(x_shape), u=np.zeros(u_shape), t=None, x_dot=np.zeros(xdot_shape))

    new_model.optimizer.coef_ = model_data["coefs"]

    return new_model


def parameter_search(parameter_array, train_and_validation_data, method="lasso", name="", plot_now=True, library=None):
    if name == "":
        name = method
    method = method.lower()
    variable = {
        "parameter": parameter_array,
        "MSE": np.zeros(len(parameter_array)),
        "SPAR": np.zeros(len(parameter_array)),
        "model": list(np.zeros(len(parameter_array))),
    }

    xdot_train, x_train, u_train, xdot_val, x_val, u_val = train_and_validation_data

    for i, para in tqdm(enumerate(parameter_array)):
        if method[:3] == "sr3":
            optimizer = ps.SR3(thresholder=method[-2:], nu=para, threshold=1e-12)
        elif method == "lasso":
            optimizer = Lasso(alpha=para, fit_intercept=False)
        elif method == "stlsq":
            optimizer = ps.STLSQ(threshold=0.1, alpha=para)
        elif method == "srr":
            optimizer = ps.SSR(alpha=para)
        else:
            raise NameError("Method is invalid")

        if library is None:
            library = ps.PolynomialLibrary(degree=2, include_interaction=True)

        model = ps.SINDy(optimizer=optimizer, feature_library=library)
        print("Fitting model")
        model.fit(x_train, u=u_train, t=None, x_dot=xdot_train)  # fit on training data
        if model.coefficients().ndim == 1:  # fix dimensions of this matrix, bug in pysindy, o this works
            model.optimizer.coef_ = model.coefficients().reshape(1, model.coefficients().shape[0])
        variable["MSE"][i] = model.score(x_val, u=u_val, x_dot=xdot_val, metric=mean_squared_error)
        # the same as: mean_squared_error(model.predict(x_val, u_val), xdot_val)
        variable["SPAR"][i] = np.count_nonzero(model.coefficients())  # number of non-zero elements
        variable["model"][i] = model

    # rel_sparsity = variable['SPAR'] / np.max(variable['SPAR'])

    # plot and save the results
    xlab = r"Sparsity weighting factor $\alpha$"
    ylab1 = "MSE"
    ylab2 = "Number of non-zero elements"
    title = "MSE and sparsity VS Sparsity weighting parameter, " + method + " method"
    xydata = [
        np.hstack(
            (variable["parameter"].reshape(len(parameter_array), 1), variable["MSE"].reshape(len(parameter_array), 1))
        ),
        np.hstack(
            (variable["parameter"].reshape(len(parameter_array), 1), variable["SPAR"].reshape(len(parameter_array), 1))
        ),
    ]
    specs = ["r", "b"]
    save_plot_data(name, xydata, title, xlab, [ylab1, ylab2], specs, plot_now=plot_now)

    idx = np.where(np.min(variable["MSE"]))  # best model, lowest MSE
    best_model = variable["model"][idx[0][0]]
    print(
        "Best model found with MSE: ",
        variable["MSE"][idx[0][0]],
        " and parameter: ",
        variable["parameter"][idx[0][0]],
        "for ",
        method,
    )
    return best_model


def parameter_search_2D(param_nu, param_lambda, train_and_validation_data, name="", plot_now=True):
    variable = {
        "param_nu": param_nu,
        "param_lambda": param_lambda,
        "MSE": np.zeros((len(param_nu), len(param_lambda))),
        "SPAR": np.zeros((len(param_nu), len(param_lambda))),
    }

    xdot_train, x_train, u_train, xdot_val, x_val, u_val = train_and_validation_data

    for i, nu in enumerate(param_nu):

        for j, lam in enumerate(param_lambda):
            print(i, " and ", j)
            optimizer = ps.SR3(thresholder='l1', nu=nu,
                               threshold=np.sqrt(2 * lam * nu))  # see https://arxiv.org/pdf/1906.10612
            model = ps.SINDy(optimizer=optimizer,
                             feature_library=ps.PolynomialLibrary(degree=2, include_interaction=True))
            model.fit(x_train, u=u_train, t=None, x_dot=xdot_train)

            variable["MSE"][i, j] = model.score(x_val, u=u_val, x_dot=xdot_val, t=None, metric=mean_squared_error)
            variable["SPAR"][i, j] = np.count_nonzero(model.coefficients())

    # plot the grid as a heatplot with MSE the color
    plt.figure()
    plt.imshow(variable["MSE"], cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.xlabel("Lambda")
    plt.ylabel("Nu")
    plt.title("MSE heatmap")
    plt.figure()
    plt.imshow(variable["SPAR"], cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.xlabel("Lambda")
    plt.ylabel("Nu")
    plt.title("Sparsity heatmap")
    plt.show()

    return


def grid_search_sr3(DATA, l_minmax, n_minmax, iter=4):
    #from https://optuna.readthedocs.io/en/stable/index.html
    def objective(trial):
        lambdas = trial.suggest_float('lambdas', l_minmax[0], l_minmax[1], log=True)
        nus = trial.suggest_float('nus', n_minmax[0], n_minmax[1], log=True)
        lib_choice = trial.suggest_categorical('lib_choice', ['poly_2nd_order', 'sincos_cross', 'system', 'higher_order', 'pure_poly_2dn_order']) # 'best' eats all the memory

        lib = get_custom_library_funcs(lib_choice)

        optimizer = ps.SR3(thresholder='l1', nu=nus,
                           threshold=lambdas)
        model = ps.SINDy(optimizer=optimizer,
                         feature_library=lib)
        model.fit(DATA["x_train"], u=DATA["u_train"], t=None, x_dot=DATA["xdot_train"])

        MSE = model.score(DATA["x_val"], u=DATA["u_val"], x_dot=DATA["xdot_val"],
                          t=None, metric=mean_squared_error)
        SPAR = np.count_nonzero(model.coefficients())

        return MSE, SPAR

    study_name = "example-study"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(directions=['minimize', 'minimize'],
                                study_name=study_name,
                                storage=storage_name,
                                load_if_exists=True)

    study.optimize(objective, n_trials=iter, n_jobs=1)

    #optuna.visualization.plot_pareto_front(study, target_names= ["MSE","SPAR"]).show(renderer="browser")

    return


def plot_everything(path_to_directory):
    files = os.listdir(path_to_directory)
    for file in files:
        if file.endswith(".pkl"):
            path = os.path.join(path_to_directory, file)
            plot_data(path, show=False)
    plt.show()
    return

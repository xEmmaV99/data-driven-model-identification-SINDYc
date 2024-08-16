import random
import scipy
import numpy as np
import os
import pickle as pkl
import numba as nb

try:
    from pysindy import FiniteDifference
except ImportError:
    print("Skipping import of pysindy")


# todo consider multithreading here, as one CPU might be the bottleneck


def prepare_data(path_to_data_file: str,
                 test_data: bool =False,
                 number_of_trainfiles: int =-1,
                 use_estimate_for_v: bool=False,
                 usage_per_trainfile: float =0.5,
                 ecc_input: bool = False):
    """
    Prepares the data for pysindy. If not test_data, the output is split and shuffeled into train and validation
    :param path_to_data_file: path to the data files
    :param test_data: if True, the path_to_data_file should point to a test file.
    :param number_of_trainfiles: number of simulations to be considered, to use 'all', pass -1
    :param use_estimate_for_v: if True, use the approximation from line voltages to find v_abc, if False, use the more general form
    :param usage_per_trainfile: float between 0 and 1, trim the data in the time-domain to reduce samples
    :param ecc_input: if True, eccentricity is added as input for the control variables
    :return:
    """
    # check if usage_per_trainfile is between 0 and 1
    if not 0 <= usage_per_trainfile <= 1:
        raise ValueError("usage_per_trainfile should be between 0 and 1")

    # load numpy file
    print("Loading data")
    dataset = dict(np.load(path_to_data_file))  # should be a dictionary
    print("Done loading data")

    if 'wcoe' not in dataset.keys(): #debug
        dataset['wcoe'] = np.zeros_like(dataset['T_em'])
    if dataset['wcoe'].shape != dataset['T_em'].shape:
        dataset['wcoe'] = np.expand_dims(dataset['wcoe'], axis=1) #make sure wcoe is same shape as T-em

    if not test_data:
        V_range = read_V_from_data(path_to_data_file)
    else:
        V_range = np.array([np.max(dataset['v_applied']) / np.sqrt(2)])

    # choose random simulations from the data
    if not test_data and number_of_trainfiles > dataset['i_st'].shape[2]:
        raise ValueError("number_of_trainfiles is larger than the amount of simulations in the dataset")
    if number_of_trainfiles == -1 or number_of_trainfiles == 'all':
        number_of_trainfiles = len(V_range)
    random_idx = random.sample(range(len(V_range)), number_of_trainfiles)  # the simulations are shuffled
    # choose random V from V_range
    V_range = V_range[random_idx]

    # initialise data
    DATA = {'x': np.array([]), 'u': np.array([]), 'xdot': np.array([]),
            'T_em': np.array([]), 'UMP': np.array([]), 'feature_names': np.array([]), 'wcoe':np.array([])}

    # crop dataset to desired amount of simulations (random_idx)
    if not test_data:
        for key in dataset.keys():
            dataset[key] = dataset[key][:, :, random_idx]

    # prepare v data
    if use_estimate_for_v:
        v_stator = reference_abc_to_dq0(v_abc_estimate_from_line(dataset["v_applied"]))  # debug
    else:
        path_to_simulation_data = os.path.join(os.path.dirname(path_to_data_file), 'SIMULATION_DATA.pkl')
        v_stator = reference_abc_to_dq0(v_abc_calculation(dataset, path_to_motor_info=path_to_simulation_data))
        #todo, debug, SIMULATION_DATA is only needed for this

    i_st = reference_abc_to_dq0(dataset['i_st'])

    if np.ndim(i_st) <= 2:  # expand such that the code works for both 2D and 3D data
        i_st = np.expand_dims(i_st, axis=2)
        v_stator = np.expand_dims(v_stator, axis=2)
        dataset["time"] = np.expand_dims(dataset['time'], axis=2)
        dataset['T_em'] = np.expand_dims(dataset['T_em'], axis=2)
        dataset['F_em'] = np.expand_dims(dataset['F_em'], axis=2)
        dataset['wcoe'] = np.expand_dims(dataset['wcoe'], axis=2)

    # get u data: potentials_st, i_st, omega_rot, gamma_rot, and the integrals.
    for simul in range(number_of_trainfiles):
        if simul == 0:  # initiliaze
            I = scipy.integrate.cumulative_trapezoid(i_st[:, :, simul],  dataset['time'][:, 0, simul], axis=0, initial=0)
            V = scipy.integrate.cumulative_trapezoid(v_stator[:, :, simul],  dataset['time'][:, 0, simul], axis=0, initial=0)
            continue
        I = np.dstack(
            (I, scipy.integrate.cumulative_trapezoid(i_st[:, :, simul],  dataset['time'][:, 0, simul], axis=0, initial=0)))
        V = np.dstack(
            (V, scipy.integrate.cumulative_trapezoid(v_stator[:, :, simul],  dataset['time'][:, 0, simul], axis=0, initial=0)))

    # get x data
    x_data = reference_abc_to_dq0(dataset['i_st'])
    ###  x_data = np.hstack((i_st, I.reshape(i_st.shape), V.reshape(i_st.shape))) #for data merged
    t_data = dataset['time']
    if np.ndim(x_data) <= 2:
        x_data = np.expand_dims(x_data, axis=2)

    # calculate xdots
    print("Calculating xdots")
    xdots = calculate_xdot(x_data, t_data)
    print("Done calculating xdots")

    if not test_data:  # trim times AFTER xdots calculation
        print('time trim: ', usage_per_trainfile)
        timepoints = len(dataset['time'][:, 0, 0])
        time_trim = random.sample(range(timepoints), int(usage_per_trainfile * timepoints))
        for key in dataset.keys():
            # error for wcoe
            if key == 'wcoe' and dataset[key].shape[0] == 1:
                dataset[key] = np.swapaxes(dataset[key], 0, 1) #debug TO BE REMOVED...

            dataset[key] = dataset[key][time_trim]

        v_stator = v_stator[time_trim]
        x_data = x_data[time_trim]
        t_data = t_data[time_trim]
        xdots = xdots[time_trim]
        I = I[time_trim]
        V = V[time_trim]

    # u_data add supply frequency to the input data
    freqs = V_range * 50 / 400  # constant proportion DEBUG todo check if this is correct

    freqs = freqs.reshape(1, 1, len(freqs))  # along third axis
    u_data = np.hstack((v_stator,
                        I.reshape(v_stator.shape),
                        V.reshape(v_stator.shape),
                        dataset['gamma_rot'].reshape(t_data.shape) % (2 * np.pi),
                        dataset['omega_rot'].reshape(t_data.shape),
                        np.repeat(freqs, dataset['omega_rot'].shape[0], axis=0)))

    DATA['feature_names'] = [r'i_d', r'i_q', r'i_0',
                             r'v_d', r'v_q', r'v_0',
                             r'I_d', r'I_q', r'I_0',
                             r'V_d', r'V_q', r'V_0',
                             r'\gamma', r'\omega', r'f']

    # add eccentricity to the input data
    if ecc_input:
        if not np.all(dataset['ecc'] < 1e-10):
            if not test_data:
                print('Non zero ecc, added to input data')
                u_data = np.hstack((u_data, dataset['ecc']))
                #DEBUGGGGGGGG

                #r = np.sqrt(dataset['ecc'][:, 0] ** 2 + dataset['ecc'][:, 1] ** 2)
                #theta = np.arctan2(dataset['ecc'][:, 1], dataset['ecc'][:, 0])
                #u_data = np.hstack((u_data, r[:,np.newaxis,:], np.cos(theta[:,np.newaxis,:])))
                #u_data = np.hstack((u_data, np.sin(theta[:, np.newaxis, :]), np.cos(theta[:, np.newaxis, :])))
            else:
                print('Non zero ecc, added to input data')
                u_data = np.hstack((u_data, dataset['ecc'][:, :, np.newaxis]))

            #DATA['feature_names'].append([r'r_x', r'r_y'])
            [DATA['feature_names'].append(j) for j in [r'r_x', r'r_y']]
        else:
            print('No ecc')

    # Now, stack data on top of each other and shuffle!
    # (Note that the transpose is needed otherwise the reshape is wrong)
    DATA['x'] = x_data.transpose(0, 2, 1).reshape(x_data.shape[0] * x_data.shape[-1], x_data.shape[1])
    DATA['u'] = u_data.transpose(0, 2, 1).reshape(u_data.shape[0] * u_data.shape[-1], u_data.shape[1])
    DATA['xdot'] = xdots.transpose(0, 2, 1).reshape(xdots.shape[0] * xdots.shape[-1], xdots.shape[1])
    DATA['T_em'] = dataset["T_em"].transpose(0, 2, 1).reshape(dataset["T_em"].shape[0] * dataset["T_em"].shape[-1])
    DATA['UMP'] = dataset["F_em"].transpose(0, 2, 1).reshape(dataset["F_em"].shape[0] * dataset["F_em"].shape[-1],
                                                             dataset["F_em"].shape[1])
    DATA['wcoe'] = dataset['wcoe'].transpose(0, 2, 1).reshape(dataset['wcoe'].shape[0] * dataset['wcoe'].shape[-1])

    if test_data:
        DATA['V'] = V_range
        DATA['t'] = t_data
        return DATA

    # shuffle the DATA entirely, but according to the same shuffle
    shuffled_indices = np.random.permutation(DATA['x'].shape[0])
    DATA['x'] = DATA['x'][shuffled_indices]
    DATA['u'] = DATA['u'][shuffled_indices]
    DATA['xdot'] = DATA['xdot'][shuffled_indices]
    DATA['T_em'] = DATA['T_em'][shuffled_indices]
    DATA['UMP'] = DATA['UMP'][shuffled_indices]
    DATA['wcoe'] = DATA['wcoe'][shuffled_indices]

    # split the data into train and validation data
    p = 0.8  # percentage of data to be used for training, 0.2 is used for parameter tuning
    cutidx = int(p * DATA['x'].shape[0])

    DATA['x_train'] = DATA['x'][:cutidx]
    DATA['u_train'] = DATA['u'][:cutidx]
    DATA['xdot_train'] = DATA['xdot'][:cutidx]
    DATA['T_em_train'] = DATA['T_em'][:cutidx]
    DATA['UMP_train'] = DATA['UMP'][:cutidx]
    DATA['wcoe_train'] = DATA['wcoe'][:cutidx]
    DATA['x_val'] = DATA['x'][cutidx:]
    DATA['u_val'] = DATA['u'][cutidx:]
    DATA['xdot_val'] = DATA['xdot'][cutidx:]
    DATA['T_em_val'] = DATA['T_em'][cutidx:]
    DATA['UMP_val'] = DATA['UMP'][cutidx:]
    DATA['wcoe_val'] = DATA['wcoe'][cutidx:]

    return DATA


def check_trapezoid_integration():
    print("this doesn't work")
    path = os.path.join(os.getcwd(), 'test-data', '07-29-default', 'IMMEC_0ecc_5.0sec.npz')
    dataset = prepare_data(path, test_data=True)
    I_end = dataset['u'][-1, 3:6]
    V_end = dataset['u'][-1, 6:9]
    i = dataset['x']
    v = dataset['u'][:, 0:3]
    print(np.sum(i, axis=0) * 5e-5, I_end)
    print(np.sum(v, axis=0) * 5e-5, V_end)
    return


def reference_abc_to_dq0(coord_array: np.array):
    """
    Changes reference system from abc to dq0. Note that if dimension is (N, 3, k)
     then for each k, the (N, 3) will be transformed
    :param coord_array: array with shape (N, 3) with N the number of samples
    :return: transformed array with shape (N, 3)
    """
    # Clarke transformation: power invariant
    T = np.sqrt(2 / 3) * np.array(
        [[1, -0.5, -0.5],
         [0, np.sqrt(3) / 2, -np.sqrt(3) / 2],
         [1 / np.sqrt(2), 1 / np.sqrt(2), 1 / np.sqrt(2)]]
    )
    return np.tensordot(T, coord_array.swapaxes(0, 1), axes=([1], [0])).swapaxes(0, 1)


@nb.njit(cache=True)
def reference_abc_to_dq0_CP(coord_array: np.array, gamma):
    print("Not finished, doesn't work")
    nd = 3
    if np.ndim(gamma) <= 2:
        nd = 2
        gamma = gamma.reshape((gamma.shape[0], 1, 1))
        coord_array = coord_array.reshape((coord_array.shape[0], coord_array.shape[1], 1))

    newcoord = np.ones_like(coord_array)
    for simul in range(coord_array.shape[-1]):
        for j in range(len(gamma[:, 0, simul])):
            g = gamma[j, 0, simul]
            # Clarck- Park transformation as a function of gamma
            CP = np.sqrt(2 / 3) * np.array(
                [[np.cos(g), np.cos(g - 2 / 3 * np.pi), np.cos(g + 2 / 3 * np.pi)],
                 [-np.sin(g), -np.sin(g - 2 / 3 * np.pi / 3), -np.sin(g + 2 / 3 * np.pi)],
                 [1 / np.sqrt(2), 1 / np.sqrt(2), 1 / np.sqrt(2)]]
            )
            newcoord[j, :, simul] = np.dot(CP, coord_array[j, :, simul].T).T

    return newcoord[:, :, 0] if nd == 2 else newcoord


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


def v_abc_calculation(data_logger: dict, path_to_motor_info: str):
    """
    This function calculates the exact abc voltages
    :param data_logger: HistoryDataLogger object or dict, containing the data
    :param path_to_motor_info: str, path to the motor data file
    :return: np.array with shape (N, 3), abc voltages
    """
    with open(path_to_motor_info, "rb") as file:
        motorinfo = pkl.load(file)

    R = motorinfo["R_st"]
    Nt = motorinfo["N_abc_T"]  # model.N_abc transposed
    L_s = motorinfo["stator_leakage_inductance"]  # model.stator_leakage_inductance

    if np.ndim(data_logger['time']) > 2:
        t = data_logger['time'][:, 0, 0]
    else:
        t = data_logger['time'][:, 0]

    dphi = FiniteDifference(order=4, axis=0)._differentiate(data_logger["flux_st_yoke"], t)
    di = FiniteDifference(order=4, axis=0)._differentiate(data_logger["i_st"], t)

    ist = data_logger["i_st"]
    v_abc = np.tensordot(R, ist.swapaxes(0, 1), axes=([1], [0])) + \
            np.tensordot(Nt, dphi.swapaxes(0, 1), axes=([1], [0])) + \
            L_s * di.swapaxes(0, 1)

    return v_abc.swapaxes(0, 1)


def v_abc_estimate_from_line(v_line: np.array):
    """
    This function estimates the abc voltages from the line voltages, using a transformation
    :param v_line: the line voltages
    :return: np.array with shape (N, 3), abc voltages
    """
    # outputs Nx3 array
    T = np.array([[1, -1, 0], [1, 2, 0], [-2, -1, 0]])
    return 1 / 3 * np.tensordot(T, v_line.swapaxes(0, 1), axes=([1], [0])).swapaxes(0, 1)

''' # TO BE REMOVED. CAN CAUSE ERRORS
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
'''

def read_V_from_data(path_to_data_file: str):
    """
    This function reads the voltage from the data file
    :param path_to_data_file: str, path to the data file
    :return: np.array, np.array of the voltage
    """
    with open(path_to_data_file, "rb") as file:
        data = np.load(file)
        V_values = np.max(data['v_applied'], axis = (0,1)) / np.sqrt(2)
    return V_values


def calculate_xdot(x: np.array, t: np.array):
    """
    Calculate the time derivative of the state x at time t, by using pySINDy
    :param x: array of shape (N , 3)
    :param t: array of shape (N , 1) or (N, )
    :return: array of shape (N , 3) with the time derivative of x, pySINDy does not remove one value.
    """
    if np.ndim(t) == 3:
        print("Assume all t_vec are equal")
        t_ = t[:, 0, 0].reshape(t.shape[0])
        return FiniteDifference(order=4, axis=0)._differentiate(x, t_)
        # t should have shape (N, )
    return FiniteDifference(order=4, axis=0)._differentiate(x, t.reshape(t.shape[0]))  # Default order is two.


if __name__ == "__main__":
    #check_trapezoid_integration()

    path = os.path.join(os.getcwd(), 'train-data', '07-29-default', 'IMMEC_0ecc_5.0sec.npz')

    path = os.path.join(os.getcwd(), 'test-data', '07-31-leuven', 'IMMEC_90ecc_5.0sec.npz')
    data = prepare_data(path,
                        test_data=True,
                        number_of_trainfiles=-1,
                        use_estimate_for_v=False, ecc_input=True)
    print(data['u'].shape)
import random
import os

import scipy

from source import *


def prepare_data(path_to_data_files, V_test_data=None, Torque=False, path_to_test_file=None):
    path_to_simulation_data = os.path.join(path_to_data_files, 'SIMULATION_data.pkl')

    # Read Voltages used for simulation
    with open(path_to_simulation_data, 'rb') as f:
        simulation_data = pkl.load(f)
    V_range = simulation_data['V']

    if V_test_data is None:  # is used for some plots, should be removed later
        V_test_data = random.choice(V_range)
    if path_to_test_file is not None:  # used if dedicated test file is present
        V_test_data = None

    t_end = 1.0  # it is assumed to be 1

    use_estimate_for_v = False  # if true, the v_abc_estimate function is used to estimate the stator voltage, else v_abc_exact is used

    # load data
    # if it is one big file
    # todo: later
    # all different files

    DATA = {'x': np.array([]), 'u': np.array([]), 'xdot': np.array([])}
    TESTDATA = {'x': np.array([]), 'u': np.array([]), 'xdot': np.array([]), 't': np.array([]), 'V': V_test_data,
                'T_em': np.array([])}

    # for each datafile, fix the data (eg calculate xdot) and add to the DATA dictionary
    for V_value in V_range:
        path = path_to_data_files + '/IMMEC_history_' + str(V_value) + 'V_' + str(t_end) + 'sec'
        with open(path + '.pkl', 'rb') as f:
            dataset = pkl.load(f)

        # get x  data
        x_data = reference_abc_to_dq0(dataset['i_st'])
        t_data = dataset['time']
        # first, calculate xdots and add to the data
        xdots = calculate_xdot(x_data, t_data)  # SINDY DOESN'T SHORTEN THE VECTOR
        xdots = xdots[:-1, :]  # crop last datapoint

        # prepare v data
        if use_estimate_for_v:
            v_stator = reference_abc_to_dq0(v_abc_estimate(dataset))
            v_stator = v_stator[:-1]  # crop last datapoint (consistent with the else)
        else:
            v_stator = reference_abc_to_dq0(v_abc_exact(dataset, path_to_motor_info=path_to_simulation_data))
            #v_stator is one shorter, so must the other data!

        # remove last timestep from datafile
        for key_to_crop in ['time', 'i_st', 'omega_rot', 'gamma_rot', 'T_em']:  # note that v_stator is already cropped
            dataset[key_to_crop] = dataset[key_to_crop][:-1]

        x_data = x_data[:-1, :]  # update x_data
        t_data = dataset['time']  # update t_data

        # get u data: potentials_st, i_st, omega_rot, gamma_rot, and the integrals.
        timestep = t_data[1] - t_data[0]
        #I = np.cumsum(x_data, 0) * timestep
        #V = np.cumsum(v_stator, 0) * timestep  # CONSIDER USING todo trapezoid method scipy.integrate.cumulative_trapezoid() makes it one shorter too
        I = scipy.integrate.cumtrapz(x_data, t_data, axis=0, initial=0)
        V = scipy.integrate.cumtrapz(v_stator, t_data, axis=0, initial=0)


        u_data = np.hstack((v_stator, I, V, dataset['gamma_rot'] % (2 * np.pi),
                            dataset['omega_rot'])) #correct size
        # u data: v, I, V, gamma, omega

        # at this point, data should me merged to one big dataset, whereafter it can be shuffled.
        if V_value == V_test_data:  # if dedicated test file is present, this is never used
            TESTDATA['x'] = x_data
            TESTDATA['u'] = u_data
            TESTDATA['xdot'] = xdots
            TESTDATA['t'] = t_data
            TESTDATA['T_em'] = dataset["T_em"]
        else:
            if DATA['x'].shape[0] == 0:
                DATA['x'] = x_data
                DATA['u'] = u_data
                DATA['xdot'] = xdots
                DATA['T_em'] = dataset["T_em"]
            else:
                DATA['x'] = np.vstack(
                    (DATA['x'], x_data))  # note that x_data is in dq0 reference frame while dataset['i_st'] is in abc
                DATA['u'] = np.vstack((DATA['u'], u_data))
                DATA['xdot'] = np.vstack((DATA['xdot'], xdots))
                DATA['T_em'] = np.vstack((DATA['T_em'], dataset["T_em"]))

    # note the slicing is done because 1) forward euler in v

    # note that the test data will not be shuffled
    # shuffle the DATA entirely, but according to the same shuffle
    shuffled_indices = np.random.permutation(DATA['x'].shape[0])
    DATA['x'] = DATA['x'][shuffled_indices, :]
    DATA['u'] = DATA['u'][shuffled_indices, :]
    DATA['xdot'] = DATA['xdot'][shuffled_indices, :]
    DATA['T_em'] = DATA['T_em'][shuffled_indices, :]

    # split the data into train and validation data
    p = 0.8  # percentage of data to be used for training
    cutidx = int(p * DATA['x'].shape[0])

    x_train = DATA['x'][:cutidx, :]
    u_train = DATA['u'][:cutidx, :]
    xdot_train = DATA['xdot'][:cutidx, :]
    T_em_train = DATA['T_em'][:cutidx, :]
    x_val = DATA['x'][cutidx:, :]
    u_val = DATA['u'][cutidx:, :]
    xdot_val = DATA['xdot'][cutidx:, :]
    T_em_val = DATA['T_em'][cutidx:, :]

    visualise_train_data = False
    if visualise_train_data:
        # todo think about it
        raise NotImplementedError('Visualisation of training data is not yet implemented')

    if path_to_test_file is not None:  # prepare the TESTDATA if a test file is present todo clean up
        with open(path_to_test_file, 'rb') as f:
            testset = pkl.load(f)
        x_data = reference_abc_to_dq0(testset['i_st'])
        t_data = testset['time']

        xdots = calculate_xdot(x_data, t_data)
        xdots = xdots[:-1, :]  # crop last datapoint
        # prepare v data
        if use_estimate_for_v:
            v_stator = reference_abc_to_dq0(v_abc_estimate(testset))
            v_stator = v_stator[:-1]  # crop last datapoint (consistent with the else)
        else:
            v_stator = reference_abc_to_dq0(v_abc_exact(testset, path_to_motor_info=path_to_simulation_data))
        for key_to_crop in ['time', 'i_st', 'omega_rot', 'gamma_rot', 'T_em']:  # note that v_stator is already cropped
            testset[key_to_crop] = testset[key_to_crop][:-1]
        x_data = x_data[:-1, :]  # update x_data
        t_data = testset['time']  # update t_data
        timestep = t_data[1] - t_data[0]
        I = np.cumsum(x_data, 0) * timestep
        V = np.cumsum(v_stator, 0) * timestep  # condisder todo trapezoid method
        u_data = np.hstack((v_stator, I, V, testset['gamma_rot'] % (2 * np.pi),
                            testset['omega_rot']))
        TESTDATA['x'] = x_data
        TESTDATA['u'] = u_data
        TESTDATA['xdot'] = xdots
        TESTDATA['t'] = t_data
        TESTDATA['T_em'] = testset["T_em"]

    if Torque:
        return T_em_train, x_train, u_train, T_em_val, x_val, u_val, TESTDATA
    return xdot_train, x_train, u_train, xdot_val, x_val, u_val, TESTDATA

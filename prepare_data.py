import random

from source import *


def prepare_data():
    path_to_data_files = 'C:/Users/emmav/PycharmProjects/SINDY_project/data/data_files'
    path_to_motor_data = 'C:/Users/emmav/PycharmProjects/SINDY_project/data/MOTORDATA.pkl'

    # specific data values
    V_range = np.linspace(40, 400, 10)
    V_test_data = random.choice(V_range)
    t_end = 1.0
    use_estimate_for_v = False  # if true, the v_abc_estimate function is used to estimate the stator voltage, else v_abc_exact is used

    # load data
    # if it is one big file
    # todo: later
    # all different files

    DATA = {'x': np.array([]), 'u': np.array([]), 'xdot': np.array([])}
    TESTDATA = {'x': np.array([]), 'u': np.array([]), 'xdot': np.array([]), 't': np.array([]), 'V': V_test_data}

    # for each datafile, fix the data (eg calculate xdot) and add to the DATA dictionary
    for V_value in V_range:
        path = path_to_data_files + '/IMMEC_history_' + str(V_value) + 'V_' + str(t_end) + 'sec'
        with open(path + '.pkl', 'rb') as f:
            dataset = pkl.load(f)

        # get x  data
        x_data = reference_abc_to_dq0(dataset['i_st'])
        t_data = dataset['time']
        # first, calculate xdots and add to the data
        xdots = calculate_xdot(x_data, t_data)

        # prepare v data
        if use_estimate_for_v:
            v_stator = reference_abc_to_dq0(v_abc_estimate(dataset))
            v_stator = v_stator[:-1]  # crop last datapoint (consistent with the else)
        else:
            v_stator = reference_abc_to_dq0(v_abc_exact(dataset, path_to_motor_info=path_to_motor_data))

        # remove last timestep from datafile
        for key_to_crop in ['time', 'i_st', 'omega_rot', 'gamma_rot']:  # note that v_stator is already cropped
            dataset[key_to_crop] = dataset[key_to_crop][:-1]

        x_data = x_data[:-1, :]  # update x_data
        t_data = dataset['time']  # update t_data

        # get u data: potentials_st, i_st, omega_rot, gamma_rot, and the integrals.
        timestep = t_data[1] - t_data[0]
        I = np.cumsum(x_data, 0) * timestep
        V = np.cumsum(v_stator, 0) * timestep  # CONSIDER USING RUNGE KUTTA METHOD FOR THIS - see ABU-SEIF ET AL.

        u_data = np.hstack((v_stator, I, V, dataset['gamma_rot'] % (2 * np.pi),
                            dataset['omega_rot']))
        # at this point, data should me merged to one big dataset, whereafter it can be shuffled.
        if V_value == V_test_data:
            TESTDATA['x'] = x_data
            TESTDATA['u'] = u_data
            TESTDATA['xdot'] = xdots
            TESTDATA['t'] = t_data
        else:
            if DATA['x'].shape[0] == 0:
                DATA['x'] = x_data
                DATA['u'] = u_data
                DATA['xdot'] = xdots
            else:
                DATA['x'] = np.vstack(
                    (DATA['x'], x_data))  # note that x_data is in dq0 reference frame while dataset['i_st'] is in abc
                DATA['u'] = np.vstack((DATA['u'], u_data))
                DATA['xdot'] = np.vstack((DATA['xdot'], xdots))

    # note that the test data will not be shuffled
    # shuffle the DATA entirely, but according to the same shuffle
    shuffled_indices = np.random.permutation(DATA['x'].shape[0])
    DATA['x'] = DATA['x'][shuffled_indices, :]
    DATA['u'] = DATA['u'][shuffled_indices, :]
    DATA['xdot'] = DATA['xdot'][shuffled_indices, :]
    # split the data into train and validation data
    p = 0.8  # percentage of data to be used for training
    cutidx = int(p * DATA['x'].shape[0])
    x_train = DATA['x'][:cutidx, :]
    u_train = DATA['u'][:cutidx, :]
    xdot_train = DATA['xdot'][:cutidx, :]
    x_val = DATA['x'][cutidx:, :]
    u_val = DATA['u'][cutidx:, :]
    xdot_val = DATA['xdot'][cutidx:, :]

    visualise_train_data = False
    if visualise_train_data:
        #todo think about it
        raise NotImplementedError('Visualisation of training data is not yet implemented')
    return xdot_train, x_train, u_train, xdot_val, x_val, u_val, TESTDATA

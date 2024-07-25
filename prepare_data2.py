import random
import scipy

from source import *


def prepare_data(path_to_data_file, V_test_data=None, Torque=False, UMP=False, path_to_test_file=None, t_end=1.0,
                 number_of_trainfiles=10, normalize_input=False, use_estimate_for_v=False):
    path_to_simulation_data = os.path.join( os.path.dirname(path_to_data_file),'SIMULATION_DATA.pkl')  # get out one
    V_range, load_range = read_V_load_from_simulationdata(path_to_simulation_data)

    if path_to_test_file is not None:  # if no testdata is demanded
        V_test_data = None

    # choose random V from V_range
    random_idx = random.sample(range(len(V_range)), number_of_trainfiles) # THIS IS SHUFFLED ALREADY, the simulations are shuffled

    V_range = V_range[random_idx]
    load_range = load_range[random_idx]

    # load data
    DATA = {'x': np.array([]), 'u': np.array([]), 'xdot': np.array([])}
    TESTDATA = {'x': np.array([]), 'u': np.array([]), 'xdot': np.array([]), 't': np.array([]), 'V': V_test_data,
                'T_em': np.array([]), 'UMP': np.array([])}

    # for each datafile, fix the data (eg calculate xdot) and add to the DATA dictionary
    #for idx in random_idx:

    # the files are dim x dim x number_of_simulations or dim x number_of_simulations

    # load numpy file
    dataset = np.load(path_to_data_file) # should be a dictionary

    # get x data
    x_data = reference_abc_to_dq0(dataset['i_st'][:,:,random_idx])
    t_data = dataset['time'][:,random_idx]

    # first, calculate xdots and add to the data
    xdots = calculate_xdot(x_data, t_data)  # SINDY DOESN'T SHORTEN THE VECTOR
    xdots = xdots[:-1]  # crop last datapoint

    # prepare v data
    if use_estimate_for_v:
        v_stator = reference_abc_to_dq0(v_abc_estimate(dataset))
        v_stator = v_stator[:-1]  # crop last datapoint (consistent with the else)
    else:
        v_stator = reference_abc_to_dq0(v_abc_exact(dataset, path_to_motor_info=path_to_simulation_data))
        # v_stator is one shorter, so must the other data!

    # remove last timestep from datafile
    for key_to_crop in ['time', 'omega_rot', 'gamma_rot', 'T_em']:  # note that v_stator is already cropped
        dataset[key_to_crop] = dataset[key_to_crop][:-1]

    for key_to_crop in ['i_st', 'F_em']:
        dataset[key_to_crop] = dataset[key_to_crop][:-1]

    x_data = x_data[:-1]  # update x_data
    t_data = dataset['time']  # update t_data

    # get u data: potentials_st, i_st, omega_rot, gamma_rot, and the integrals.

    I = scipy.integrate.cumtrapz(x_data, t_data, axis=0, initial=0)
    V = scipy.integrate.cumtrapz(v_stator, t_data, axis=0, initial=0)

    # u_data add supply frequency to the input data
    freqs = V_range[random_idx] * 50 / 400  # constant proportion
    '''
    u_data = np.hstack((v_stator, I, V, dataset['gamma_rot'] % (2 * np.pi),
                        dataset['omega_rot'],
                        np.repeat(freq, len(dataset['omega_rot'])).reshape(len(dataset['omega_rot']), 1)))
    '''
    u_data = np.stack((v_stator, I, V, dataset['gamma_rot'] % (2 * np.pi),
                        dataset['omega_rot'],
                        np.repeat(freqs, len(dataset['omega_rot'])).reshape(len(dataset['omega_rot']), 1)), axis = 1) #stack along the second axis

    u_names = [r'$v_d$', r'$v_q$', r'$v_0$', r'$I_d$', r'$I_q$', r'$I_0$', r'$V_d$', r'$V_q$', r'$V_0$',
               r'$\gamma$', r'$\omega$', r'$f$']

    # Now, stack data on top of each other and shuffle!

    DATA['x'] = x_data.reshape(x_data.shape[0]*x_data.shape[-1],x_data.shape[1])
    DATA['u'] = u_data.reshape(u_data.shape[0]*u_data.shape[-1],u_data.shape[1])
    DATA['xdot'] = xdots.reshape(xdots.shape[0]*xdots.shape[-1],xdots.shape[1])
    DATA['T_em'] = dataset["T_em"].reshape(dataset["T_em"].shape[0]*dataset["T_em"].shape[-1])
    DATA['UMP'] = dataset["F_em"].reshape(dataset["F_em"].shape[0]*dataset["F_em"].shape[-1],dataset["F_em"].shape[1])

    # shuffle the DATA entirely, but according to the same shuffle
    shuffled_indices = np.random.permutation(DATA['x'].shape[0])
    DATA['x'] = DATA['x'][shuffled_indices, :]
    DATA['u'] = DATA['u'][shuffled_indices, :]
    DATA['xdot'] = DATA['xdot'][shuffled_indices, :]
    DATA['T_em'] = DATA['T_em'][shuffled_indices, :]
    DATA['UMP'] = DATA['UMP'][shuffled_indices, :]

    # split the data into train and validation data
    p = 0.8  # percentage of data to be used for training
    cutidx = int(p * DATA['x'].shape[0])

    x_train = DATA['x'][:cutidx, :]
    u_train = DATA['u'][:cutidx, :]
    xdot_train = DATA['xdot'][:cutidx, :]
    T_em_train = DATA['T_em'][:cutidx, :]
    UMP_train = DATA['UMP'][:cutidx, :]
    x_val = DATA['x'][cutidx:, :]
    u_val = DATA['u'][cutidx:, :]
    xdot_val = DATA['xdot'][cutidx:, :]
    T_em_val = DATA['T_em'][cutidx:, :]
    UMP_val = DATA['UMP'][cutidx:, :]

    visualise_train_data = False
    if visualise_train_data:
        print(V_range)
        # todo think about it
        raise NotImplementedError('Visualisation of training data is not yet implemented')

    if path_to_test_file is not None:  # prepare the TESTDATA if a test file is present todo clean up
       pass

    TESTDATA['u_names'] = u_names  # Save the names of u_data for SINDy
    if Torque:
        return T_em_train, x_train, u_train, T_em_val, x_val, u_val, TESTDATA
    elif UMP:
        return UMP_train, x_train, u_train, UMP_val, x_val, u_val, TESTDATA
    return xdot_train, x_train, u_train, xdot_val, x_val, u_val, TESTDATA


if __name__ == "__main__":
    path = os.path.join(os.getcwd(), 'train-data', '07-25', 'IMMEC_0ecc_1.0sec.npz')
    data = prepare_data(path, number_of_trainfiles=2)
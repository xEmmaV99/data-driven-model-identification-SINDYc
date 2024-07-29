import random

import matplotlib.pyplot as plt
import scipy

from source import *


def prepare_data(path_to_data_file, V_test_data=None, Torque=False, UMP=False, path_to_test_file=None, t_end=1.0,
                 number_of_trainfiles=10, use_estimate_for_v=False):
    path_to_simulation_data = os.path.join( os.path.dirname(path_to_data_file),'SIMULATION_DATA.pkl')  # get out one
    V_range, load_range = read_V_load_from_simulationdata(path_to_simulation_data)
    #todo consider multithreading here, as one CPU might be the bottleneck

    if path_to_test_file is not None:  # if no testdata is demanded
        V_test_data = None

    print(number_of_trainfiles) #debug
    # choose random V from V_range
    random_idx = random.sample(range(len(V_range)), number_of_trainfiles) # the simulations are shuffled

    V_range = V_range[random_idx]
    load_range = load_range[random_idx]

    # load data
    DATA = {'x': np.array([]), 'u': np.array([]), 'xdot': np.array([])}
    TESTDATA = {'x': np.array([]), 'u': np.array([]), 'xdot': np.array([]), 't': np.array([]), 'V': V_test_data,
                'T_em': np.array([]), 'UMP': np.array([])}

    # load numpy file
    dataset = dict(np.load(path_to_data_file)) # should be a dictionary

    # crop dataset to desired amount of simulations (random_idx)
    for key in dataset.keys():
        dataset[key] = dataset[key][:, :, random_idx]

    # get x data
    x_data = reference_abc_to_dq0(dataset['i_st'])
    t_data = dataset['time']

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
    for simul in range(len(random_idx)):
        if simul == 0: #initiliaze
            I = scipy.integrate.cumulative_trapezoid(x_data[:,:,simul], t_data[:,0,simul], axis=0, initial=0)
            V = scipy.integrate.cumulative_trapezoid(v_stator[:,:,simul], t_data[:,0,simul], axis=0, initial=0)
            continue
        I = np.dstack((I,scipy.integrate.cumulative_trapezoid(x_data[:,:,simul], t_data[:,0,simul], axis=0, initial=0)))
        V = np.dstack((V, scipy.integrate.cumulative_trapezoid(v_stator[:,:,simul], t_data[:,0,simul], axis=0, initial=0)))

    # u_data add supply frequency to the input data
    freqs = V_range * 50 / 400  # constant proportion

    freqs = freqs.reshape(1,1,len(freqs)) #along third axis
    u_data = np.hstack((v_stator, I, V, dataset['gamma_rot'] % (2 * np.pi),
                        dataset['omega_rot'],
                        np.repeat(freqs, dataset['omega_rot'].shape[0], axis=0)))

    feature_names = [r'$i_d$',r'$i_q$',r'$i_0$',
                     r'$v_d$', r'$v_q$', r'$v_0$',
                     r'$I_d$', r'$I_q$', r'$I_0$',
                     r'$V_d$', r'$V_q$', r'$V_0$',
                     r'$\gamma$', r'$\omega$', r'$f$']

    # Now, stack data on top of each other and shuffle! (Note that the transpose is needed otherwise the reshape is wrong)
    DATA['x'] = x_data.transpose(0, 2, 1).reshape(x_data.shape[0]*x_data.shape[-1],x_data.shape[1])
    DATA['u'] = u_data.transpose(0, 2, 1).reshape(u_data.shape[0]*u_data.shape[-1],u_data.shape[1])
    DATA['xdot'] = xdots.transpose(0, 2, 1).reshape(xdots.shape[0]*xdots.shape[-1],xdots.shape[1])
    DATA['T_em'] = dataset["T_em"].transpose(0, 2, 1).reshape(dataset["T_em"].shape[0]*dataset["T_em"].shape[-1])
    DATA['UMP'] = dataset["F_em"].transpose(0, 2, 1).reshape(dataset["F_em"].shape[0]*dataset["F_em"].shape[-1],dataset["F_em"].shape[1])

    # shuffle the DATA entirely, but according to the same shuffle
    shuffled_indices = np.random.permutation(DATA['x'].shape[0])
    DATA['x'] = DATA['x'][shuffled_indices] #debug
    DATA['u'] = DATA['u'][shuffled_indices]
    DATA['xdot'] = DATA['xdot'][shuffled_indices]
    DATA['T_em'] = DATA['T_em'][shuffled_indices]
    DATA['UMP'] = DATA['UMP'][shuffled_indices]

    # split the data into train and validation data
    p = 0.8  # percentage of data to be used for training
    cutidx = int(p * DATA['x'].shape[0])

    x_train = DATA['x'][:cutidx]
    u_train = DATA['u'][:cutidx]
    xdot_train = DATA['xdot'][:cutidx]
    T_em_train = DATA['T_em'][:cutidx]
    UMP_train = DATA['UMP'][:cutidx]
    x_val = DATA['x'][cutidx:]
    u_val = DATA['u'][cutidx:]
    xdot_val = DATA['xdot'][cutidx:]
    T_em_val = DATA['T_em'][cutidx:]
    UMP_val = DATA['UMP'][cutidx:]

    visualise_train_data = False
    if visualise_train_data:
        print(DATA["xdot"].shape)
        plt.plot(DATA['xdot'])
        plt.show()
        # todo think about it
        raise NotImplementedError('Visualisation of training data is not yet implemented')

    if path_to_test_file is not None:
        testset = dict(np.load(path_to_test_file))
        pass

    TESTDATA['feature_names'] = feature_names  # Save the names of u_data for SINDy
    if Torque:
        return T_em_train, x_train, u_train, T_em_val, x_val, u_val, TESTDATA
    elif UMP:
        return UMP_train, x_train, u_train, UMP_val, x_val, u_val, TESTDATA
    return xdot_train, x_train, u_train, xdot_val, x_val, u_val, TESTDATA


if __name__ == "__main__":
    path = os.path.join(os.getcwd(), 'train-data', '07-25', 'IMMEC_0ecc_0.001sec.npz')
    data = prepare_data(path, number_of_trainfiles=5)
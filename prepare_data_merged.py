"""
This file is used for preparing data for the merged model, i.e. the model that predicts the currents and the I and V.
This can be merged with prepare_data.py, but for now it is kept separate for clarity, especially because the SINDy solver
works better on smaller problems.

This .py file is not used and might be outdated.
"""
import random
import scipy
from source import *
from prepare_data import v_abc_calculation, v_abc_estimate_from_line, read_V_from_data

def prepare_data(path_to_data_file,
                 test_data=False,
                 number_of_trainfiles=-1,
                 use_estimate_for_v=False,
                 usage_per_trainfile = 0.2):
    """
    See prepare_data.py for more info.
    """
    # load numpy file
    dataset = dict(np.load(path_to_data_file))  # should be a dictionary

    path_to_simulation_data = os.path.join(os.path.dirname(path_to_data_file), 'SIMULATION_DATA.pkl')  # get out one dir
    if not test_data:
        V_range= read_V_from_data(path_to_simulation_data)
    else:
        V_range = np.array([np.max(dataset['v_applied'])/np.sqrt(2)])

    if number_of_trainfiles == 'all' or number_of_trainfiles == -1:
        number_of_trainfiles = len(V_range)

    # choose random V from V_range
    random_idx = random.sample(range(len(V_range)), number_of_trainfiles)  # the simulations are shuffled
    V_range = V_range[random_idx]

    # initialise data
    DATA = {'x': np.array([]), 'u': np.array([]), 'xdot': np.array([]),
            'T_em': np.array([]), 'UMP': np.array([]), 'feature_names': np.array([])}

    # crop dataset to desired amount of simulations (random_idx)
    # also time_trim the arrays
    if not test_data:
        for key in dataset.keys():
            dataset[key] = dataset[key][:, :, random_idx]

    # prepare v data, note that I and V should be in x_data as well
    if use_estimate_for_v:
        v_stator = reference_abc_to_dq0(v_abc_estimate_from_line(dataset))
        v_stator = v_stator[:-1]  # crop last datapoint (consistent with the else)
    else:
        v_stator = reference_abc_to_dq0(v_abc_calculation(dataset, path_to_motor_info=path_to_simulation_data))
        # v_stator is one shorter, so must the other data!

    i_st = reference_abc_to_dq0(dataset['i_st'])

    if np.ndim(i_st) <= 2: #expand such that the code works for both 2D and 3D data
        i_st = np.expand_dims(i_st, axis=2)
        v_stator = np.expand_dims(v_stator, axis=2)
        dataset["time"] = np.expand_dims(dataset['time'], axis=2)
        dataset['T_em'] = np.expand_dims(dataset['T_em'], axis=2)
        dataset['F_em'] = np.expand_dims(dataset['F_em'], axis=2)


    # get u data: potentials_st, i_st, omega_rot, gamma_rot, and the integrals.
    for simul in range(len(random_idx)):
        if simul == 0:  # initiliaze
            I = scipy.integrate.cumulative_trapezoid(i_st[:, :, simul], dataset['time'][:, 0, simul], axis=0, initial=0)
            V = scipy.integrate.cumulative_trapezoid(v_stator[:, :, simul], dataset['time'][:, 0, simul], axis=0,
                                                     initial=0)
            continue
        I = np.dstack(
            (I,
             scipy.integrate.cumulative_trapezoid(i_st[:, :, simul], dataset['time'][:, 0, simul], axis=0, initial=0)))
        V = np.dstack(
            (V, scipy.integrate.cumulative_trapezoid(v_stator[:, :, simul], dataset['time'][:, 0, simul], axis=0,
                                                     initial=0)))

    # get x data, should contain i, I and V
    x_data = np.hstack((i_st, I.reshape(i_st.shape), V.reshape(i_st.shape)))
    t_data = dataset['time']

    # first, calculate xdots and add to the data
    xdots = calculate_xdot(x_data, t_data)

    if not test_data: # trim times AFTER xdots calculation
        print('time trim: ', usage_per_trainfile)
        timepoints = len(dataset['time'][:,0,0])
        time_trim = random.sample(range(timepoints),int(usage_per_trainfile*timepoints) )
        for key in dataset.keys():
            dataset[key] = dataset[key][time_trim]

        v_stator = v_stator[time_trim]
        x_data = x_data[time_trim]
        t_data = t_data[time_trim]
        xdots = xdots[time_trim]


    # u_data add supply frequency to the input data
    freqs = V_range * 50 / 400  # constant proportion

    freqs = freqs.reshape(1, 1, len(freqs))  # along third axis
    u_data = np.hstack((v_stator, dataset['gamma_rot'].reshape(dataset['time'].shape) % (2 * np.pi),
                        dataset['omega_rot'].reshape(dataset['time'].shape),
                        np.repeat(freqs, dataset['omega_rot'].shape[0], axis=0)))

    DATA['feature_names'] = [r'$i_d$', r'$i_q$', r'$i_0$',
                             r'$I_d$', r'$I_q$', r'$I_0$',
                             r'$V_d$', r'$V_q$', r'$V_0$',
                             r'$v_d$', r'$v_q$', r'$v_0$',
                             r'$\gamma$', r'$\omega$', r'$f$']

    # Now, stack data on top of each other and shuffle! (Note that the transpose is needed otherwise the reshape is wrong)
    DATA['x'] = x_data.transpose(0, 2, 1).reshape(x_data.shape[0] * x_data.shape[-1], x_data.shape[1])
    DATA['u'] = u_data.transpose(0, 2, 1).reshape(u_data.shape[0] * u_data.shape[-1], u_data.shape[1])
    DATA['xdot'] = xdots.transpose(0, 2, 1).reshape(xdots.shape[0] * xdots.shape[-1], xdots.shape[1])
    DATA['T_em'] = dataset["T_em"].transpose(0, 2, 1).reshape(dataset["T_em"].shape[0] * dataset["T_em"].shape[-1])
    DATA['UMP'] = dataset["F_em"].transpose(0, 2, 1).reshape(dataset["F_em"].shape[0] * dataset["F_em"].shape[-1],
                                                             dataset["F_em"].shape[1])

    if test_data:
        DATA['V'] = V_range
        DATA['t'] = t_data
        return DATA #return here such that it is not shuffled

    # shuffle the DATA entirely, but according to the same shuffle
    shuffled_indices = np.random.permutation(DATA['x'].shape[0])
    DATA['x'] = DATA['x'][shuffled_indices]  # debug
    DATA['u'] = DATA['u'][shuffled_indices]
    DATA['xdot'] = DATA['xdot'][shuffled_indices]
    DATA['T_em'] = DATA['T_em'][shuffled_indices]
    DATA['UMP'] = DATA['UMP'][shuffled_indices]

    # split the data into train and validation data
    p = 0.8  # percentage of data to be used for training
    cutidx = int(p * DATA['x'].shape[0])

    DATA['x_train'] = DATA['x'][:cutidx]
    DATA['u_train'] = DATA['u'][:cutidx]
    DATA['xdot_train'] = DATA['xdot'][:cutidx]
    DATA['T_em_train'] = DATA['T_em'][:cutidx]
    DATA['UMP_train'] = DATA['UMP'][:cutidx]
    DATA['x_val'] = DATA['x'][cutidx:]
    DATA['u_val'] = DATA['u'][cutidx:]
    DATA['xdot_val'] = DATA['xdot'][cutidx:]
    DATA['T_em_val'] = DATA['T_em'][cutidx:]
    DATA['UMP_val'] = DATA['UMP'][cutidx:]

    return DATA


if __name__ == "__main__":
    path = os.path.join(os.getcwd(), 'train-data', '07-25', 'IMMEC_0ecc_0.001sec.npz')
    data = prepare_data(path, number_of_trainfiles=5)

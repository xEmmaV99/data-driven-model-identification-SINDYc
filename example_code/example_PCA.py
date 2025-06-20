"""
This file was used for the PCA analysis, it adapts the function `prepare_data` to allow for coordinate transformation
"""
import random
import pandas as pd
import scipy
import numpy as np
import os
import pickle as pkl
import numba as nb
from matplotlib import pyplot as plt
from sklearn import decomposition
from sklearn.linear_model import Lasso
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import pysindy as ps
import libs
plt.rcParams['text.usetex'] = True
try:
    from pysindy import FiniteDifference
except ImportError:
    print("Skipping import of pysindy")


def prepare_data(path_to_data_file,
                 test_data=False,
                 number_of_trainfiles=-1,
                 use_estimate_for_v=False,
                 usage_per_trainfile=0.25, pca = None, scaling = None, useScaling = True):
    # load numpy file
    print("Loading data")
    dataset = dict(np.load(path_to_data_file))  # should be a dictionary
    print("Done loading data")

    path_to_simulation_data = os.path.join(os.path.dirname(path_to_data_file), 'SIMULATION_DATA.pkl')
    if not test_data:
        V_range, load_range = read_V_load_from_simulationdata(path_to_simulation_data)
    else:
        V_range = np.array([np.max(dataset['v_applied']) / np.sqrt(2)])

    # choose random V from V_range
    if number_of_trainfiles == -1 or number_of_trainfiles == 'all':
        number_of_trainfiles = len(V_range)

    random_idx = random.sample(range(len(V_range)), number_of_trainfiles)  # the simulations are shuffled
    V_range = V_range[random_idx]

    # initialise data
    DATA = {'x': np.array([]), 'u': np.array([]), 'xdot': np.array([]),
            'T_em': np.array([]), 'UMP': np.array([]), 'feature_names': np.array([])}  # ,'wcoe':np.array([])}

    # crop dataset to desired amount of simulations (random_idx)
    if not test_data:
        for key in dataset.keys():
            dataset[key] = dataset[key][:, :, random_idx]

    # prepare v data
    if use_estimate_for_v:
        v_stator = reference_abc_to_dq0(v_abc_estimate_from_line(dataset["v_applied"]))  # debug
    else:
        v_stator = reference_abc_to_dq0(v_abc_calculation(dataset, path_to_motor_info=path_to_simulation_data))  # debug

    i_st = reference_abc_to_dq0(dataset['i_st'])

    if np.ndim(i_st) <= 2:  # expand such that the code works for both 2D and 3D data
        i_st = np.expand_dims(i_st, axis=2)
        v_stator = np.expand_dims(v_stator, axis=2)
        dataset["time"] = np.expand_dims(dataset['time'], axis=2)
        dataset['T_em'] = np.expand_dims(dataset['T_em'], axis=2)
        dataset['F_em'] = np.expand_dims(dataset['F_em'], axis=2)
        # dataset['wcoe'] = np.expand_dims(dataset['wcoe'], axis=2)

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
            dataset[key] = dataset[key][time_trim]

        v_stator = v_stator[time_trim]
        x_data = x_data[time_trim]
        t_data = t_data[time_trim]
        xdots = xdots[time_trim]
        I = I[time_trim]
        V = V[time_trim]

    # u_data add supply frequency to the input data
    freqs = V_range * 50 / 400  # constant proportion

    freqs = freqs.reshape(1, 1, len(freqs))  # along third axis
    u_data = np.hstack((v_stator,
                        I.reshape(v_stator.shape),
                        V.reshape(v_stator.shape),
                        dataset['gamma_rot'].reshape(t_data.shape) % (2 * np.pi),
                        dataset['omega_rot'].reshape(t_data.shape),
                        np.repeat(freqs, dataset['omega_rot'].shape[0], axis=0)))


    # u_data contains ALL variables needed for prediction
    # principal component analysis on u_data
    # First, I need to reshape the data to be 2D
    u_pca = np.hstack((x_data, v_stator,
                        I.reshape(v_stator.shape),
                        V.reshape(v_stator.shape),
                        dataset['gamma_rot'].reshape(t_data.shape) % (2 * np.pi),
                        dataset['omega_rot'].reshape(t_data.shape),
                        np.repeat(freqs, dataset['omega_rot'].shape[0], axis=0)))
    u_pca = u_pca.transpose(0, 2, 1).reshape(u_pca.shape[0] * u_pca.shape[-1], u_pca.shape[1])

    # Use fit and transform method
    do_pca_analysis = True
    if pca is None:
        # Scale data before applying PCA
        scaling = StandardScaler()
        scaling.fit(u_pca)
        Scaled_data = scaling.transform(u_pca)
        Non_Scaled_data = u_pca

        if do_pca_analysis:
            pca = decomposition.PCA()
            pca.fit(Scaled_data)

            print(pca.components_)
            plot_pca_heatmap(pca, featurenames=[r'$i_d$', r'$i_q$', r'$i_0$',r'$v_d$',r'$v_q$',r'$v_0$',
                                                r'$I_d$', r'$I_q$', r'$I_0$',r'$V_d$', r'$V_q$', r'$V_0$',
                                                r'$\gamma_{rot}$', r'$\omega_{rot}$', r'$f$'],
                             title = "Scaled data")
            plot_pca_variance_ratio(pca, title="on Scaled data")
            pca = decomposition.PCA()
            pca.fit(Non_Scaled_data)

            print(pca.components_)
            plot_pca_heatmap(pca, featurenames=[r'$i_d$', r'$i_q$', r'$i_0$',r'$v_d$',r'$v_q$',r'$v_0$',
                                                r'$I_d$', r'$I_q$', r'$I_0$',r'$V_d$', r'$V_q$', r'$V_0$',
                                                r'$\gamma_{rot}$', r'$\omega_{rot}$', r'$f$'],
                             title = "Non-scaled data")
            plot_pca_variance_ratio(pca, title = "on Non-scaled data")

        pca = decomposition.PCA(n_components=10) #n_components=7
        if useScaling:
            pca.fit(Scaled_data)
            u_pca = pca.transform(Scaled_data)
        else:
            pca.fit(Non_Scaled_data)
            u_pca = pca.transform(Non_Scaled_data)
    else: #pca is provided
        Scaled_data = scaling.transform(u_pca)
        Non_Scaled_data = u_pca
        if useScaling:
            u_pca = pca.transform(Scaled_data)
        else:
            u_pca = pca.transform(Non_Scaled_data)
    DATA['u_pca'] = u_pca

    # Now, stack data on top of each other and shuffle! (Note that the transpose is needed otherwise the reshape is wrong)
    DATA['x'] = x_data.transpose(0, 2, 1).reshape(x_data.shape[0] * x_data.shape[-1], x_data.shape[1])
    DATA['u'] = u_data.transpose(0, 2, 1).reshape(u_data.shape[0] * u_data.shape[-1], u_data.shape[1])
    DATA['xdot'] = xdots.transpose(0, 2, 1).reshape(xdots.shape[0] * xdots.shape[-1], xdots.shape[1])
    DATA['T_em'] = dataset["T_em"].transpose(0, 2, 1).reshape(dataset["T_em"].shape[0] * dataset["T_em"].shape[-1])
    DATA['UMP'] = dataset["F_em"].transpose(0, 2, 1).reshape(dataset["F_em"].shape[0] * dataset["F_em"].shape[-1],
                                                             dataset["F_em"].shape[1])
    # DATA['wcoe'] = dataset['wcoe'].transpose(0, 2, 1).reshape(dataset['wcoe'].shape[0] * dataset['wcoe'].shape[-1])

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
    # DATA['wcoe'] = DATA['wcoe'][shuffled_indices]
    DATA['u_pca'] = DATA['u_pca'][shuffled_indices]

    # split the data into train and validation data
    p = 0.8  # percentage of data to be used for training
    cutidx = int(p * DATA['x'].shape[0])

    DATA['x_train'] = DATA['x'][:cutidx]
    DATA['u_train'] = DATA['u'][:cutidx]
    DATA['xdot_train'] = DATA['xdot'][:cutidx]
    DATA['T_em_train'] = DATA['T_em'][:cutidx]
    DATA['UMP_train'] = DATA['UMP'][:cutidx]
    # DATA['wcoe_train'] = DATA['wcoe'][:cutidx]
    DATA['u_pca_train'] = DATA['u_pca'][:cutidx]
    DATA['x_val'] = DATA['x'][cutidx:]
    DATA['u_val'] = DATA['u'][cutidx:]
    DATA['xdot_val'] = DATA['xdot'][cutidx:]
    DATA['T_em_val'] = DATA['T_em'][cutidx:]
    DATA['UMP_val'] = DATA['UMP'][cutidx:]
    # DATA['wcoe_val'] = DATA['wcoe'][cutidx:]
    DATA['u_pca_val'] = DATA['u_pca'][cutidx:]

    return DATA, pca, scaling


def plot_pca_heatmap(pca, featurenames, title=""):
    plt.figure()
    d = pd.DataFrame(pca.components_, columns=featurenames)
    sns.heatmap(d, cmap='coolwarm', center=0)
    plt.ylabel("Principal component")
    plt.title(title)
    plt.tight_layout()
    plt.show()

    return
def plot_pca_variance_ratio(pca, title=""):
    plt.figure()
    plt.title("Variance ratio "+title)
    plt.xlabel("Number of components")
    plt.ylabel("Percentage of variance")
    plt.ylim([-0.1,1.1])
    plt.plot(pca.explained_variance_ratio_)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.legend(["Variance ratio", "Cumulative variance ratio"])
    plt.figure()
    plt.title("Cumulative variance ratio "+title)
    plt.xlabel("Number of components")
    plt.ylabel("Percentage of variance")
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.yscale("log")
    plt.show()


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
        return FiniteDifference(order=4, axis=0)._differentiate(x, t_)
        # t should have shape (N, )
    return FiniteDifference(order=4, axis=0)._differentiate(x, t.reshape(t.shape[0]))  # Default order is two.


if __name__ == "__main__":
    #Choose the datafiles
    #path = os.path.join(os.getcwd(), 'train-data', '07-29-default', 'IMMEC_0ecc_5.0sec.npz')
    path = os.path.join(os.getcwd(), 'train-data', '07-31-ecc-50', 'IMMEC_50ecc_5.0sec.npz')
    #testpath = os.path.join(os.getcwd(), 'test-data', '07-29', 'IMMEC_0ecc_5.0sec.npz')
    testpath = os.path.join(os.getcwd(), 'test-data', '08-05','IMMEC_50eccecc_5.0sec.npz')

    # the scaling-transformation and pca-transformation needs to be saved -> the exact same transformation is needed for the training data
    useScaling = True
    data,pca,scaling = prepare_data(path,
                        test_data=False,
                        number_of_trainfiles=-1,
                        use_estimate_for_v=False, useScaling = useScaling)
    library = libs.get_custom_library_funcs("pca", nmbr_input_features=data['u_pca_train'].shape[1])

    # Select the model type
    #train = data['T_em_train']
    #train = data['UMP_train']
    train = data['xdot_train']

    name = "pca"
    opt = ps.STLSQ(threshold=0.01)
    #opt = ps.SR3(thresholder="l1", threshold=0.01, nu=.01, unbias=True, normalize_columns = True)
    #opt = ps.WrappedOptimizer(Lasso(alpha=10, fit_intercept=False, precompute=True))
    model = ps.SINDy(optimizer=opt, feature_library=library)
                     #feature_names=data['feature_names']) # Feature names don't make sense anymore

    print("Fitting model")
    model.fit(data['u_pca_train'], t=None, x_dot=train)
    model.print(precision=10)

    test = prepare_data(testpath,
                     test_data=True,
                     number_of_trainfiles=-1,
                     use_estimate_for_v=False, pca=pca, scaling=scaling, useScaling=useScaling) #pass the pca and scaling from the model training

    # Should be accordingly to the train values.
    #test_values = test['T_em'] # for the torque, it is a bad idea to tranform coordinates, don't need to show
    #test_values = test['UMP']
    test_values = test['xdot']

    test_predicted = model.predict(test['u_pca'])

    #test_values = test['x']
    #test_predicted = model.simulate(test_values[0,:], t = test['t'].reshape(test['t'].shape[0]))

    print("MSE on test: ", mean_squared_error(test_values, test_predicted))
    print("Sparsity: ", np.count_nonzero(model.coefficients()))
    plt.figure()
    plt.plot(test_predicted)
    plt.plot(test_values, 'k:')

    plt.show()
import sklearn.utils
from tqdm import tqdm
from immec import *
import matplotlib.pyplot as plt
import numpy as np
import pysindy as ps
import seaborn as sns
import os
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error


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
    return x_train, u_train, t_train, x_valid, u_valid, t_valid


def save_motor_data(motor_path, save_path, extra_dict=None):
    motordict = read_motordict(motor_path)
    motor_model = MotorModel(motordict, 1e-4, "wye", solver='newton')

    dictionary = {}
    dictionary['stator_leakage_inductance'] = motor_model.stator_leakage_inductance
    dictionary['N_abc_T'] = motor_model.N_abc_T
    dictionary['R_st'] = motor_model.R_st

    for key in extra_dict.keys():  # add extra things
        dictionary[key] = extra_dict[key]

    with open(save_path + '/SIMULATION_DATA.pkl', 'wb') as file:
        pkl.dump(dictionary, file)
    return


def create_and_save_immec_data(timestep, t_end, path_to_motor, save_path, V=400, mode='linear', solving_tolerance=1e-4):
    # V should always be below 400, minimal V is 40 (means 5hz f)
    motordict = read_motordict(path_to_motor)
    stator_connection = 'wye'

    if mode == 'linear':
        motor_model = MotorModel(motordict, timestep, stator_connection)
    else:
        motor_model = MotorModel(motordict, timestep, stator_connection, solver='newton',
                                 solving_tolerance=solving_tolerance)

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


def create_immec_data(timestep, t_end, path_to_motor, save_path, V=400, mode='linear', solving_tolerance=1e-4):
    # V should always be below 400, minimal V is 40 (means 5hz f)
    motordict = read_motordict(path_to_motor)
    stator_connection = 'wye'

    if mode == 'linear':
        motor_model = MotorModel(motordict, timestep, stator_connection)
    else:
        motor_model = MotorModel(motordict, timestep, stator_connection, solver='newton',
                                 solving_tolerance=solving_tolerance)

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
    return data_logger


def v_abc_exact(data_logger, path_to_motor_info):
    with open(path_to_motor_info, 'rb') as file:
        motorinfo = pkl.load(file)

    # input the entire data_logger
    R = motorinfo['R_st']
    Nt = motorinfo['N_abc_T']  # model.N_abc
    L_s = motorinfo['stator_leakage_inductance']  # model.stator_leakage_inductance

    dt = data_logger['time'][-1] - data_logger['time'][-2]
    dphi = 1 / dt * np.diff(data_logger['flux_st_yoke'].T)  # forward euler, flux_st_yoke, this array is one shorter
    di = 1 / dt * np.diff(data_logger['i_st'].T)  # forward euler, flux_st_yoke, !!! this array is one shorter !!!

    ist = data_logger['i_st'][:-1, :]  # remove the last value of i_st
    v_abc = np.dot(R, ist.T) + np.dot(Nt, dphi) + L_s * di

    return v_abc.T


def read_V_from_directory(path_to_directory):
    # most datafiles have the name IMMEC_history_{voltage}V_1.0sec.pkl, so this function reads the voltage from the
    # directory name, return an array with all voltages from files in this directory
    files = os.listdir(path_to_directory)
    V = []
    for file in files:
        if file.endswith('.pkl') and file.startswith('IMMEC_history'):
            V.append(int(file.split('_')[2][:-1]))
    return np.array(V)


def v_abc_estimate(data_logger):
    # actually, only data_logger['v_applied'] is needed here
    # outputs Nx3 array
    T = np.array([[1, -1, 0],
                  [1, 2, 0],
                  [-2, -1, 0]])
    return 1 / 3 * np.dot(T, data_logger['v_applied'].T).T


def calculate_xdot(x, t):  # use pySINDy to calculate xdot
    return ps.FiniteDifference(order=2, axis=0)._differentiate(x, t.reshape(t.shape[0]))  # Default order is two.


def save_plot_data(save_name, xydata, title, xlab, ylab, legend=None, plot_now=False, specs=None, sindy_model=None):
    # xydata contains the data to plot, but if multiple axis should be plotted, xy data should be a list of arrays
    # if it is only one x,y then [np.array([x,y])] should be the input
    # create the dictionary to save as is
    pltdata = {'title': title, 'xlab': xlab, 'ylab': ylab, 'legend': legend, 'plots': {}, 'specs': specs,
               'model': sindy_model}
    for i, xy_array in enumerate(xydata):
        pltdata['plots'][str(i)] = xy_array
    cwd = os.getcwd()
    save_path = os.path.join(cwd, 'plot_data\\', save_name + '.pkl')

    with open(save_path, 'wb') as file:
        pkl.dump(pltdata, file)
    if plot_now:
        plot_data(save_path)
    return save_path


def plot_data(path='plotdata.pkl', show=True, figure=True, limits = None):
    with open(path, 'rb') as file:
        data = pkl.load(file)

    if type(data['ylab']) != str:  # multiple axis
        print("Multiple axis plot detected.")
        print("loglog ax1 and semilogx ax2.")
        fig, ax1 = plt.subplots()
        ax1.set_xlabel(data['xlab'])

        ax1.set_ylabel(data['ylab'][0], color='r')
        ax1.loglog(data['plots']['0'][:, 0], data['plots']['0'][:, 1:], 'r')

        ax2 = ax1.twinx()
        ax2.set_ylabel(data['ylab'][1], color='b')
        ax2.semilogx(data['plots']['1'][:, 0], data['plots']['1'][:, 1:], 'b')

        plt.title(data['title'])
        if limits == None:
            fig.tight_layout()
        else:
            ax1.set_ylim(limits[0])
            ax2.set_ylim(limits[1])

    else:
        plt.figure()
        plt.xlabel(data['xlab']), plt.ylabel(data['ylab'])
        specs = data['specs']
        for idx in data['plots']:
            if specs[int(idx)] is not None:
                plt.plot(data['plots'][idx][:, 0], data['plots'][idx][:, 1:], specs[int(idx)])
            else:
                plt.plot(data['plots'][idx][:, 0], data['plots'][idx][:, 1:])

        plt.legend(data['legend'])
        plt.title(data['title'])
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

        sns.heatmap(
            coefs[:, :len(input_names)].T,
            **heatmap_args
        )

        ax.tick_params(axis="y", rotation=0)
    return


def plot_coefs2(model, normalize_values=False, show=False):
    xticknames = model.get_feature_names()
    for i in range(len(xticknames)):
        xticknames[i] = xticknames[i]
    plt.figure(figsize=(len(xticknames), 4))
    colors = ["b", "r", "k"]
    coefs = model.coefficients()

    if normalize_values:
        raise NotImplementedError("This function is not implemented yet.")  # todo

    for i in range(2):
        plt.scatter(np.arange(0, len(xticknames), 1), coefs[i, :].T, color=colors[i],
                    label=r"Equation for $\dot{" + xticknames[i + 1].strip("$") + "}$")

    plt.grid(True)
    plt.xticks(range(len(xticknames)), xticknames, rotation=90)

    plt.legend()
    if show:
        plt.show()
    return


def save_model_coef(model, name):
    #todo!!!!!!!!!!!!
    path = 'C:/Users/emmav/PycharmProjects/SINDY_project/models/' + name + '.pkl'
    lib = {'coefs': model.coefficients(), 'features': model.feature_names()}
    with open(path, 'wb') as file:
        pkl.dump(lib, file)


def read_model_coef(name):
    path = 'C:/Users/emmav/PycharmProjects/SINDY_project/models/' + name + '.pkl'
    with open(path, 'rb') as file:
        coefs = pkl.load(file)
    return coefs


def parameter_search(parameter_array, train_and_validation_data, method="lasso", name="", plot_now=True, library=None):
    if name == "":
        name = method
    method = method.lower()
    variable = {'parameter': parameter_array,
                'MSE': np.zeros(len(parameter_array)),
                'SPAR': np.zeros(len(parameter_array)),
                'model': list(np.zeros(len(parameter_array)))}

    xdot_train, x_train, u_train, xdot_val, x_val, u_val = train_and_validation_data

    for i, para in enumerate(parameter_array):
        print(i)
        if method[:3] == 'sr3':
            optimizer = ps.SR3(thresholder=method[-2:], threshold=para)
        elif method == 'lasso':
            optimizer = Lasso(alpha=para, fit_intercept=False)
        elif method == 'stlsq':
            optimizer = ps.STLSQ(threshold=0.1, alpha=para)
        elif method == 'srr':
            optimizer = ps.SSR(alpha=para)
        else:
            raise NameError("Method is invalid")

        if library is None:
            library = ps.PolynomialLibrary(degree=2, include_interaction=True)

        model = ps.SINDy(optimizer=optimizer, feature_library=library)
        print('Fitting model')
        model.fit(x_train, u=u_train, t=None, x_dot=xdot_train)  # fit on training data
        if model.coefficients().ndim == 1:  # fix dimensions of this matrix, bug in pysindy, o this works
            model.optimizer.coef_ = model.coefficients().reshape(1, model.coefficients().shape[0])
        variable['MSE'][i] = model.score(x_val, u=u_val, x_dot=xdot_val, metric=mean_squared_error)
        # the same as: mean_squared_error(model.predict(x_val, u_val), xdot_val) 
        variable['SPAR'][i] = np.count_nonzero(model.coefficients())  # number of non-zero elements
        variable['model'][i] = model

    # rel_sparsity = variable['SPAR'] / np.max(variable['SPAR'])

    # plot the results
    # save the plots
    xlab = r'Sparsity weighting factor $\alpha$'
    ylab1 = 'MSE'
    ylab2 = 'Number of non-zero elements'
    title = 'MSE and sparsity VS Sparsity weighting parameter, ' + method + ' method'
    xydata = [
        np.hstack(
            (variable['parameter'].reshape(len(parameter_array), 1), variable['MSE'].reshape(len(parameter_array), 1))),
        np.hstack((variable['parameter'].reshape(len(parameter_array), 1),
                   variable['SPAR'].reshape(len(parameter_array), 1)))]
    specs = ['r', 'b']
    save_plot_data(name, xydata, title, xlab, [ylab1, ylab2], specs, plot_now=plot_now)

    idx = np.where(np.min(variable['MSE']))  # best model, lowest MSE
    best_model = variable['model'][idx[0][0]]
    print("Best model found with MSE: ", variable['MSE'][idx[0][0]], " and parameter: ",
          variable['parameter'][idx[0][0]], "for ", method)
    return best_model



def plot_everything(path_to_directory):
    files = os.listdir(path_to_directory)
    for file in files:
        if file.endswith('.pkl'):
            path = os.path.join(path_to_directory, file)
            plot_data(path, show=False)
    plt.show()
    return

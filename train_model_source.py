from source import *
from prepare_data import prepare_data
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import datetime


def make_model(path_to_data_files, modeltype, optimizer, nmbr_of_train=-1, lib="",
               alpha=None, nu=None, lamb=None, modelname=None):
    """
    Make model
    :param path_to_data_files:
    :param modeltype:
    :param optimizer:
    :param nmbr_of_train:
    :param lib:
    :param alpha:
    :param nu:
    :param lamb:
    :param modelname:
    :return:
    """
    DATA = prepare_data(path_to_data_files, number_of_trainfiles=nmbr_of_train)
    library = get_custom_library_funcs(lib)

    # merge the make-model functions
    if modeltype == 'torque':
        train = DATA['T_em_train']
        val = DATA['T_em_val'] # only for MSE calculation
        name = "Torque"
    elif modeltype == 'ump':
        train = DATA['UMP_train']
        val = DATA['UMP_val'] # only for MSE calculation
        name = "UMP"
    elif modeltype == 'torque-ump':
        train = np.hstack((DATA['T_em_train'].reshape(-1, 1), DATA['UMP_train']))
        val = np.hstack((DATA['T_em_val'].reshape(-1, 1), DATA['UMP_val'])) # only for MSE calculation
        name = "Torque-UMP"
    elif modeltype == 'currents':
        train = DATA['xdot_train']
        val = DATA['xdot_val'] # only for MSE calculation
        name = "Currents"
    elif modeltype == 'wcoe':
        raise NotImplementedError("Not implemented yet")
        train = DATA['wcoe_train']
        val = DATA['wcoe_val'] # only for MSE calculation
        name = "Wcoe"
    else:
        raise ValueError("Model type not known")

    if modelname is not None:
        name = modelname #overwrite default name

    # Select the correct optimizer
    if optimizer == 'sr3':
        print("SR3_L1 optimisation")
        opt = ps.SR3(thresholder="l1", threshold=lamb, nu=nu)

    elif optimizer == 'lasso':
        print("Lasso optimisation")
        opt = ps.WrappedOptimizer(Lasso(alpha=alpha, fit_intercept=False, precompute=True, max_iter=1000))
    else:
        raise ValueError("Optimizer not known")

    model = ps.SINDy(optimizer=opt, feature_library=library,
                     feature_names=DATA['feature_names'])

    print("Fitting model")
    model.fit(DATA['x_train'], u=DATA['u_train'], t=None, x_dot=train)
    model.print(precision=10)


    print("SPAR", np.count_nonzero(model.coefficients()))
    print("MSE: " + str(
        model.score(DATA["x_val"], t=None, x_dot=val, u=DATA['u_val'], metric=mean_squared_error)))

    # plot_coefs2(model, show = False, log = True)

    save_model(model, name + "_model", lib)


def simulate_model(model_name, path_to_test_file, modeltype, do_time_simulation=False, show=True):
    model = load_model(model_name)
    model.print()
    TEST = prepare_data(path_to_test_file, test_data=True)
    ## plot_coefs2(model, show=True, log=True)
    # merge the simulation functions
    if modeltype == 'torque':
        test_values = TEST['T_em'].reshape(-1, 1)
    elif modeltype == 'ump':
        test_values = TEST['UMP']
    elif modeltype == 'torque-ump':
        test_values = np.hstack((TEST['T_em'].reshape(-1, 1), TEST['UMP']))
    elif modeltype == 'currents':
        test_values = TEST['xdot']
    elif modeltype == 'wcoe':
        raise NotImplementedError("Not implemented yet")
        test_values = TEST['wcoe']
    else:
        raise ValueError("Model type not known")

    x_test = TEST['x']
    u_test = TEST['u']
    t = TEST['t'][:, :, 0]  # assume the t is the same for all simulations
    test_predicted = model.predict(x_test, u_test)

    print("MSE: ", mean_squared_error(test_values, test_predicted))

    if modeltype == 'currents':
        xydata = [np.hstack((t, test_predicted)), np.hstack((t, test_values))]
        xlab = r"$t$"
        ylab = r'$\dot{x}$'
        title = 'Predicted vs computed derivatives on test set V = ' + str(TEST['V'])
        leg = [r"$\partial_t{i_d}$", r"$\partial_t{i_q}$", r"$\partial_t{i_0}$", "computed"]
        specs = [None, "k--"]
        save_plot_data("currents", xydata, title, xlab, ylab, legend=leg, plot_now=True, specs=specs)

        if do_time_simulation:
            simulation_time = 5.0
            plt.figure()
            print("Starting time simulation")
            t_value = t[t < simulation_time]

            # x_sim = model.simulate(x_test[0, :],
            #                       u=u_test[(t < simulation_time).reshape(-1), :],
            #                       t=t_value.reshape(t_value.shape[0]),
            #                       integrator_kws={'method': 'RK45'})

            x_sim = model_simulate(x_test[0, :],
                                   u=u_test[(t < simulation_time).reshape(-1), :],
                                   t=t_value.reshape(t_value.shape[0]), model=model
                                   ).y.T

            print("Finished simulation")
            print("MSE on simulation: ", mean_squared_error(x_sim, x_test[:len(t_value), :]))

            save_plot_data("currents_simulation",
                           [np.hstack((t_value[:].reshape(len(t_value), 1), x_sim)),
                            np.hstack((t, x_test))],
                           "Simulated currents on test set V = " + str(TEST['V']),
                           r"$t$", r"$x$", plot_now=True, specs=[None, "k--"],
                           legend=[r"$i_d$", r"$i_q$", r"$i_0$", "test data"])
            return x_sim, x_test
        if show:
            plt.show()
        return test_predicted, test_values

    # Compare with alternative approach for Torque calculation on testdata
    if modeltype == 'torque' or modeltype == 'torque-ump':
        # Clarke alternative torque calculation
        V = u_test[:, 6:9]
        I = u_test[:, 3:6]
        x = TEST['x']

        path_to_data_dir = os.path.dirname(path_to_test_file)
        with open(path_to_data_dir + '/SIMULATION_DATA.pkl', 'rb') as f:
            data = pkl.load(f)
        R = data['R_st']
        print(R)
        lambda_dq0 = (V.T - np.dot(R, I.T)).T
        Torque_value = lambda_dq0[:, 0] * x[:, 1] - lambda_dq0[:, 1] * x[:, 0]  # lambda_d * i_q - lambda_q * i_d

        # Save the plots, torque first
        xydata = [np.hstack((t, test_predicted[:, 0].reshape(-1, 1))), np.hstack((t, test_values[:, 0].reshape(-1, 1))),
                  np.hstack((t, Torque_value.reshape(-1, 1)))]
        xlab = r"$t$"
        ylab = r'$T_{em}$ (Newton-meters)'
        title = 'Predicted (SINDy) vs Torque (Clarke) on test set V = ' + str(TEST['V'])
        legend = [r"Predicted by SINDYc", r"Reference", r"Simplified Torque model"]
        specs = [None, "k--", "r--"]
        save_plot_data("Torque", xydata, title, xlab, ylab, legend=legend, plot_now=True, specs=specs)

        # Plot the error of torque
        plt.figure()  # dont plot first point
        plt.semilogy(t[1:], np.abs(test_predicted[:, 0] - test_values[:, 0])[1:])
        plt.semilogy(t[1:], np.abs(Torque_value.reshape(test_values[:, 0].shape) - test_values[:, 0])[1:], 'r--')
        plt.legend([r"Predicted by SINDYc", r"Simplified Torque model"])
        plt.title(r'Absolute error compared to Reference')
        plt.xlabel(r"$t$")
        plt.ylabel(r"$\log_{10}|T_{em} - T_{em,ref}|$")

    if modeltype == 'ump' or modeltype == 'torque-ump':
        # Save the plots for UMP
        xydata = [np.hstack((t, test_predicted[:, 0].reshape(len(t), 1))),
                  np.hstack((t, test_predicted[:, 1].reshape(len(t), 1))), np.hstack((t, test_values[:, -2:]))]
        xlab = r"$t$"
        ylab = r'$UMP$ (Newton)'
        title = 'Predicted (SINDy) vs simulated UMP on test set V = ' + str(TEST['V'])
        legend = [r"Predicted UMP_x", r"Predicted UMP_y", r"Reference"]
        specs = ["b", "r", "k--"]
        save_plot_data("UMP", xydata, title, xlab, ylab, legend=legend, plot_now=True, specs=specs)
    if show:
        plt.show()
    return test_predicted, test_values

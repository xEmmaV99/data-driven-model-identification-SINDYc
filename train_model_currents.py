from optimize_parameters import parameter_search, optimize_parameters, plot_optuna_data
from source import *
from prepare_data import prepare_data
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error


def make_model_currents(path_to_data_files, alpha, optimizer='sr3', nmbr_of_train=-1):
    """
    Simulation for the currents and compared with testdata
    :param path_to_data_files:
    :param alpha:
    :param optimizer:
    :param path_to_test_file:
    :param do_time_simulation:
    :return:
    """

    DATA = prepare_data(path_to_data_files, number_of_trainfiles=nmbr_of_train)

    lib = 'poly_2nd_order'
    library = get_custom_library_funcs(lib)

    if optimizer == 'sr3':
        opt = ps.SR3(thresholder="l1", threshold=alpha, nu=0.1)
        print("SR3_L1 optimisation, with nu = 0.1 ")
    elif optimizer == 'lasso':
        opt = Lasso(alpha=alpha, fit_intercept=False)
        print("Lasso optimisation")
    else:
        raise ValueError("Unknown optimizer")

    model = ps.SINDy(optimizer=opt, feature_library=library,
                     feature_names=DATA['feature_names'])

    print("Fitting model")
    model.fit(DATA["x_train"], u=DATA["u_train"], t=None, x_dot=DATA['xdot_train'])
    model.print()

    print("MSE: " + str(
        model.score(DATA["x_val"], t=None, x_dot=DATA['xdot_val'], u=DATA['u_val'], metric=mean_squared_error)))
    # plot_coefs2(model, log=True)

    save_model(model, "currents_model", lib)


def simulate_currents(model_name, path_to_test_file, do_time_simulation=False):
    model = load_model(model_name)
    model.print()
    TEST = prepare_data(path_to_test_file, test_data=True)

    # Validate the best model, on testdata
    xdot_test = TEST['xdot']
    x_test = TEST['x']
    u_test = TEST['u']
    t = TEST['t'].reshape((TEST['t'].shape[0], 1))

    # Predict derivatives using the learned model
    x_dot_test_predicted = model.predict(x_test, u_test)

    xydata = [np.hstack((t, x_dot_test_predicted)), np.hstack((t, xdot_test))]
    xlab = r"$t$"
    ylab = r'$\dot{x}$'
    title = 'Predicted vs computed derivatives on test set V = ' + str(TEST['V'])
    leg = [r"$\partial_t{i_d}$", r"$\partial_t{i_q}$", r"$\partial_t{i_0}$", "computed"]
    specs = [None, "k--"]
    save_plot_data("currents", xydata, title, xlab, ylab, legend=leg, plot_now=True,
                   specs=specs)

    print("MSE on testdata: "+ str(mean_squared_error(x_dot_test_predicted, xdot_test)))
    if do_time_simulation:
        simulation_time = 1.0
        plt.figure()
        print("Starting simulation")
        t_value = t[t < simulation_time]

        x_sim = model.simulate(x_test[0, :],
                               u=u_test[(t < simulation_time).reshape(-1), :],
                               t=t_value.reshape(t_value.shape[0]),
                               integrator_kws={'method': 'RK23'})  # todo : change integration method

        print("Finished simulation")

        save_plot_data("currents_simulation",
                       [np.hstack((t_value[:-1].reshape(len(t_value) - 1, 1), x_sim)),
                        np.hstack((t, x_test))],
                       "Simulated currents on test set V = " + str(TEST['V']),
                       r"$t$", r"$x$", plot_now=True, specs=[None, "k--"],
                       legend=[r"$i_d$", r"$i_q$", r"$i_0$", "test data"])

    plt.show()
    return x_dot_test_predicted, xdot_test


if __name__ == "__main__":
    path_to_data_files = os.path.join(os.getcwd(), 'train-data', '07-29-default', 'IMMEC_0ecc_5.0sec.npz')

    ### OPTIMIZE ALPHA
    #optimize_parameters(path_to_data_files, mode = "currents")

    ### PLOT MSE AND SPARSEITY FOR DIFFERENT PARAMETERS
    #plot_optuna_data('currents-lasso-study')
    #plot_optuna_data('currents-sr3-study')

    ### CREATE A MODEL
    make_model_currents(path_to_data_files, alpha=1, optimizer='lasso', nmbr_of_train=10)

    ### SIMULATE
    #path_to_test_file = os.path.join(os.getcwd(), 'test-data', '07-29-default', 'IMMEC_0ecc_5.0sec.npz')
    #path_to_test_file = os.path.join(os.getcwd(), 'test-data', '07-29', 'IMMEC_0ecc_5.0sec.npz')
    #simulate_currents('currents_model', path_to_test_file, do_time_simulation=True)

    #plt.show()

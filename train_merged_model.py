from prepare_data_system import *
from libs import *

# todo add constraints

def optimize_currents_simulation(path_to_data_files, nmbr_models=20, loglwrbnd=None, loguprbnd=None):
    """
    Calculates for various parameters, plots MSE and Sparsity, for SR3 and Lasso optimisation
    :param path_to_data_files:
    :param nmbr_models:
    :param loglwrbnd:
    :param loguprbnd:
    :return:
    """
    if loguprbnd is None:
        loguprbnd = [0, 0]
    if loglwrbnd is None:
        loglwrbnd = [-12, -12]

    # now, the xdot_train contains i and V and I,

    DATA = prepare_data(path_to_data_files, number_of_trainfiles=10)

    library = get_custom_library_funcs("default")

    print("using custom library: ", library)

    print("SR3_L1 optimisation")
    parameter_search(np.logspace(loglwrbnd[0], loguprbnd[0], nmbr_models),
                     train_and_validation_data=[DATA["xdot_train"], DATA["x_train"], DATA["u_train"], DATA["xdot_val"], DATA["x_val"], DATA["u_val"]],
                     method="SR3_L1", name="currents_merged_sr3", plot_now=False, library=library)

    print("Lasso optimisation")
    parameter_search(np.logspace(loglwrbnd[1], loguprbnd[1], nmbr_models),
                     train_and_validation_data=[DATA["xdot_train"], DATA["x_train"], DATA["u_train"], DATA["xdot_val"], DATA["x_val"], DATA["u_val"]],
                     method="Lasso", name="currents_merged_lasso", plot_now=False, library=library)
    return


def make_model_currents(path_to_data_files, alpha, optimizer='sr3',  nmbr_of_train = 'all'):
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

    #library = ps.PolynomialLibrary(degree=2, interaction_only=True)
    #lib = "poly_2nd_order"
    lib = "best"
    library = get_custom_library_funcs(lib)
    print(lib)
    #library = ps.PolynomialLibrary(degree=2, include_interaction=True)

    if optimizer == 'sr3':
        opt = ps.SR3(thresholder="l1", threshold=alpha)
        print("SR3_L1 optimisation")
    elif optimizer == 'lasso':
        opt = Lasso(alpha=alpha, fit_intercept=False)
        print("Lasso optimisation")
    else:
        raise ValueError("Unknown optimizer")

    model = ps.SINDy(optimizer=opt, feature_library=library,
                     feature_names=DATA['feature_names'])

    print("Fitting model")
    model.fit(DATA['x_train'], u=DATA["u_train"], t=None, x_dot=DATA["xdot_train"])
    model.print()

    print("MSE: "+ str(model.score(DATA["x_val"], t=None, x_dot=DATA["xdot_val"], u=DATA["u_val"], metric=mean_squared_error)))

    #plot_coefs2(model, log = True)

    save_model(model, "model_merged", libstr = lib)
    return

def simulate_currents(model_name,path_to_test_file, do_time_simulation=False):
    model = load_model(model_name)
    model.print()
    TEST = prepare_data(path_to_test_file, test_data=True)

    # Validate the best model, on testdata
    xdot_test = TEST['xdot'][:,:3]
    x_test = TEST['x']
    u_test = TEST['u']
    t = TEST['t']

    # Predict derivatives using the learned model
    x_dot_test_predicted = model.predict(x_test, u_test)

    x_dot_test_predicted = x_dot_test_predicted[:,:3]

    xydata = [np.hstack((t.reshape(t.shape[0],1), x_dot_test_predicted)), np.hstack((t.reshape(t.shape[0],1), xdot_test))]
    xlab = r"$t$"
    ylab = r'$\dot{x}$'
    title = 'Predicted vs computed derivatives on test set V = ' + str(TEST['V'])
    leg = [r"$\partial_t{i_d}$", r"$\partial_t{i_q}$", r"$\partial_t{i_0}$", "computed"]
    specs = [None, "k--"]
    save_plot_data("currents", xydata, title, xlab, ylab, legend=leg, plot_now=True,
                   specs=specs)

    if do_time_simulation:
        simulation_time = 1.0
        plt.figure()
        print("Starting simulation")
        t_value = t[t < simulation_time]

        x_sim = model.simulate(x_test[0, :],
                               u=u_test[(t < simulation_time).reshape(-1), :],
                               t=t_value.reshape(t_value.shape[0]),
                               integrator_kws={'method': 'RK23'}) # todo : change integration method

        print("Finished simulation")

        save_plot_data("currents_simulation",
                       [np.hstack((t_value[:-1].reshape(len(t_value) - 1, 1), x_sim)),
                        np.hstack((t.reshape(len(t_value), 1), x_test))],
                       "Simulated currents on test set V = " + str(TEST['V']),
                       r"$t$", r"$x$", plot_now=True, specs=[None, "k--"],
                       legend=[r"$i_d$", r"$i_q$", r"$i_0$", "test data"])

    plt.show()
    return


if __name__ == "__main__":

    path_to_data_files = os.path.join(os.getcwd(), 'train-data', "07-29-default","IMMEC_0ecc_5.0sec.npz")

    ### OPTIMIZE ALPHA
    #optimize_currents_simulation(path_to_data_files, nmbr_models=1, loglwrbnd=[-7, -7], loguprbnd=[3, 3])

    ### PLOT MSE FOR DIFFERENT ALPHA
    #plot_data([os.getcwd() + "\\plot_data" + p + ".pkl" for p in ["\\currents_sr3", "\\currents_lasso"]], show=False, limits=[[1e0,5e2],[0,150]])

    ### create a model
    #make_model_currents(path_to_data_files, alpha = 0.001, optimizer='sr3', nmbr_of_train=20)
    #make_model_currents(path_to_data_files, alpha=0.1, optimizer='lasso', nmbr_of_train=25)
    ## SIMULATE CURRENTS
    path_to_test_file = os.path.join(os.getcwd(), 'test-data', "07-29-default","IMMEC_0ecc_5.0sec.npz")
    #path_to_test_file = os.path.join(os.getcwd(), 'test-data', "07-26","IMMEC_0ecc_5.0sec.npz")

    simulate_currents('model_merged', path_to_test_file, do_time_simulation=False)

    plt.show()
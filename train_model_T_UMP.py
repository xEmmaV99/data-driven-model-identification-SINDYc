import matplotlib.pyplot as plt

from preparedata2_tes import *
from libs import *


def optimise_simulation(path_to_data_files, nmbr_models='all', loglwrbnd=None, loguprbnd=None):
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

    DATA = prepare_data(path_to_data_files, number_of_trainfiles=nmbr_models)
    library = get_custom_library_funcs()

    train = np.hstack((DATA['T_train'], DATA['UMP_train']))
    val = np.hstack((DATA['T_val'], DATA['UMP_val']))

    print("SR3_L1 optimisation")
    parameter_search(np.logspace(loglwrbnd[0], loguprbnd[0], nmbr_models),
                     train_and_validation_data=[train, DATA['x_train'], DATA['u_train'], val, DATA['x_val'],
                                                DATA['u_val']],
                     method="SR3_L1", name="torque_sr3", plot_now=False, library=library)
    print("Lasso optimisation")
    parameter_search(np.logspace(loglwrbnd[1], loguprbnd[1], nmbr_models),
                     train_and_validation_data=[train, DATA['x_train'], DATA['u_train'], val, DATA['x_val'],
                                                DATA['u_val']],
                     method="lasso", name="torque_lasso", plot_now=False, library=library)
    path = os.path.join(os.getcwd(), "plot_data")
    for p in ["\\torque_and_UMP_sr3", "\\torque_and_UMP_lasso"]:
        plot_data(path + p + ".pkl")
    return


def simulate(path_to_data_files, alpha, optimizer, path_to_test_file):
    """
    Simulates the Torque on test data, and compares with the Clarke model
    :param path_to_data_files:
    :param alpha:
    :param optimizer:
    :param path_to_test_file:
    :return:
    """
    DATA = prepare_data(path_to_data_files, number_of_trainfiles='all')
    TEST = prepare_data(path_to_test_file, test_data=True)

    if optimizer == 'sr3':
        print("SR3_L1 optimisation")
        opt = ps.SR3(thresholder="l1", threshold=alpha)
    elif optimizer == 'lasso':
        print("Lasso optimisation")
        opt = Lasso(alpha=alpha, fit_intercept=False)
    else:
        raise ValueError("Optimizer not known")

    library = get_custom_library_funcs()
    model = ps.SINDy(optimizer=opt, feature_library=library,
                     feature_names=DATA['feature_names'])
    print("Fitting model")
    model.fit(DATA['x_train'], u=DATA['u_train'], t=None, x_dot=DATA['x_dot_train'])

    if model.coefficients().ndim == 1:  # fix dimensions of this matrix, bug in pysindy, o this works
        model.optimizer.coef_ = model.coefficients().reshape(1, model.coefficients().shape[0])
    model.print()
    # plot_coefs2(model, show = False, log = True)

    save_model(model, "UMP-torque_model")

    # testdata
    UMP_test = TEST['UMP']
    T_test = TEST['T_em']
    test_values = np.hstack((T_test, UMP_test))
    x_test = TEST['x']
    u_test = TEST['u']
    t = TEST['t']
    test_predicted = model.predict(x_test, u_test)

    # Compare with alternative approach for Torque calculation on testdata
    V = u_test[:, 6:9]
    I = u_test[:, 3:6]
    x = TEST['x']
    with open(path_to_data_files + '/SIMULATION_DATA.pkl', 'rb') as f:
        data = pkl.load(f)
    R = data['R_st']
    lambda_dq0 = (V.T - np.dot(R, I.T)).T
    Torque = lambda_dq0[:, 0] * x[:, 1] - lambda_dq0[:, 1] * x[:, 0]  # lambda_d * i_q - lambda_q * i_d

    # Save the plots, torque first
    xydata = [np.hstack((t, test_predicted[:,0])), np.hstack((t, T_test)), np.hstack((t, Torque.reshape(len(t), 1)))]
    xlab = r"$t$"
    ylab = r'$T_{em}$ (Newton-meters)'
    title = 'Predicted (SINDy) vs Torque (Clarke) on test set V = ' + str(TEST['V'])
    legend = [r"Predicted by SINDYc", r"Reference", r"Simplified Torque model"]
    specs = [None, "k--", "r--"]
    save_plot_data("Torque", xydata, title, xlab, ylab, legend=legend, plot_now=True, specs=specs)

    # Save the plots for UMP
    xydata = [np.hstack((t, test_predicted[:, 1].reshape(len(t), 1))),
              np.hstack((t, test_predicted[:, 2].reshape(len(t), 1))), np.hstack((t, UMP_test))]
    xlab = r"$t$"
    ylab = r'$UMP$ (Newton)'
    title = 'Predicted (SINDy) vs simulated UMP on test set V = ' + str(TEST['V'])
    legend = [r"Predicted UMP_x", r"Predicted UMP_y", r"Reference"]
    specs = ["b", "r", "k--"]
    save_plot_data("UMP", xydata, title, xlab, ylab, legend=legend, plot_now=True, specs=specs)

    # Plot the error of torque
    plt.figure()  # dont plot first point
    plt.semilogy(t[1:], np.abs(test_predicted[:,0] - T_test)[1:])
    plt.semilogy(t[1:], np.abs(Torque.reshape(T_test.shape) - T_test)[1:], 'r--')
    plt.legend([r"Predicted by SINDYc", r"Simplified Torque model"])
    plt.title(r'Absolute error compared to Reference')
    plt.xlabel(r"$t$")
    plt.ylabel(r"$\log_{10}|T_{em} - T_{em,ref}|$")
    plt.show()
    return


if __name__ == "__main__":
    path_to_data_files = os.path.join(os.getcwd(), "data/07-24-default-5e-5")

    ### OPTIMISE ALPHA FOR TORQUE SIMULATION
    optimise_simulation(path_to_data_files, nmbr_models='all', loglwrbnd=[-12, -12], loguprbnd=[0, 0])

    ### PLOT MSE FOR TORQUE SIMULATION
    #plot_data([os.getcwd() + "\\plots_2607_presentation" + p + ".pkl" for p in ["\\torque_SR3_", "\\torque_lasso"]],
    #          show=False, limits=[[1e-6, 1], [0, 100]])

    plt.show()
    ### SIMULATE TORQUE WITH CHOSEN ALPHA AND OPTIMIZER
    # simulate_torque(path_to_data_files, alpha=1e-4, optimizer='lasso')
    # plt.show()

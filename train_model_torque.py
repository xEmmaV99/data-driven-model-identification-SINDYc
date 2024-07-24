import matplotlib.pyplot as plt

from prepare_data import *


def optimise_torque_simulation(path_to_data_files, nmbr_models=20, loglwrbnd=None, loguprbnd=None):
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

    T_train, x_train, u_train, T_val, x_val, u_val, _ = prepare_data(path_to_data_files,
                                                                     Torque=True,
                                                                     t_end=1.0,
                                                                     number_of_trainfiles=40)
    library = ps.PolynomialLibrary(degree=2, include_interaction=True)
    print("SR3_L1 optimisation")
    parameter_search(np.logspace(loglwrbnd[0], loguprbnd[0], nmbr_models),
                     train_and_validation_data=[T_train, x_train, u_train, T_val, x_val, u_val],
                     method="SR3_L1", name="torque_sr3", plot_now=False, library=library)
    print("Lasso optimisation")
    parameter_search(np.logspace(loglwrbnd[1], loguprbnd[1], nmbr_models),
                     train_and_validation_data=[T_train, x_train, u_train, T_val, x_val, u_val],
                     method="lasso", name="torque_lasso", plot_now=False, library=library)
    path = os.path.join(os.getcwd(), "plot_data")
    for p in ["\\torque_sr3", "\\torque_lasso"]:
        plot_data(path + p + ".pkl")
    return


def simulate_torque(path_to_data_files, alpha, optimizer='sr3', path_to_test_file=None):
    """
    Simulates the Torque on test data, and compares with the Clarke model
    :param path_to_data_files:
    :param alpha:
    :param optimizer:
    :param path_to_test_file:
    :return:
    """
    T_train, x_train, u_train, T_val, x_val, u_val, testdata = prepare_data(path_to_data_files,
                                                                            Torque=True,
                                                                            path_to_test_file=path_to_test_file,
                                                                            t_end=1.0, number_of_trainfiles=40)
    if optimizer == 'sr3':
        print("SR3_L1 optimisation")
        opt = ps.SR3(thresholder="l1", threshold=alpha)
    elif optimizer == 'lasso':
        print("Lasso optimisation")
        opt = Lasso(alpha=alpha, fit_intercept=False)
    else:
        raise ValueError("Optimizer not known")

    library = ps.PolynomialLibrary(degree=2, include_interaction=True)
    model = ps.SINDy(optimizer=opt, feature_library=library,
                     feature_names=["i_d", "i_q", "i_0"] + testdata['u_names'])
    print("Fitting model")
    model.fit(x_train, u=u_train, t=None, x_dot=T_train)

    if model.coefficients().ndim == 1:  # fix dimensions of this matrix, bug in pysindy, o this works
        model.optimizer.coef_ = model.coefficients().reshape(1, model.coefficients().shape[0])
    model.print()
    # plot_coefs2(model, show = True)

    # testdata
    T_test = testdata['T_em']
    x_test = testdata['x']
    u_test = testdata['u']
    t = testdata['t']
    T_test_predicted = model.predict(x_test, u_test)

    # Compare with alternative approach for Torque calculation on testdata
    V = u_test[:, 6:9]
    I = u_test[:, 3:6]
    x = testdata['x']
    with open(path_to_data_files + '/SIMULATION_DATA.pkl', 'rb') as f:
        data = pkl.load(f)

    R = data['R_st']
    lambda_dq0 = (V.T - np.dot(R, I.T)).T
    Torque = lambda_dq0[:, 0] * x[:, 1] - lambda_dq0[:, 1] * x[:, 0]  # lambda_d * i_q - lambda_q * i_d

    # Save the plots
    xydata = [np.hstack((t, T_test_predicted)), np.hstack((t, T_test)), np.hstack((t, Torque.reshape(len(t), 1)))]
    xlab = r"$t$"
    ylab = r'$T_{em}$ (Newton-meters)'
    title = 'Predicted (SINDy) vs Torque (Clarke) on test set V = ' + str(testdata['V'])
    legend = [r"Predicted by SINDYc", r"Reference", r"Simplified Torque model"]
    specs = [None, "k--", "r--"]
    save_plot_data("torque", xydata, title, xlab, ylab, legend=legend, plot_now=True, specs=specs)

    plt.figure()  # dont plot first point
    plt.semilogy(t[1:], np.abs(T_test_predicted - T_test)[1:])
    plt.semilogy(t[1:], np.abs(Torque.reshape(T_test.shape) - T_test)[1:], 'r--')
    plt.legend([r"Predicted by SINDYc", r"Simplified Torque model"])
    plt.title(r'Absolute error compared to Reference')
    plt.xlabel(r"$t$")
    plt.ylabel(r"$\log_{10}|T_{em} - T_{em,ref}|$")
    plt.show()
    return


if __name__ == "__main__":
    path_to_data_files = os.path.join(os.getcwd(), "data/07-24-default-5e-5")
    # optimise_torque_simulation(path_to_data_files, nmbr_models=20, loglwrbnd=[-12, -12], loguprbnd=[0, 0])
    for p in ["\\torque_sr3", "\\torque_lasso"]:
        plot_data(os.getcwd() + "\\plots" + p + ".pkl", show=False, limits=[[1e-6, 1], [0, 100]])
    plt.show()
    # simulate_torque(path_to_data_files, alpha=1e-3, optimizer='lasso')

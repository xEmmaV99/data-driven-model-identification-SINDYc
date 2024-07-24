from prepare_data import *
def simulate_torque():
    """
    Simulation for the Torque
    """
    path_to_test_file = None
    T_train, x_train, u_train, T_val, x_val, u_val, TESTDATA = prepare_data(path_to_data_files,
                                                                            Torque=True,
                                                                            path_to_test_file=path_to_test_file,
                                                                            t_end=1.0, number_of_trainfiles=40)

    inputs_per_library = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14], [12]]
    big_lib = ps.GeneralizedLibrary([ps.PolynomialLibrary(degree=3, include_interaction=True),
                                     ps.FourierLibrary(n_frequencies=1, include_cos=True, include_sin=True)],
                                    tensor_array=None,  # don't merge the libraries
                                    inputs_per_library=inputs_per_library) # are crossterms needed?

    fit_multiple = False
    if fit_multiple:
        model = theshold_search(np.logspace(-12, -3, 15),
                                train_and_validation_data=[T_train, x_train, u_train, T_val, x_val, u_val],
                                method="SR3_L1", name="torque_opt", plot_now=True, library=big_lib)

    else:
        #threshold = 0.0
        threshold = 1e-2 #for sr3
        #optimizer = ps.SR3(thresholder="l1", threshold=threshold) # this runs, only reasonable for threshold = 0
        optimizer = Lasso(alpha=threshold, fit_intercept=False)  # now, SR3 really is a lot better, it is weird that adding things to the library makes this worse too

        #library = ps.PolynomialLibrary(degree=3, include_interaction=True)
        library = big_lib
        model = ps.SINDy(optimizer=optimizer, feature_library=library,
                         feature_names=["i_d", "i_q", "i_0"] + TESTDATA['u_names'])
        #model.feature_names = ["i_d", "i_q", "i_0"] + TESTDATA['u_names']
        model.fit(x_train, u=u_train, t=None, x_dot=T_train)

    if model.coefficients().ndim == 1:  # fix dimensions of this matrix, bug in pysindy, o this works
        model.optimizer.coef_ = model.coefficients().reshape(1, model.coefficients().shape[0])
    model.print()

    # TESTDATA
    T_test = TESTDATA['T_em']
    x_test = TESTDATA['x']
    u_test = TESTDATA['u']
    t = TESTDATA['t']
    T_test_predicted = model.predict(x_test, u_test)

    # Compare with alternative approach for Torque calculation on TESTDATA
    V = u_test[:, 6:9]
    I = u_test[:, 3:6]
    x = TESTDATA['x']
    with open(path_to_data_files + '/SIMULATION_DATA.pkl', 'rb') as f:
        data = pkl.load(f)

    R = data['R_st']
    lambda_dq0 = (V.T - np.dot(R, I.T)).T
    Torque = lambda_dq0[:, 0] * x[:, 1] - lambda_dq0[:, 1] * x[:, 0]  # lambda_d * i_q - lambda_q * i_d

    # save the plots
    xydata = [np.hstack((t, T_test_predicted)), np.hstack((t, T_test)), np.hstack((t, Torque.reshape(len(t), 1)))]
    xlab = r"$t$"
    ylab = r'$T_{em}$ (Newton-meters)'
    title = 'Predicted (SINDy) vs Torque (Clarke) on test set V = ' + str(TESTDATA['V'])
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

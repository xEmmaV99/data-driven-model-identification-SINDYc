from prepare_data import *


def simulate_UMP():
    """
    Simulation for the UMP
    """
    cwd = os.getcwd()
    path_to_data_files = os.path.join(cwd, 'data/Combined/07-20-ecc-x50-y0')
    print(path_to_data_files)
    path_to_test_file = None
    UMP_train, x_train, u_train, UMP_val, x_val, u_val, TESTDATA = prepare_data(path_to_data_files,
                                                                                UMP=True,
                                                                                path_to_test_file=path_to_test_file,
                                                                                t_end=1.0, number_of_trainfiles=40)
    inputs_per_library = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14], [12]]
    big_lib = ps.GeneralizedLibrary([ps.PolynomialLibrary(degree=3, include_interaction=True),
                                     ps.FourierLibrary(n_frequencies=1, include_cos=True, include_sin=True)],
                                    tensor_array=None,  # don't merge the libraries
                                    inputs_per_library=inputs_per_library)  # are crossterms needed?
    number_of_models = 10
    if number_of_models >= 2:
        model = theshold_search(np.linspace(0.01, 0.1, number_of_models),
                                train_and_validation_data=[UMP_train, x_train, u_train, UMP_val, x_val, u_val],
                                method="lasso", name="UMP_opt", plot_now=True, library=big_lib)
    else:
        threshold = 0.0
        #optimizer = ps.SR3(thresholder="l1", threshold=threshold)
        optimizer = Lasso(alpha=threshold, fit_intercept=False)
        #library = ps.PolynomialLibrary(degree=2, include_interaction=True)
        library = big_lib
        model = ps.SINDy(optimizer=optimizer, feature_library=library,
                         feature_names=["i_d", "i_q", "i_0"] + TESTDATA['u_names'])
        model.fit(x_train, u=u_train, t=None, x_dot=UMP_train)

    model.print()

    # TESTDATA
    UMP_test = TESTDATA['UMP']
    x_test = TESTDATA['x']
    u_test = TESTDATA['u']
    t = TESTDATA['t']
    UMP_test_predicted = model.predict(x_test, u_test)

    # save the plots
    xydata = [np.hstack((t, UMP_test_predicted[:, 0].reshape(len(t), 1))),
              np.hstack((t, UMP_test_predicted[:, 1].reshape(len(t), 1))), np.hstack((t, UMP_test))]
    xlab = r"$t$"
    ylab = r'$UMP$ (Newton)'
    title = 'Predicted (SINDy) vs simulated UMP on test set V = ' + str(TESTDATA['V'])
    legend = [r"Predicted UMP_x", r"Predicted UMP_y", r"Reference"]
    specs = ["b", "r", "k--"]
    save_plot_data("UMP_lasso_ecc_load", xydata, title, xlab, ylab, legend=legend, plot_now=True, specs=specs)
    return
import matplotlib.pyplot as plt
import sklearn.linear_model

from prepare_data import *
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso

cwd = os.getcwd()
path_to_data_files = os.path.join(cwd, 'data\\Combined\\07-20-ecc-x0-y50')


def simulate_currents():
    """
    Simulation for the currents
    """
    path_to_test_file = None
    do_time_simulation = False
    # get the data
    xdot_train, x_train, u_train, xdot_val, x_val, u_val, TESTDATA = prepare_data(path_to_data_files,
                                                                                  path_to_test_file=path_to_test_file,
                                                                                  t_end=1.0, number_of_trainfiles=20)
    print(TESTDATA['u_names'])
    inputs_per_library = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14], [12]]

    big_lib = ps.GeneralizedLibrary([ps.PolynomialLibrary(degree=3, include_interaction=True),
                                     ps.FourierLibrary(n_frequencies=1, include_cos=True, include_sin=True)],
                                    tensor_array=None,  # don't merge the libraries
                                    inputs_per_library=inputs_per_library) # are crossterms needed?


    # Fit the model
    number_of_models = 10
    # opt was 0.0001 but 10 gave reasonable results
    if number_of_models == 1:
        threshold = 5.0
        # optimizer = ps.SR3(thresholder="l1", threshold=threshold)
        optimizer = Lasso(alpha=threshold, fit_intercept=False)

        #print(TESTDATA['u_names'])

        #library = ps.PolynomialLibrary(degree=3, include_interaction=True)  # ps.FourierLibrary(n_frequencies=5)
        library = big_lib
        model = ps.SINDy(optimizer=optimizer, feature_library=library, feature_names=['i_d', 'i_q', 'i_0'] + TESTDATA['u_names'])
        print("Fitting model")
        model.fit(x_train, u=u_train, t=None, x_dot=xdot_train)
        model.print()
        plot_coefs2(model)
        plt.figure()
    else:
        # make models and calucalte RMSE and sparisty
        threshold_grid = np.logspace(-4, 2, number_of_models)
        # threshold_grid = np.linspace(0.01, 0.1, number_of_models)
        model = theshold_search(threshold_grid, train_and_validation_data=[xdot_train, x_train, u_train, xdot_val, x_val, u_val], library = big_lib, method="lasso", name="main2")

    # CHECK THE BEST MODEL ON THE TESTDATA
    xdot_test = TESTDATA['xdot']
    x_test = TESTDATA['x']
    u_test = TESTDATA['u']
    t = TESTDATA['t']

    # Predict derivatives using the learned model
    x_dot_test_predicted = model.predict(x_test, u_test)

    xydata = [np.hstack((t, x_dot_test_predicted)), np.hstack((t, xdot_test))]
    xlab = r"$t$"
    ylab = r'$\dot{x}$'
    title = 'Predicted vs computed derivatives on test set V = ' + str(TESTDATA['V'])
    leg = [r"$\partial_t{i_d}$", r"$\partial_t{i_q}$", r"$\partial_t{i_0}$", "computed"]
    specs = [None, "k--"]
    save_plot_data("currents", xydata, title, xlab, ylab, legend=leg, plot_now=True,
                   specs=specs)  # , sindy_model=model)

    if do_time_simulation:
        simulation_time = 1.0
        plt.figure()
        print("Starting simulation")
        t_value = t[t < simulation_time]

        x_sim = model.simulate(x_test[0, :],
                               u=u_test[(t < simulation_time).reshape(-1), :],
                               t=t_value.reshape(t_value.shape[0]),
                               integrator_kws={'method': 'RK23'})

        print("Finished simulation")


        xy_data = []
        save_plot_data("currents_simulation",
                       [np.hstack((t_value[:-1].reshape(len(t_value)-1,1), x_sim)), np.hstack((t.reshape(len(t_value),1), x_test))],
                       "Simulated currents on test set V = " + str(TESTDATA['V']),
                       r"$t$", r"$x$", plot_now=True, specs = [None, "k--"],
                       legend=[r"$i_d$", r"$i_q$", r"$i_0$", "test data"])

        '''plt.plot(t_value[:-1], x_sim)
        plt.plot(t, x_test, 'k--')  # plot the test data
        plt.legend([r"$i_d$", r"$i_q$", r"$i_0$", "test data"])
        plt.title('Simulated vs test currents on test set V = ' + str(TESTDATA['V']))
        plt.xlabel(r"$t$")
        plt.ylabel(r"$x$")'''
    plt.show()
    return


def simulate_torque():
    """
    Simulation for the Torque
    """
    path_to_test_file = None
    T_train, x_train, u_train, T_val, x_val, u_val, TESTDATA = prepare_data(path_to_data_files,
                                                                            Torque=True,
                                                                            path_to_test_file=path_to_test_file,
                                                                            t_end=1.0, number_of_trainfiles=20)
    fit_multiple = True
    if fit_multiple:
        model = theshold_search(np.logspace(-12, -3, 15),
                                train_and_validation_data=[T_train, x_train, u_train, T_val, x_val, u_val],
                                method="lasso", name="torque_opt", plot_now=True)

    else:
        threshold = 0.0
        #optimizer = ps.SR3(thresholder="l1", threshold=threshold) # this runs, only reasonable for threshold = 0
        optimizer = Lasso(alpha=threshold,
                          fit_intercept=False)  # now, SR3 really is a lot better, it is weird that adding things to the library makes this worse too

        library = ps.PolynomialLibrary(degree=3, include_interaction=True)

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

    number_of_models = 1
    if number_of_models > 1:
        model = theshold_search(np.linspace(0.01, 0.1, number_of_models),
                                train_and_validation_data=[UMP_train, x_train, u_train, UMP_val, x_val, u_val],
                                method="lasso", name="UMP_opt", plot_now=True)
    else:
        threshold = 0.0
        #optimizer = ps.SR3(thresholder="l1", threshold=threshold)
        optimizer = Lasso(alpha=threshold, fit_intercept=False)
        library = ps.PolynomialLibrary(degree=2, include_interaction=True)

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


def plot_everything(path_to_directory):
    files = os.listdir(path_to_directory)
    for file in files:
        if file.endswith('.pkl'):
            path = os.path.join(path_to_directory, file)
            plot_data(path, show=False)
    plt.show()
    return

'''
simulate_currents()
simulate_torque()
simulate_UMP() '''

# plot_everything('C:\\Users\\emmav\\PycharmProjects\\SINDY_project\\plots\\')
# simulate_torque()
simulate_currents()
# simulate_UMP()
# simulate_currents()
# todo add non linear to immec model (and try to solve that with sindy)
# todo add static ecc
# todo add dynamic ecc

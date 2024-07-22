import matplotlib.pyplot as plt
import sklearn.linear_model

from prepare_data import *
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso

cwd = os.getcwd()
path_to_data_files = os.path.join(cwd, 'data/Combined/07-20-load')


def simulate_currents():
    """
    Simulation for the currents
    """
    path_to_test_file = None  # os.path.join(cwd, 'data/07-19/IMMEC_history_long200V_2.0sec.pkl')  # None -> test is chosen from data
    do_time_simulation = False
    # get the data
    xdot_train, x_train, u_train, xdot_val, x_val, u_val, TESTDATA = prepare_data(path_to_data_files,
                                                                                  path_to_test_file=path_to_test_file,
                                                                                  t_end=2.5)
    # Fit the model
    # threshold = 0.041 ?
    # using LASSO - 0.07
    number_of_models = 1
    if number_of_models == 1:
        threshold = 0.07
        #optimizer = ps.SR3(thresholder="l1", threshold=threshold)
        optimizer = Lasso(alpha=threshold)
        # print("Lasso optimizer")

        library = ps.PolynomialLibrary(degree=2, include_interaction=True)  # ps.FourierLibrary(n_frequencies=5)
        model = ps.SINDy(optimizer=optimizer, feature_library=library,
                         feature_names=['i_d', 'i_q', 'i_0'] + TESTDATA['u_names'])
        model.fit(x_train, u=u_train, t=None, x_dot=xdot_train)
        model.print()
        plot_coefs(model.coefficients(), featurenames=model.get_feature_names())
        plt.figure()
    else:
        # make models and calucalte RMSE and sparisty
        # threshold_grid = np.logspace(-8, 1, number_of_models)
        threshold_grid = np.linspace(0.01, 0.08, number_of_models)
        variable = {'threshold': threshold_grid,
                    'MSE': np.zeros(number_of_models),
                    'SPAR': np.zeros(number_of_models),
                    'model': list(np.zeros(number_of_models))}
        for i, threshold in enumerate(threshold_grid):
            print(i)
            # optimizer = ps.SR3(thresholder="l1", threshold=threshold)
            optimizer = Lasso(alpha=threshold,
                              fit_intercept=False)

            library = ps.PolynomialLibrary(degree=2, include_interaction=True)
            model = ps.SINDy(optimizer=optimizer, feature_library=library)
            model.fit(x_train, u=u_train, t=None, x_dot=xdot_train)  # fit on training data

            variable['MSE'][i] = mean_squared_error(model.predict(x_val, u_val), xdot_val)  # validate
            variable['SPAR'][i] = np.count_nonzero(model.coefficients())  # number of non-zero elements
            variable['model'][i] = model

        rel_sparsity = variable['SPAR'] / np.max(variable['SPAR'])

        # plot the results
        fig, ax1 = plt.subplots()
        ax1.set_xlabel(r'Threshold $\lambda$')
        ax1.set_ylabel('MSE', color='r')
        ax1.plot(variable['threshold'], variable['MSE'], color='r')
        ax2 = ax1.twinx()

        ax2.set_ylabel('Relative sparsity', color='b')
        ax2.plot(variable['threshold'], rel_sparsity, color='b')

        fig.tight_layout()
        plt.show()

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
    specs = [None,"k--"]
    save_plot_data("currents", xydata, title, xlab, ylab, legend=leg, plot_now=True, specs=specs)

    ''' 
    plt.xlabel(r"$t$")
    plt.ylabel(r'$\dot{x}$')
    plt.plot(t, x_dot_test_predicted)
    plt.plot(t, xdot_test, 'k--')
    plt.legend([r"$\partial_t{i_d}$", r"$\partial_t{i_q}$", r"$\partial_t{i_0}$", "computed"])
    plt.title('Predicted vs computed derivatives on test set V = ' + str(TESTDATA['V']))
    '''

    if do_time_simulation:
        simulation_time = 2.0
        plt.figure()
        print("Starting simulation")
        t_value = t[t < simulation_time]

        x_sim = model.simulate(x_test[0, :],
                               u=u_test[(t < simulation_time).reshape(-1), :],
                               t=t_value.reshape(t_value.shape[0]),
                               integrator_kws={'method': 'RK23'})

        print("Finished simulation")

        plt.plot(t_value[:-1], x_sim)
        plt.plot(t, x_test, 'k--')  # plot the test data
        plt.legend([r"$i_d$", r"$i_q$", r"$i_0$", "test data"])
        plt.title('Simulated vs test currents on test set V = ' + str(TESTDATA['V']))
        plt.xlabel(r"$t$")
        plt.ylabel(r"$x$")
    plt.show()
    return


def simulate_torque():
    """
    Simulation for the Torque
    """

    # path_to_test_file = os.path.join(cwd,'data/07-19/IMMEC_history_torque-200V_1.0sec.pkl')  # None if testfile should be gathered from the data folder
    path_to_test_file = None
    T_train, x_train, u_train, T_val, x_val, u_val, TESTDATA = prepare_data(path_to_data_files,
                                                                            Torque=True,
                                                                            path_to_test_file=path_to_test_file,
                                                                            t_end=2.5)

    threshold = 1e-4
    optimizer = ps.SR3(thresholder="l1", threshold=threshold)
    library = ps.PolynomialLibrary(degree=2, include_interaction=True)

    model = ps.SINDy(optimizer=optimizer, feature_library=library)
    model.fit(x_train, u=u_train, t=None, x_dot=T_train)

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

    # plot the results
    plt.xlabel(r"$t$")
    plt.ylabel(r"$T_{em}$ (Newton-meters)")
    plt.plot(t, T_test_predicted)
    plt.plot(t, T_test, 'k--')
    plt.legend(["Predicted", "Reference"])
    plt.title('Predicted vs test Torque on test set V = ' + str(TESTDATA['V']))

    plt.plot(t, Torque, 'r--')
    plt.legend([r"Predicted by SINDYc", r"Reference", r"Simplified Torque model"])

    plt.figure()  # dont plot first point
    plt.semilogy(t[1:], np.abs(T_test_predicted - T_test)[1:])
    plt.semilogy(t[1:], np.abs(Torque.reshape(T_test.shape) - T_test)[1:], 'r--')
    plt.legend([r"Predicted by SINDYc", r"Simplified Torque model"])
    plt.title(r'Absolute error compared to Reference')
    plt.xlabel(r"$t$")
    plt.ylabel(r"$\log_{10}|T_{em} - T_{em,ref}|$")
    plt.show()
    return


simulate_currents()
# todo add non linear to immec model (and try to solve that with sindy)
# todo add static ecc
# todo add dynamic ecc

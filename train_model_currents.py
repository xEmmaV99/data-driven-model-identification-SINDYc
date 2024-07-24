import os

from prepare_data import *
from sklearn.linear_model import Lasso

cwd = os.getcwd()
path_to_data_files = os.path.join(cwd, 'data\\07-23-load-5e-5')


def simulate_currents():
    """
    Simulation for the currents
    """
    #path_to_test_file ='C:\\Users\\emmav\\PycharmProjects\\SINDY_project\\data\\Combined\\07-20-load\\IMMEC_history_207V_2.5sec.pkl'
    path_to_test_file = None
    do_time_simulation = False
    # get the data
    xdot_train, x_train, u_train, xdot_val, x_val, u_val, TESTDATA = prepare_data(path_to_data_files,
                                                                                  path_to_test_file=path_to_test_file,
                                                                                  t_end=2.5, number_of_trainfiles=60,
                                                                                  normalize_input = False)
    print(TESTDATA['u_names'])
    '''
    #inputs_per_library = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [12]]
    big_lib = ps.GeneralizedLibrary([ps.PolynomialLibrary(degree=3, include_interaction=True),
                                     ps.FourierLibrary(n_frequencies=1, include_cos=True, include_sin=True)],
                                    tensor_array=None,  # don't merge the libraries
                                    inputs_per_library=inputs_per_library) # are crossterms needed?
    '''

    # Fit the model
    number_of_models = 1
    # opt was 0.0001 but 10 gave reasonable results
    if number_of_models == 1:
        alpha = 0.1
        optimizer = ps.SR3(thresholder="l1", threshold=alpha, normalize_columns=True)
        #optimizer = Lasso(alpha=alpha, fit_intercept=False)


        library = ps.PolynomialLibrary(degree=2, include_interaction=True)  # ps.FourierLibrary(n_frequencies=5)

        model = ps.SINDy(optimizer=optimizer, feature_library=library, feature_names=['i_d', 'i_q', 'i_0'] + TESTDATA['u_names'])
        print("Fitting model")
        model.fit(x_train, u=u_train, t=None, x_dot=xdot_train)
        model.print()
        print(model.score(x_val, t=None, x_dot = xdot_val, u=u_val, metric = mean_squared_error))
        #plot_coefs2(model)
        #plt.figure()
    else:
        # make models and calucalte RMSE and sparisty
        alpha_grid = np.logspace(-4, 2, number_of_models)
        # alpha_grid = np.linspace(0.01, 0.1, number_of_models)
        model = parameter_search(alpha_grid, train_and_validation_data=[xdot_train, x_train, u_train, xdot_val, x_val, u_val], library = library, method="lasso", name="main2")

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
                   specs=specs)

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


def plot_everything(path_to_directory):
    files = os.listdir(path_to_directory)
    for file in files:
        if file.endswith('.pkl'):
            path = os.path.join(path_to_directory, file)
            plot_data(path, show=False)
    plt.show()
    return

if __name__ == "__main__":
    path_to_data_files = os.path.join(os.getcwd(), "data\\07-24-default-5e-5")
    optimise_simulation(path_to_data_files, nmbr_models=20, loglwrbnd = [-12,12], logupbnd = [0,0])
    simulate(path)

# now, Training without Load but test WITH load
# todo add non linear to immec model (and try to solve that with sindy)
# todo add static ecc
# todo add dynamic ecc

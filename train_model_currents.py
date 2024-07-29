import os
from tqdm import tqdm
from prepare_data_old_data import *
from libs import *

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

    xdot_train, x_train, u_train, xdot_val, x_val, u_val, _ = prepare_data(path_to_data_files,
                                                                           t_end=1.0,
                                                                           number_of_trainfiles=40)
    library = get_custom_library_funcs("exp")
    print("using custom library: ", library)
    print("SR3_L1 optimisation")

    parameter_search_2D(np.linspace(0.5,1.5, nmbr_models),
                         np.linspace(1.5, 2.5, nmbr_models),
                         train_and_validation_data=[xdot_train, x_train, u_train, xdot_val, x_val, u_val],
                         name="currents_sr3", plot_now=False)

    '''    parameter_search(np.logspace(loglwrbnd[0], loguprbnd[0], nmbr_models),
                     train_and_validation_data=[xdot_train, x_train, u_train, xdot_val, x_val, u_val],
                     method="SR3_L1", name="currents_sr3", plot_now=False, library=library)
    '''

    #print("Lasso optimisation")
    #parameter_search(np.logspace(loglwrbnd[1], loguprbnd[1], nmbr_models),
    #                 train_and_validation_data=[xdot_train, x_train, u_train, xdot_val, x_val, u_val],
    #                 method="Lasso", name="currents_lasso", plot_now=False, library=library)
    #path = os.path.join(os.getcwd(), "plot_data")
    #for p in ["\\currents_sr3", "\\currents_lasso"]:
    #    plot_data(path+p+'.pkl')
    return


def simulate_currents(path_to_data_files, alpha, optimizer='sr3', path_to_test_file=None, do_time_simulation=False):
    """
    Simulation for the currents and compared with testdata
    :param path_to_data_files:
    :param alpha:
    :param optimizer:
    :param path_to_test_file:
    :param do_time_simulation:
    :return:
    """
    xdot_train, x_train, u_train, xdot_val, x_val, u_val, testdata = prepare_data(path_to_data_files,
                                                                                  path_to_test_file=path_to_test_file,
                                                                                  t_end=1.0, number_of_trainfiles=60)

    #library = ps.PolynomialLibrary(degree=2, interaction_only=True)
    #library = get_custom_library_funcs("best")
    library = ps.PolynomialLibrary(degree=2, include_interaction=True)

    if optimizer == 'sr3':
        opt = ps.SR3(thresholder="l1", threshold=np.sqrt(2*2*0.001), nu = 0.001)
        print("SR3_L1 optimisation")
    elif optimizer == 'lasso':
        opt = Lasso(alpha=alpha, fit_intercept=False)
        print("Lasso optimisation")
    else:
        raise ValueError("Unknown optimizer")

    model = ps.SINDy(optimizer=opt, feature_library=library,
                     feature_names=testdata['feature_names'])
    print("Fitting model")
    model.fit(x_train, u=u_train, t=None, x_dot=xdot_train)
    model.print()


    print("MSE: "+ str(model.score(x_val, t=None, x_dot=xdot_val, u=u_val, metric=mean_squared_error)))
    plot_coefs2(model, log = True)

    save_model(model, "currents_model")

    # Validate the best model, on testdata
    xdot_test = testdata['xdot']
    x_test = testdata['x']
    u_test = testdata['u']
    t = testdata['t']

    # Predict derivatives using the learned model
    x_dot_test_predicted = model.predict(x_test, u_test)

    xydata = [np.hstack((t, x_dot_test_predicted)), np.hstack((t, xdot_test))]
    xlab = r"$t$"
    ylab = r'$\dot{x}$'
    title = 'Predicted vs computed derivatives on test set V = ' + str(testdata['V'])
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
                       "Simulated currents on test set V = " + str(testdata['V']),
                       r"$t$", r"$x$", plot_now=True, specs=[None, "k--"],
                       legend=[r"$i_d$", r"$i_q$", r"$i_0$", "test data"])

    plt.show()
    return


if __name__ == "__main__":

    path_to_data_files = os.path.join(os.getcwd(), 'data\\07-24-default-5e-5')

    ### OPTIMIZE ALPHA
    #optimize_currents_simulation(path_to_data_files, nmbr_models=3, loglwrbnd=[-7, -7], loguprbnd=[3, 3])

    ### PLOT MSE FOR DIFFERENT ALPHA
    #plot_data([os.getcwd()+"\\plot_data"+p+".pkl" for p in ["\\currents_sr3", "\\currents_lasso"]], show = False, limits=[[1e0,5e2], [0,150]])
    #plot_data(os.getcwd()+'\\plot_data\\currents_sr3.pkl', show = True)

    ### SIMULATE CURRENTS
    simulate_currents(path_to_data_files, alpha=1e-1, optimizer='lasso', do_time_simulation=False)
    #plt.show()

    # TEST WITH NEW DATAFILES
    #path_to_data_files = os.path.join(os.getcwd(), 'train-data','07-25','IMMEC_0ecc_1.0sec.npz')
    #simulate_currents(path_to_data_files, alpha=1e1, optimizer='lasso', do_time_simulation=False)
    plt.show()



# todo add non linear to immec model (and try to solve that with sindy)
# todo add static ecc
# todo add dynamic ecc

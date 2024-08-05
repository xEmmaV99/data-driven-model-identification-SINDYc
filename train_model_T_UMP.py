from source import *
import os
from prepare_data import prepare_data
from optimize_parameters import parameter_search, optimize_parameters, plot_optuna_data
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

# Can optionally be merged with currents, only difference is xdot and plots

def optimize_simulation(path_to_data_files, nmbr_models=-1, loglwrbnd=None, loguprbnd=None, Torque=False, UMP=False):
    """
    OLD TO BE REMOVED

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

    DATA = prepare_data(path_to_data_files, number_of_trainfiles=40)

    lib = "best"
    library = get_custom_library_funcs(lib)

    if Torque:
        train = DATA['T_em_train']
        val = DATA['T_em_val']
        name = 'torque'
    elif UMP:
        train = DATA['UMP_train']
        val = DATA['UMP_val']
        name = 'ump'
    else: # fit both UMP and Torque with one sindy model
        train = np.hstack((DATA['T_em_train'], DATA['UMP_train']))
        val = np.hstack((DATA['T_em_val'], DATA['UMP_val']))
        name = 'umptorque'

    #todo smac3
    print("SR3_L1 optimisation")
    parameter_search(np.logspace(loglwrbnd[0], loguprbnd[0], nmbr_models),
                     train_and_validation_data=[train, DATA['x_train'], DATA['u_train'], val, DATA['x_val'],
                                                DATA['u_val']],
                     method="SR3_L1", name=name +"_sr3", plot_now=False, library=library)

    print("Lasso optimisation")
    parameter_search(np.logspace(loglwrbnd[1], loguprbnd[1], nmbr_models),
                     train_and_validation_data=[train, DATA['x_train'], DATA['u_train'], val, DATA['x_val'],
                                                DATA['u_val']],
                     method="lasso", name=name+"_lasso", plot_now=False, library=library)
    return


def make_model(path_to_data_files, optimizer, nmbr_of_train=-1,
               Torque = False, UMP = False, lib = "",
               alpha=None, nu=None,lamb = None):
    """
    Simulates the Torque on test data, and compares with the Clarke model
    :param path_to_data_files:
    :param alpha:
    :param optimizer:
    :param path_to_test_file:
    :return:
    """
    DATA = prepare_data(path_to_data_files, number_of_trainfiles=nmbr_of_train)

    library = get_custom_library_funcs(lib)

    if Torque:
        train = DATA['T_em_train']
        name = "Torque"
    elif UMP:
        train = DATA['UMP_train']
        name = "UMP"
    else:
        train = np.hstack((DATA['T_em_train'].reshape(-1, 1), DATA['UMP_train']))
        name = "Torque-UMP"

    if optimizer == 'sr3':
        print("SR3_L1 optimisation")
        opt = ps.SR3(thresholder="l1", threshold=lamb, nu=nu)
    elif optimizer == 'lasso':
        print("Lasso optimisation")
        opt = ps.WrappedOptimizer(Lasso(alpha=alpha, fit_intercept=False, precompute=True, max_iter=1000))
    else:
        raise ValueError("Optimizer not known")


    model = ps.SINDy(optimizer=opt, feature_library=library,
                     feature_names=DATA['feature_names'])

    print("Fitting model")
    model.fit(DATA['x_train'], u=DATA['u_train'], t=None, x_dot=train)

    model.print(precision=10)


    print("SPAR", np.count_nonzero(model.coefficients()))
    #plot_coefs2(model, show = False, log = True)

    save_model(model, name+"_model", lib)


def simulate(model_name, path_to_test_file, Torque = False, UMP = False):
    model = load_model(model_name)
    model.print()
    plot_coefs2(model ,show=True) #debgu!!
    TEST = prepare_data(path_to_test_file, test_data=True)

    # testdata
    if Torque:
        test_values = TEST['T_em'].reshape(-1,1)
    elif UMP:
        test_values = TEST['UMP']
    else:
        test_values = np.hstack((TEST['T_em'].reshape(-1, 1), TEST['UMP']))

    x_test = TEST['x']
    u_test = TEST['u']
    t = TEST['t'][:, :, 0]  # assume the t is the same for all simulations
    test_predicted = model.predict(x_test, u_test)

    print("MSE: ", mean_squared_error(test_values, test_predicted))
    # Compare with alternative approach for Torque calculation on testdata
    V = u_test[:, 6:9]
    I = u_test[:, 3:6]
    x = TEST['x']

    if Torque:
        path_to_data_dir = os.path.dirname(path_to_test_file)
        with open(path_to_data_dir + '/SIMULATION_DATA.pkl', 'rb') as f:
            data = pkl.load(f)
        R = data['R_st']
        print(R)
        lambda_dq0 = (V.T - np.dot(R, I.T)).T
        Torque_value = lambda_dq0[:, 0] * x[:, 1] - lambda_dq0[:, 1] * x[:, 0]  # lambda_d * i_q - lambda_q * i_d


    if not UMP:
        # Save the plots, torque first
        xydata = [np.hstack((t, test_predicted[:, 0].reshape(-1, 1))), np.hstack((t, test_values[:,0].reshape(-1, 1))),
                  np.hstack((t, Torque_value.reshape(-1, 1)))]
        xlab = r"$t$"
        ylab = r'$T_{em}$ (Newton-meters)'
        title = 'Predicted (SINDy) vs Torque (Clarke) on test set V = ' + str(TEST['V'])
        legend = [r"Predicted by SINDYc", r"Reference", r"Simplified Torque model"]
        specs = [None, "k--", "r--"]
        save_plot_data("Torque", xydata, title, xlab, ylab, legend=legend, plot_now=True, specs=specs)

        # Plot the error of torque
        plt.figure()  # dont plot first point
        plt.semilogy(t[1:], np.abs(test_predicted[:, 0] - test_values[:,0])[1:])
        plt.semilogy(t[1:], np.abs(Torque_value.reshape(test_values[:,0].shape) - test_values[:,0])[1:], 'r--')
        plt.legend([r"Predicted by SINDYc", r"Simplified Torque model"])
        plt.title(r'Absolute error compared to Reference')
        plt.xlabel(r"$t$")
        plt.ylabel(r"$\log_{10}|T_{em} - T_{em,ref}|$")

    if not Torque:
        # Save the plots for UMP
        xydata = [np.hstack((t, test_predicted[:, 0].reshape(len(t), 1))),
                  np.hstack((t, test_predicted[:, 1].reshape(len(t), 1))), np.hstack((t, test_values[:,-2:]))]
        xlab = r"$t$"
        ylab = r'$UMP$ (Newton)'
        title = 'Predicted (SINDy) vs simulated UMP on test set V = ' + str(TEST['V'])
        legend = [r"Predicted UMP_x", r"Predicted UMP_y", r"Reference"]
        specs = ["b", "r", "k--"]
        save_plot_data("UMP", xydata, title, xlab, ylab, legend=legend, plot_now=True, specs=specs)

    plt.show()
    return


if __name__ == "__main__":
    path_to_data_files = os.path.join(os.getcwd(), "train-data", "07-29-default", "IMMEC_0ecc_5.0sec.npz")
    path_to_test_file = os.path.join(os.getcwd(), "test-data", "07-29-default", "IMMEC_0ecc_5.0sec.npz")
    path_to_data_files = os.path.join(os.getcwd(), "train-data", "07-31-ecc-50", "IMMEC_50ecc_5.0sec.npz")
    path_to_data_files = os.path.join(os.getcwd(), "train-data", "07-31-nonlin50", "IMMEC_nonlinear-50ecc_5.0sec.npz")
    path_to_test_file = os.path.join(os.getcwd(), "test-data", "07-31-50ecc-load", "IMMEC_50ecc_5.0sec.npz")
    #path_to_test_file = os.path.join(os.getcwd(), "test-data", "08-02", "IMMEC_xy50ecc_5.0sec.npz")
    #path_to_test_file = os.path.join(os.getcwd(), "test-data", "08-02", "IMMEC_y50ecc_5.0sec.npz")

    optimize = False
    plot_pareto = False
    create_model = False
    simulation = True

    if optimize:
        ### OPTIMISE ALPHA FOR TORQUE SIMULATION
        optimize_parameters(path_to_data_files, mode = 'torque')
        optimize_parameters(path_to_data_files, mode = 'ump')


    ### PLOT MSE AND SPARSITY FOR TORQUE AND UMP SIMULATION
    if plot_pareto:
        plot_optuna_data('torque-lasso-study')
        plot_optuna_data('UMP-lasso-study')

    ### MAKE A MODEL
    if create_model:
        make_model(path_to_data_files, alpha =36 ,lamb=4.15e-5, nu=3.4e-11, optimizer="sr3", nmbr_of_train=-1, lib = "poly_2nd_order",
                   Torque = False, UMP=True)


    #make_model(path_to_data_files, alpha=1e-5, optimizer="lasso", nmbr_of_train=20)
    #make_model(path_to_data_files, alpha = 1e-1, optimizer="sr3", nmbr_of_train=30)
    #make_model(path_to_data_files, alpha = .1, optimizer="lasso", nmbr_of_train=50, Torque=True, lib = "poly_2nd_order")
    #make_model(path_to_data_files, alpha = 8.14, optimizer="lasso", nmbr_of_train=50, Torque=True, UMP=False, lib = 'poly_2nd_order')

    #make_model(path_to_data_files, nu = 1.03e-6, lamb = 0.00091, optimizer="sr3", nmbr_of_train=50, Torque=True, UMP=False, lib = 'poly_2nd_order')
    #make_model(path_to_data_files, nu=3.45e-10, lamb=1.197e-5,
    #           optimizer="sr3", nmbr_of_train=50, Torque=True, UMP=False,
    #           lib='poly_2nd_order')
    #make_model(path_to_data_files, nu=1.0449e-9, lamb=1.47e-5,
    #           optimizer="sr3", nmbr_of_train=50, Torque=True, UMP=False,
    #           lib='poly_2nd_order')

    if simulation:
        ### SIMULATE TORQUE WITH CHOSEN ALPHA AND OPTIMIZER
        #simulate("Torque-UMP_model", path_to_test_file, Torque=True)
        simulate("torque_model", path_to_test_file, Torque=True)
from source import *
import os
from param_optimizer import optimize_parameters
import matplotlib.pyplot as plt
from prepare_data import prepare_data
from sklearn.linear_model import Lasso


# Can optionally be merged with currents, only difference is xdot and plots

def optimize_simulation(path_to_data_files, nmbr_models=-1, loglwrbnd=None, loguprbnd=None, Torque=False, UMP=False):
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



def make_model(path_to_data_files, alpha, optimizer, nmbr_of_train=-1, Torque = False, UMP = False):
    """
    Simulates the Torque on test data, and compares with the Clarke model
    :param path_to_data_files:
    :param alpha:
    :param optimizer:
    :param path_to_test_file:
    :return:
    """
    DATA = prepare_data(path_to_data_files, number_of_trainfiles=nmbr_of_train)

    lib = 'best'
    library = get_custom_library_funcs(lib)

    if Torque:
        train = DATA['T_em_train']
        name = "Torque"
    elif UMP:
        train = DATA['UMP_Train']
        name = "UMP"
    else:
        train = np.hstack((DATA['T_em_train'].reshape(-1, 1), DATA['UMP_train']))
        name = "Torque-UMP"

    if optimizer == 'sr3':
        print("SR3_L1 optimisation")
        opt = ps.SR3(thresholder="l1", threshold=alpha)
    elif optimizer == 'lasso':
        print("Lasso optimisation")
        opt = Lasso(alpha=alpha, fit_intercept=False)
    else:
        raise ValueError("Optimizer not known")

    model = ps.SINDy(optimizer=opt, feature_library=library,
                     feature_names=DATA['feature_names'])

    print("Fitting model")
    model.fit(DATA['x_train'], u=DATA['u_train'], t=None, x_dot=train)

    if model.coefficients().ndim == 1:  # fix dimensions of this matrix, bug in pysindy, o this works
        model.optimizer.coef_ = model.coefficients().reshape(1, model.coefficients().shape[0])
    model.print()

    #plot_coefs2(model, show = False, log = True)

    save_model(model, name+"_model", lib)


def simulate(model_name, path_to_test_file, Torque = False, UMP = False):
    model = load_model(model_name)
    model.print()
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

    # Compare with alternative approach for Torque calculation on testdata
    V = u_test[:, 6:9]
    I = u_test[:, 3:6]
    x = TEST['x']

    print('hardcoded')
    path_to_data_dir = os.path.join(os.getcwd(), 'train-data', '07-29-default')
    with open(path_to_data_dir + '/SIMULATION_DATA.pkl', 'rb') as f:
        data = pkl.load(f)
    R = data['R_st']
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
        xydata = [np.hstack((t, test_predicted[:, 1].reshape(len(t), 1))),
                  np.hstack((t, test_predicted[:, 2].reshape(len(t), 1))), np.hstack((t, test_values[:,-2:]))]
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

    ### OPTIMISE ALPHA FOR TORQUE SIMULATION
    # optimise_simulation(path_to_data_files, nmbr_models='all', loglwrbnd=[-12, -12], loguprbnd=[0, 0])

    ### PLOT MSE FOR TORQUE SIMULATION
    # plot_data([os.getcwd() + "\\plots_2607_presentation" + p + ".pkl" for p in ["\\torque_SR3_", "\\torque_lasso"]],
    # show=False, limits=[[1e-6, 1], [0, 100]])

    ### MAKE A MODEL
    #make_model(path_to_data_files, alpha=1e-5, optimizer="lasso", nmbr_of_train=20)
    #make_model(path_to_data_files, alpha = 1e-1, optimizer="sr3", nmbr_of_train=30)
    #make_model(path_to_data_files, alpha = 1e-5, optimizer="lasso", nmbr_of_train=25, Torque=True)

    ### SIMULATE TORQUE WITH CHOSEN ALPHA AND OPTIMIZER
    simulate("Torque-UMP_model", path_to_test_file, Torque=True)
    simulate("torque_model", path_to_test_file, Torque=True)
from source import *
import os
from prepare_data import prepare_data
from optimize_parameters import parameter_search, optimize_parameters, plot_optuna_data
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error



def make_model(path_to_data_files, optimizer, nmbr_of_train=-1, lib = "",
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

    train = DATA["W_mag_train"]
    name = "W_mag"

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


def simulate(model_name, path_to_test_file):
    model = load_model(model_name)
    model.print()
    plot_coefs2(model ,show=True) #debgu!!
    TEST = prepare_data(path_to_test_file, test_data=True)


    test_values = TEST['W_mag']

    x_test = TEST['x']
    u_test = TEST['u']
    t = TEST['t'][:, :, 0]  # assume the t is the same for all simulations
    test_predicted = model.predict(x_test, u_test)

    print("MSE: ", mean_squared_error(test_values, test_predicted))
    # Compare with alternative approach for Torque calculation on testdata
    V = u_test[:, 6:9]
    I = u_test[:, 3:6]
    x = TEST['x']

    # todo: calculate UMP and torque (manually? unsure)
    # T = dW/dgamma and F = dW/dr

    '''
    # Save the plots for UMP
    xydata = [np.hstack((t, test_predicted[:, 0].reshape(len(t), 1))),
              np.hstack((t, test_predicted[:, 1].reshape(len(t), 1))), np.hstack((t, test_values[:,-2:]))]
    xlab = r"$t$"
    ylab = r'$UMP$ (Newton)'
    title = 'Predicted (SINDy) vs simulated UMP on test set V = ' + str(TEST['V'])
    legend = [r"Predicted UMP_x", r"Predicted UMP_y", r"Reference"]
    specs = ["b", "r", "k--"]
    save_plot_data("UMP", xydata, title, xlab, ylab, legend=legend, plot_now=True, specs=specs)'''

    return


if __name__ == "__main__":
    path_to_data_files = os.path.join(os.getcwd(), "train-data", "07-29-default", "IMMEC_0ecc_5.0sec.npz")
    path_to_test_file = os.path.join(os.getcwd(), "test-data", "07-29-default", "IMMEC_0ecc_5.0sec.npz")
    path_to_data_files = os.path.join(os.getcwd(), "train-data", "07-31-ecc-50", "IMMEC_50ecc_5.0sec.npz")
    #path_to_data_files = os.path.join(os.getcwd(), "train-data", "07-31-nonlin50", "IMMEC_nonlinear-50ecc_5.0sec.npz")
    #path_to_test_file = os.path.join(os.getcwd(), "test-data", "07-31-50ecc-load", "IMMEC_50ecc_5.0sec.npz")
    #path_to_test_file = os.path.join(os.getcwd(), "test-data", "08-02", "IMMEC_xy50ecc_5.0sec.npz")
    #path_to_test_file = os.path.join(os.getcwd(), "test-data", "08-02", "IMMEC_y50ecc_5.0sec.npz")
    path_to_test_file = os.path.join(os.getcwd(), "test-data", "07-31-50ecc-load", "IMMEC_50ecc_5.0sec.npz")

    optimize = False
    plot_pareto = False
    create_model = False
    simulation = True

    if optimize:
        ### OPTIMISE ALPHA FOR TORQUE SIMULATION
        optimize_parameters(path_to_data_files)


    ### PLOT MSE AND SPARSITY FOR TORQUE AND UMP SIMULATION
    if plot_pareto:
        plot_optuna_data('W_mag-optuna-study')

    ### MAKE A MODEL
    if create_model:
        make_model(path_to_data_files, alpha =36 ,lamb=4.15e-5, nu=3.4e-11, optimizer="sr3", nmbr_of_train=-1, lib = "poly_2nd_order")

    if simulation:
        simulate("W_model", path_to_test_file)
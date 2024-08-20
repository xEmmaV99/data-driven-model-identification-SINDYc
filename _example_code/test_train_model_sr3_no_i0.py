from prepare_data import *
from libs import *
from optimize_parameters import parameter_search
from source import *

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

    library = ps.PolynomialLibrary(degree=2, include_interaction=True)
    print("SR3_L1 optimisation, full model")
    parameter_search(np.logspace(loglwrbnd[0], loguprbnd[0], nmbr_models),
                     train_and_validation_data=[xdot_train, x_train, u_train, xdot_val, x_val, u_val],
                     method="SR3_L1", name="currents_sr3", plot_now=False, library=library)


    print("SR3_L1 optimisation, only i_d and i_q")
    # only select i_d and i_q
    xdot_train = xdot_train[:, :2]
    xdot_val = xdot_val[:, :2]
    x_train = x_train[:, :2]
    x_val = x_val[:, :2]
    parameter_search(np.logspace(loglwrbnd[0], loguprbnd[0], nmbr_models),
                     train_and_validation_data=[xdot_train, x_train, u_train, xdot_val, x_val, u_val],
                     method="SR3_L1", name="currents_sr3_noi0", plot_now=False, library=library)

    path = os.path.join(os.getcwd(), "plot_data")
    for p in ["\\currents_sr3", "\\currents_sr3_noi0"]:
        plot_data(path+p+'.pkl')
    return

if __name__ == "__main__":
    raise NotImplementedError('This code is outdated, it does not use optuna for hyperparameter selection and uses the old datafiles. '
          'Either remove this testing code or adapt it to the new methods.')

    path_to_data_files = os.path.join(os.getcwd(), 'data\\07-24-default-5e-5')

    ### OPTIMIZE ALPHA
    #optimize_currents_simulation(path_to_data_files, nmbr_models=20, loglwrbnd=[-7, -7], loguprbnd=[3, 3])

    ### PLOT MSE FOR DIFFERENT ALPHA
    for p in ["\\currents_sr3", "\\currents_sr3_noi0"]:
        plot_data(os.getcwd() + "\\plot_data" + p + ".pkl", show=False)
    plt.show()



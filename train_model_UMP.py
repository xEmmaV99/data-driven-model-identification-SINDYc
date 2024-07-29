import os

from prepare_data import *
from libs import *

def optimise_UMP_simulation(path_to_data_files, nmbr_models=20, loglwrbnd=None, loguprbnd=None):
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

    UMP_train, x_train, u_train, UMP_val, x_val, u_val, _ = prepare_data(path_to_data_files,
                                                                         UMP=True,
                                                                         t_end=1.0,
                                                                         number_of_trainfiles=30)
    library = get_custom_library_funcs()
    print("SR3_L1 optimisation")
    parameter_search(np.logspace(loglwrbnd[0], loguprbnd[0], nmbr_models),
                     train_and_validation_data=[UMP_train, x_train, u_train, UMP_val, x_val, u_val],
                     method="SR3_L1", name="UMP_sr3", plot_now=False, library=library)
    print("Lasso optimisation")
    parameter_search(np.logspace(loglwrbnd[1], loguprbnd[1], nmbr_models),
                     train_and_validation_data=[UMP_train, x_train, u_train, UMP_val, x_val, u_val],
                     method="Lasso", name="UMP_lasso", plot_now=False, library=library)
    path = os.path.join(os.getcwd(), "plot_data")
    for p in ["\\UMP_sr3", "\\UMP_lasso"]:
        plot_data(path+p+".pkl")
    return


def simulate_UMP(path_to_data_files, alpha, optimizer = 'sr3', path_to_test_file=None):
    """
    Simulates the UMP using the SINDy model
    :param path_to_data_files:
    :param alpha:
    :param optmizer:
    :param path_to_test_file:
    :return:
    """
    UMP_train, x_train, u_train, UMP_val, x_val, u_val, testdata = prepare_data(path_to_data_files,
                                                                                UMP=True,
                                                                                path_to_test_file=path_to_test_file,
                                                                                t_end=1.0, number_of_trainfiles=30)
    if optimizer == 'sr3':
        opt = ps.SR3(thresholder="l1", threshold=alpha)
    elif optimizer == 'lasso':
        opt = Lasso(alpha=alpha, fit_intercept=False)
    else:
        raise ValueError("Unknown optimizer")

    library = get_custom_library_funcs()
    model = ps.SINDy(optimizer=opt, feature_library=library,
                     feature_names=["i_d", "i_q", "i_0"] + testdata['u_names'])
    model.fit(x_train, u=u_train, t=None, x_dot=UMP_train)
    model.print()
    plot_coefs2(model, show = False, log=True)

    save_model(model, "UMP_model")

    # testdata
    UMP_test = testdata['UMP']
    x_test = testdata['x']
    u_test = testdata['u']
    t = testdata['t']
    UMP_test_predicted = model.predict(x_test, u_test)

    # Save the plots
    xydata = [np.hstack((t, UMP_test_predicted[:, 0].reshape(len(t), 1))),
              np.hstack((t, UMP_test_predicted[:, 1].reshape(len(t), 1))), np.hstack((t, UMP_test))]
    xlab = r"$t$"
    ylab = r'$UMP$ (Newton)'
    title = 'Predicted (SINDy) vs simulated UMP on test set V = ' + str(testdata['V'])
    legend = [r"Predicted UMP_x", r"Predicted UMP_y", r"Reference"]
    specs = ["b", "r", "k--"]
    save_plot_data("UMP_lasso_ecc_load", xydata, title, xlab, ylab, legend=legend, plot_now=True, specs=specs)
    return


if __name__ == "__main__":
    path_to_data_files = os.path.join(os.getcwd(), "data\\Combined\\07-21-ecc-x90-y0")
    path_to_data_files = os.path.join(os.getcwd(), "data\\Combined\\07-20-ecc-x50-y0")
    ### OPTIMSING ALPHA FOR UMP
    #optimise_UMP_simulation(path_to_data_files, nmbr_models=20, loglwrbnd=[-7, -7], loguprbnd=[3, 3])

    ### PLOT MSE AND SPARSITY FOR UMP

    plot_data([os.getcwd()+"\\plots_2607_presentation"+p+".pkl" for p in ["\\UMP_sr3", "\\UMP_lasso"]], show=False, limits=[[1e1,1e3],[0,200]])

    ### SIMULATE UMP WITH SHOWEN METHOD AND ALPHA
    #simulate_UMP(path_to_data_files, alpha=0.05, optimizer='lasso')
    #simulate_UMP(path_to_data_files, alpha=1e1, optimizer='lasso')
    plt.show()

import matplotlib.pyplot as plt
from source import *
from prepare_data import prepare_data
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import datetime


def make_model(
    path_to_data_files: str,
    modeltype: str,
    optimizer: str,
    lib: str = "",
    nmbr_of_train: int = -1,
    alpha: float = None,
    nu: float = None,
    lamb: float = None,
    modelname: str = None,
    threshold: float = 0.1,
    ecc_input = False,
):
    """
    Initialises and fits a SINDy model
        sr3 optimises the function with objective dot(X) = theta(X)*xi as follows (Champion et al. 2020):
                                ||dot(X)- theta(X)*xi ||**2_2 + lamb * L_1(W) + 1/(2nu) * ||xi - W||^2_2

        STLSQ optimises the objective (Brunton et al. 2016) by using hard thresholding on the coefficients:
                                ||dot(X)- theta(X)*xi ||^2_2 + alpha * ||xi||^2_2

        Lasso optimises the objective (see sklearn documentation) as follows:
                                1/(2*n_samples) * ||dot(X)- theta(X)*xi ||^2_2 + alpha * ||xi||_1

    :param path_to_data_files: str, path to the training data
    :param modeltype: either 'torque', 'ump', 'torque-ump', 'currents', or 'wcoe'
    :param optimizer: either 'sr3' or 'lasso'
    :param nmbr_of_train: how many simulations should be considered in the training data, to select 'all', pass -1
    :param lib: name of the library to be used in SINDy (will be passsed to lib.py)
    :param alpha: sparsity weighting factor for the lasso/STLSQ optimisation, should be passed if optimizer is 'lasso' or 'STLSQ'
    :param nu: parameter 1 for sr3 optimisation
    :param lamb: parameter 2 for sr3 optimisation
    :param modelname: name for the model to be saved with. If None, default name is modeltype
    :param threshold: threshold for the STLSQ optimisation
    :return:
    """
    # load in training and validation data
    print("ecc input ", ecc_input)
    DATA = prepare_data(
        path_to_data_files, number_of_trainfiles=nmbr_of_train, ecc_input=ecc_input
    )
    library = get_custom_library_funcs(
        lib, nmbr_input_features=DATA["u"].shape[1] + DATA["x"].shape[1]
    )

    # select inputs for the desired model
    modeltype = modeltype.lower()
    if modeltype == "torque":
        train = DATA["T_em_train"]
        val = DATA["T_em_val"]  # only for MSE calculation
        name = "Torque"
    elif modeltype == "ump":
        train = DATA["UMP_train"]
        val = DATA["UMP_val"]  # only for MSE calculation
        name = "UMP"
    elif modeltype == "torque-ump":
        train = np.hstack((DATA["T_em_train"].reshape(-1, 1), DATA["UMP_train"]))
        val = np.hstack(
            (DATA["T_em_val"].reshape(-1, 1), DATA["UMP_val"])
        )  # only for MSE calculation
        name = "Torque-UMP"
    elif modeltype == "currents":
        train = DATA["xdot_train"]
        val = DATA["xdot_val"]  # only for MSE calculation
        name = "Currents"
    elif modeltype == "wcoe":
        train = DATA["wcoe_train"]
        val = DATA["wcoe_val"]  # only for MSE calculation
        name = "Wcoe"
    else:
        raise ValueError(
            "Model type not equal to 'torque', 'ump', 'torque-ump', 'currents' or 'wcoe' "
        )

    # overwrite saving name if modelname is not None
    if modelname is not None:
        name = modelname

    # Select the correct optimizer
    if optimizer == "sr3":
        print("SR3_L1 optimisation")
        if lamb is None or nu is None:
            raise ValueError(
                "The values lamb and nu should be passed when using sr3 optimisation"
            )
        #DEBUG opt = ps.SR3(thresholder="l1", threshold=lamb, nu=nu)
        opt = ps.SR3(regularizer="l1", reg_weight_lam=lamb, relax_coeff_nu=nu)
    elif optimizer == "lasso":
        print("Lasso optimisation")
        if alpha is None:
            raise ValueError(
                "The value alpha should be passed when using lasso optimisation"
            )
        opt = ps.WrappedOptimizer(
            Lasso(alpha=alpha, fit_intercept=False, precompute=True, max_iter=1000)
        )

    elif optimizer == "STLSQ":
        if alpha is None:
            raise ValueError("Alpha should be provided")
        opt = ps.STLSQ(alpha=alpha, threshold=threshold)

    else:
        raise ValueError(
            "Optimizer not known, only 'sr3', 'STLSQ' or 'lasso' are possible"
        )

    # initialise model
    model = ps.SINDy(
        optimizer=opt, feature_library=library, feature_names=DATA["feature_names"]
    )

    print("Fitting model")
    model.fit(DATA["x_train"], u=DATA["u_train"], t=None, x_dot=train)
    model.print(precision=10)

    print("Non-zero elements: ", np.count_nonzero(model.coefficients()))
    print(
        "MSE: "
        + str(
            model.score(
                DATA["x_val"],
                t=None,
                x_dot=val,
                u=DATA["u_val"],
                metric=mean_squared_error,
            )
        )
    )

    # plot_coefs2(model, show = False, log = True, type = modeltype)

    return save_model(model, name + "_model", lib)


def simulate_model(
    model_name,
    path_to_test_file: str,
    modeltype: str,
    do_time_simulation: bool = False,
    show: bool = True,
    ecc_input=False,
):
    """
    Evaluates a fitted model on a new test dataset
    :param model_name: name of the .pkl file OR the model itself, assumes it is inside the 'models'-directory, or simply uses the passed model
    :param path_to_test_file: path to the data-file
    :param modeltype: either 'torque', 'ump', 'torque-ump', 'currents' or 'wcoe'
    :param do_time_simulation: if the modeltype is 'currents', setting this 'True' will call model.simulate to retrieve the currents (instead of their time derivative)
    :param show: if True, plt.show is called
    :param ecc_input: if True, the eccentricity is added as input variable
    :return: predicted values, test values
    """
    # load in the model and the data
    # if the model is a string, open the .pkl file, if not, just use the model
    if isinstance(model_name, str):
        if model_name.endswith(".pkl"):
            model_name = model_name[:-4]
        model = load_model(
            model_name
        )
    else:
        model = model_name
    model.print()

    print("ecc input: ", ecc_input)
    TEST = prepare_data(path_to_test_file, test_data=True, ecc_input=ecc_input)
    ## plot_coefs2(model, show=True, log=True, type=modeltype)

    # select the corresponding test_values, depending on modeltype
    if modeltype == "torque":
        test_values = TEST["T_em"].reshape(-1, 1)
    elif modeltype == "ump":
        test_values = TEST["UMP"]
    elif modeltype == "torque-ump":
        test_values = np.hstack((TEST["T_em"].reshape(-1, 1), TEST["UMP"]))
    elif modeltype == "currents":
        test_values = TEST["xdot"]
    elif modeltype == "wcoe":
        test_values = TEST["wcoe"]
    else:
        raise ValueError(
            "Model type not equal to 'torque', 'ump', 'torque-ump', 'currents' or 'wcoe' "
        )

    x_test = TEST["x"]
    u_test = TEST["u"]
    t = TEST["t"][:, :, 0]  # assume the t is the same for all simulations
    test_predicted = model.predict(x_test, u_test)

    print("MSE on test: ", mean_squared_error(test_values, test_predicted))
    print("Non-zero elements: ", np.count_nonzero(model.coefficients()))

    # plot the results, according to the modeltype
    if modeltype == "currents":
        # gather the elements for the plot (to be saved)
        xydata = [np.hstack((t, test_predicted)), np.hstack((t, test_values))]
        xlab = r"$t$"
        ylab = r"$\dot{x}$"
        title = "Predicted vs computed derivatives on test set V = " + str(TEST["V"])
        leg = [
            r"$\partial_t{i_d}$",
            r"$\partial_t{i_q}$",
            r"$\partial_t{i_0}$",
            "computed",
        ]
        specs = [None, "k--"]
        # save and plot the result
        save_plot_data(
            "currents",
            xydata,
            title,
            xlab,
            ylab,
            legend=leg,
            plot_now=True,
            specs=specs,
        )

        if do_time_simulation:
            simulation_time = 0.7
            plt.figure()
            print("Starting time simulation, max", simulation_time)
            t_value = t[t < simulation_time]

            # x_sim = model.simulate(x_test[0, :],
            #                       u=u_test[(t < simulation_time).reshape(-1), :],
            #                       t=t_value.reshape(t_value.shape[0]),
            #                       integrator_kws={'method': 'RK45'})

            x_sim = model_simulate(
                x_test[0, :],
                u=u_test[(t < simulation_time).reshape(-1), :],
                t=t_value.reshape(t_value.shape[0]),
                model=model,
            ).y.T
            print("Finished simulation")
            print(x_sim.shape)
            print(x_test[: len(t_value), :].shape)

            print(
                "MSE on simulation: ",
                mean_squared_error(x_sim, x_test[: len(t_value), :]),
            )

            save_plot_data(
                "currents_simulation",
                [
                    np.hstack((t_value[:].reshape(len(t_value), 1), x_sim)),
                    np.hstack((t_value[:].reshape(len(t_value), 1), x_test[: len(t_value), :])),
                ],
                "Simulated currents on test set V = " + str(TEST["V"]),
                r"$t$",
                r"$x$",
                plot_now=True,
                specs=[None, "k--"],
                legend=[r"$i_d$", r"$i_q$", r"$i_0$", "test data"],
            )
            return x_sim, x_test  # returns the simulated values of the currents
        if show:
            plt.show()
        return (
            test_predicted,
            test_values,
        )  # returns the derivative values of the currents

    # Compare with alternative approach for Torque calculation on testdata
    if modeltype == "torque" or modeltype == "torque-ump":
        # Clarke alternative torque calculation
        V = u_test[:, 6:9]
        I = u_test[:, 3:6]
        x = TEST["x"]

        #DEBUG, this uses SIMULATION_DATA.pkl. Could be hardcoded instead (then additional file is not necesary)
        path_to_data_dir = os.path.dirname(path_to_test_file)
        with open(path_to_data_dir + "/SIMULATION_DATA.pkl", "rb") as f:
            data = pkl.load(f)
        R = data["R_st"]
        lambda_dq0 = (V.T - np.dot(R, I.T)).T
        Torque_value = (
            lambda_dq0[:, 0] * x[:, 1] - lambda_dq0[:, 1] * x[:, 0]
        )  # lambda_d * i_q - lambda_q * i_d

        print(
            "MSE simplified model", mean_squared_error(Torque_value, test_values[:, 0])
        )

        """ # tried to add running mean to plot, but this looks bad 
        def running_mean(x, N):
            cumsum = np.cumsum(np.insert(x, 0, 0))
            res = (cumsum[N:] - cumsum[:-N]) / float(N)
            # add nan to start of arra
            res = np.insert(res, 0, np.full(N-1, np.nan))
            return res 
        
        rmean = running_mean(np.array(test_predicted), 200)
        print(rmean.shape)
        plt.plot(rmean)
        plt.plot(test_predicted)
        plt.plot(test_values, 'k--')
        plt.plot(Torque_value, 'r--')
        plt.show()
        """
        # Save the plots, torque first
        xydata = [
            np.hstack((t, test_predicted[:, 0].reshape(-1, 1))),
            np.hstack((t, test_values[:, 0].reshape(-1, 1))),
            np.hstack((t, Torque_value.reshape(-1, 1))),
        ]
        xlab = r"$t$"
        ylab = r"$T_{em}$ (Newton-meters)"
        title = "Predicted (SINDy) vs Torque (Clarke) on test set V = " + str(TEST["V"])
        legend = [r"Predicted by SINDYc", r"Reference", r"Simplified Torque model"]
        specs = [None, "k--", "r--"]
        save_plot_data(
            "Torque",
            xydata,
            title,
            xlab,
            ylab,
            legend=legend,
            plot_now=True,
            specs=specs,
        )

        # Plot the error of torque
        plt.figure()  # dont plot first point
        plt.semilogy(t[1:], np.abs(test_predicted[:, 0] - test_values[:, 0])[1:])
        plt.semilogy(
            t[1:],
            np.abs(Torque_value.reshape(test_values[:, 0].shape) - test_values[:, 0])[
                1:
            ],
            "r--",
        )
        plt.legend([r"Predicted by SINDYc", r"Simplified Torque model"])
        plt.title(r"Absolute error compared to Reference")
        plt.xlabel(r"$t$")
        plt.ylabel(r"$\log_{10}|T_{em} - T_{em,ref}|$")

    if modeltype == "ump" or modeltype == "torque-ump":
        # Save the plots for UMP
        xydata = [
            np.hstack((t, test_predicted[:, 0].reshape(len(t), 1))),
            np.hstack((t, test_predicted[:, 1].reshape(len(t), 1))),
            np.hstack((t, test_values[:, -2:])),
        ]
        xlab = r"$t$"
        ylab = r"$UMP$ (Newton)"
        title = "Predicted (SINDy) vs simulated UMP on test set V = " + str(TEST["V"])
        legend = [r"Predicted UMP_x", r"Predicted UMP_y", r"Reference"]
        specs = ["b", "r", "k--"]
        save_plot_data(
            "UMP", xydata, title, xlab, ylab, legend=legend, plot_now=True, specs=specs
        )

    if modeltype == "wcoe":
        # save and plot the result
        save_plot_data(
            "wcoe",
            [
                np.hstack((t, test_predicted)),
                np.hstack((t, test_values.reshape((-1, 1)))),
            ],
            "Predicted vs Reference on test set V = " + str(TEST["V"]),
            r"$t$",
            r"$W_{coe}$",
            plot_now=True,
            specs=[None, "k--"],
            legend=[r"Predicted", r"Reference"],
        )
    if show:
        plt.show()
    return test_predicted, test_values

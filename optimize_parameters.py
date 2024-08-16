import joblib

from prepare_data import *
import optuna
import multiprocessing
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import pysindy as ps
from libs import get_custom_library_funcs
from libs import get_library_names
import tqdm


def optimize_parameters(path_to_data_files: str, mode: str = 'torque', additional_name: str = "", n_jobs: int = 1,
                        n_trials: int = 100, ecc_input=False):
    """
    This function is used to optimize the parameters of the SINDy model. It uses the Optuna package to select the libaray, optimizer and the hyperparameters of the optimizer.
    :param path_to_data_files: str, path to the data files
    :param mode: what the model should predict, can be 'torque', 'currents', 'ump' or 'wcoe' (merged = currents + I + V is not implemented)
    :param additional_name: str, additional name to add to the study name
    :param n_jobs: number of cores to use
    :param n_trials: number of trials for optuna to run for each job
    :param ecc_input: bool, if True, the eccentricity will be added as input to the model
    :return:
    """

    print("ecc_input =", ecc_input)
    DATA = prepare_data(path_to_data_files, number_of_trainfiles=30, usage_per_trainfile=.50, ecc_input=ecc_input)

    # Select the desired mode
    if mode == "currents":
        XDOT = [DATA['xdot_train'], DATA['xdot_val']]
        namestr = "currents"
    elif mode == "torque":
        XDOT = [DATA['T_em_train'], DATA['T_em_val']]
        namestr = "torque"
    elif mode == "ump":
        XDOT = [DATA['UMP_train'], DATA['UMP_val']]
        namestr = "ump"
    elif mode == "wcoe":
        XDOT = [DATA['wcoe_train'], DATA['wcoe_val']]
        namestr = "W"
    elif mode == "merged":
        # now, the xdot_train contains i and V and I,
        raise NotImplementedError("Merged mode is not fully implemented yet")
    else:
        raise ValueError("mode is either currents, torque or ump")

    a_range = [1e-5, 1e2]
    l_range = [1e-10, 1e2]
    n_range = [1e-12, 1e2]

    _study(namestr+additional_name) # initialize the study
    with joblib.parallel_config(n_jobs=n_jobs, backend="loky", inner_max_num_threads=1):
        joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(optuna_search)(DATA, XDOT, l_range, n_range, a_range, namestr + additional_name, n_trials)
            for _ in range(n_jobs))
    return


def optuna_search(DATA: dict, XDOT: np.array, lminmax: list, nminmax: list, aminmax: list, studyname: str, iter: int):
    """
    This function handles the optuna search for the best parameters for the SINDy model. Uses code from https://optuna.readthedocs.io/en/stable/index.html
    :param DATA: The data dictionary
    :param XDOT: The desired output of the model, assumes first element is training, second is validation
    :param lminmax: lambda range [min, max]
    :param nminmax: nu range [min, max]
    :param aminmax: alpha range [min, max]
    :param studyname: additional name for the study
    :param iter: number of trials for the optuna study
    :return:
    """
    # Set some parameters
    optimizer_list = ['lasso', 'sr3', 'STLSQ']
    library_list = get_library_names()

    def objective(trial):
        lib_choice = trial.suggest_categorical('lib_choice', library_list)
        lib = get_custom_library_funcs(lib_choice, DATA["u"].shape[1] + DATA["x"].shape[1])

        optimizer_name = trial.suggest_categorical('optimizer', optimizer_list)
        if optimizer_name == 'lasso':
            alphas = trial.suggest_float('alphas', aminmax[0], aminmax[1], log=True)
            optimizer = ps.WrappedOptimizer(Lasso(alpha=alphas, fit_intercept=False, precompute=True))

        elif optimizer_name == 'sr3':
            lambdas = trial.suggest_float('lambdas', lminmax[0], lminmax[1], log=True)
            nus = trial.suggest_float('nus', nminmax[0], nminmax[1], log=True)
            optimizer = ps.SR3(thresholder='l1', nu=nus,
                               threshold=lambdas)
        elif optimizer_name == 'stlsq':
            alphas = trial.suggest_float('alphas', aminmax[0], aminmax[1], log=True)
            # alpha is penalising the l2 norm of the coefficients (ridge regression)
            threshold = trial.suggest_float('threshold', 0.001, 1, log=True)
            optimizer = ps.STLSQ(alpha=alphas, threshold=threshold)
        else:
            raise ValueError("Optimizer not known.")

        try: # OOM error handling
            model = ps.SINDy(optimizer=optimizer, feature_library=lib)
            model.fit(DATA["x_train"], u=DATA["u_train"], t=None, x_dot=XDOT[0])
        except Exception as e:
            print("Exception in model fitting:", e)
            raise optuna.TrialPruned()

        MSE = model.score(DATA["x_val"], u=DATA["u_val"], x_dot=XDOT[1],
                          t=None, metric=mean_squared_error)
        SPAR = np.count_nonzero(model.coefficients())

        return MSE, SPAR

    study = _study(studyname)

    study.optimize(objective, n_trials=iter, n_jobs=1)
    return


def _study(studyname):
    study_name = "optuna_studies//" + studyname + "-optuna-study"  # Unique identifier of the study.

    # check if the directory exists
    if not os.path.exists("optuna_studies"):
        os.makedirs("optuna_studies")

    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(directions=['minimize', 'minimize'],  # minimize MSE and sparsity
                                study_name=study_name,
                                storage=storage_name,
                                load_if_exists=True)
    return study


def plot_optuna_data(name):
    """
    Plots the pareto front of the optuna study
    :param name: name of the optuna study, usually ends with '-optuna-study', assuming it is in the optuna_studies directory
    :return:
    """
    print(optuna.study.get_all_study_names(storage="sqlite:///" + "optuna_studies/" + name + ".db"))
    stud = optuna.load_study(study_name=None, storage="sqlite:///" + "optuna_studies/" + name + ".db")
    optuna.visualization.plot_pareto_front(stud, target_names=["MSE", "SPAR"]).show(renderer="browser")
    print(f"Trial count: {len(stud.trials)}")
    return


def parameter_search(parameter_array, train_and_validation_data, method="lasso", name="", plot_now=True, library=None):
    """
    Old implementation of the parameter search, is not used as Optuna does a better job, including multi-parameter search
    :param parameter_array:
    :param train_and_validation_data:
    :param method:
    :param name:
    :param plot_now:
    :param library:
    :return:
    """
    if name == "":
        name = method
    method = method.lower()
    variable = {
        "parameter": parameter_array,
        "MSE": np.zeros(len(parameter_array)),
        "SPAR": np.zeros(len(parameter_array)),
        "model": list(np.zeros(len(parameter_array))),
    }

    xdot_train, x_train, u_train, xdot_val, x_val, u_val = train_and_validation_data

    for i, para in tqdm(enumerate(parameter_array)):
        if method[:3] == "sr3":
            optimizer = ps.SR3(thresholder=method[-2:], nu=para, threshold=1e-12)
        elif method == "lasso":
            optimizer = Lasso(alpha=para, fit_intercept=False)
        elif method == "stlsq":
            optimizer = ps.STLSQ(threshold=0.1, alpha=para)
        elif method == "srr":
            optimizer = ps.SSR(alpha=para)
        else:
            raise NameError("Method is invalid")

        if library is None:
            library = ps.PolynomialLibrary(degree=2, include_interaction=True)

        model = ps.SINDy(optimizer=optimizer, feature_library=library)
        print("Fitting model")
        model.fit(x_train, u=u_train, t=None, x_dot=xdot_train)  # fit on training data
        if model.coefficients().ndim == 1:  # fix dimensions of this matrix, bug in pysindy, o this works
            model.optimizer.coef_ = model.coefficients().reshape(1, model.coefficients().shape[0])
        variable["MSE"][i] = model.score(x_val, u=u_val, x_dot=xdot_val, metric=mean_squared_error)
        # the same as: mean_squared_error(model.predict(x_val, u_val), xdot_val)
        variable["SPAR"][i] = np.count_nonzero(model.coefficients())  # number of non-zero elements
        variable["model"][i] = model

    # rel_sparsity = variable['SPAR'] / np.max(variable['SPAR'])

    # plot and save the results
    xlab = r"Sparsity weighting factor $\alpha$"
    ylab1 = "MSE"
    ylab2 = "Number of non-zero elements"
    title = "MSE and sparsity VS Sparsity weighting parameter, " + method + " method"
    xydata = [
        np.hstack(
            (variable["parameter"].reshape(len(parameter_array), 1), variable["MSE"].reshape(len(parameter_array), 1))
        ),
        np.hstack(
            (variable["parameter"].reshape(len(parameter_array), 1), variable["SPAR"].reshape(len(parameter_array), 1))
        ),
    ]
    specs = ["r", "b"]
    # save_plot_data(name, xydata, title, xlab, [ylab1, ylab2], specs, plot_now=plot_now)

    idx = np.where(np.min(variable["MSE"]))  # best model, lowest MSE
    best_model = variable["model"][idx[0][0]]
    print(
        "Best model found with MSE: ",
        variable["MSE"][idx[0][0]],
        " and parameter: ",
        variable["parameter"][idx[0][0]],
        "for ",
        method,
    )
    return best_model

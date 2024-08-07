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


def optimize_parameters(path_to_data_files, mode='torque', additional_name=""):
    """
    Calculates for various parameters, plots MSE and Sparsity, for SR3 and Lasso optimisation
    """
    both = True

    DATA = prepare_data(path_to_data_files, number_of_trainfiles=30, usage_per_trainfile=.25) #use 30 random samples, each 25% data
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
        raise NotImplementedError("Magnetic coenergy is not fully implemented yet")
        XDOT = [DATA['wcoe_train'], DATA['wcoe_mag']]
        namestr= "W"
    elif mode == "merged":
        # now, the xdot_train contains i and V and I,
        raise NotImplementedError("Merged mode is not fully implemented yet")


    else:
        raise ValueError("mode is either currents, torque or ump")

    if not both:
        print("SR3_L1 optimisation")
        n = 1
        trials = 1
        l_range = [1e-5, 1e2]
        n_range = [1e-11, 1e-5]
        with multiprocessing.Pool(n) as pool:
            pool.starmap(optuna_search_sr3, [[DATA, XDOT, l_range, n_range, namestr, trials] for _ in range(n)])
        pool.join()
        pool.close()
        # plot_optuna_data(name=namestr + "-sr3-study")

        print("Lasso optimisation")
        n = 1
        trials = 1
        a_range = [1e-5, 1e2]
        with multiprocessing.Pool(n) as pool:
            pool.starmap(optuna_search_lasso, [[DATA, XDOT, a_range, namestr, trials] for _ in range(n)])

        pool.join()
        pool.close()
        # plot_optuna_data(name=namestr + "-lasso-study")

    elif both:
        print("SR3_L1 and lasso optimisation")
        n = 2
        trials = 100
        a_range = [1e-5, 1e2]
        l_range = [1e-10, 1e2]
        n_range = [1e-12, 1e2]

        with joblib.parallel_config(n_jobs = n, backend = "loky", inner_max_num_threads=1):
            joblib.Parallel(n_jobs=n)(joblib.delayed(optuna_search_both)(DATA, XDOT, l_range, n_range, a_range, namestr+additional_name, trials) for _ in range(n))

        #with multiprocessing.Pool(n) as pool:
        #    pool.starmap(optuna_search_both, [[DATA, XDOT, l_range, n_range, a_range, namestr, trials] for _ in range(n)])
        #pool.join()
        #pool.close()

    # 32Gb ram -> 4 cores (yikes)
    return


def optuna_search_both(DATA, XDOT, lminmax, nminmax, aminmax, studyname, iter):
    # from https://optuna.readthedocs.io/en/stable/index.html
    # XDOT = f(DATA) with first element training, second validation
    def objective(trial):
        lib_choice = trial.suggest_categorical('lib_choice',
                                               get_library_names())

        lib = get_custom_library_funcs(lib_choice)
        optimizer_name = trial.suggest_categorical('optimizer', ['lasso', 'sr3'])
        if optimizer_name == 'lasso':
            alphas = trial.suggest_float('alphas', aminmax[0], aminmax[1], log=True)
            optimizer = ps.WrappedOptimizer(Lasso(alpha=alphas, fit_intercept=False, precompute=True))

        elif optimizer_name == 'sr3':
            lambdas = trial.suggest_float('lambdas', lminmax[0], lminmax[1], log=True)
            nus = trial.suggest_float('nus', nminmax[0], nminmax[1], log=True)
            optimizer = ps.SR3(thresholder='l1', nu=nus,
                               threshold=lambdas)

        model = ps.SINDy(optimizer=optimizer, feature_library=lib)
        model.fit(DATA["x_train"], u=DATA["u_train"], t=None, x_dot=XDOT[0])

        MSE = model.score(DATA["x_val"], u=DATA["u_val"], x_dot=XDOT[1],
                          t=None, metric=mean_squared_error)
        SPAR = np.count_nonzero(model.coefficients())

        return MSE, SPAR

    study_name = studyname + "-optuna-study"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(directions=['minimize', 'minimize'],
                                study_name=study_name,
                                storage=storage_name,
                                load_if_exists=True)

    study.optimize(objective, n_trials=iter, n_jobs=1)
    return


def optuna_search_lasso(DATA, XDOT, minmax, studyname, iter):
    # from https://optuna.readthedocs.io/en/stable/index.html
    # XDOT = f(DATA) with first element training, second validation
    def objective(trial):
        alphas = trial.suggest_float('alphas', minmax[0], minmax[1], log=True)
        lib_choice = trial.suggest_categorical('lib_choice',
                                               get_library_names())

        lib = get_custom_library_funcs(lib_choice)

        optimizer = ps.WrappedOptimizer(Lasso(alpha=alphas, fit_intercept=False))

        model = ps.SINDy(optimizer=optimizer,
                         feature_library=lib)

        model.fit(DATA["x_train"], u=DATA["u_train"], t=None, x_dot=XDOT[0])

        MSE = model.score(DATA["x_val"], u=DATA["u_val"], x_dot=XDOT[1],
                          t=None, metric=mean_squared_error)
        SPAR = np.count_nonzero(model.coefficients())

        return MSE, SPAR

    study_name = studyname + "-lasso-study"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(directions=['minimize', 'minimize'],
                                study_name=study_name,
                                storage=storage_name,
                                load_if_exists=True)

    study.optimize(objective, n_trials=iter, n_jobs=1)

    return


def optuna_search_sr3(DATA, XDOT, l_minmax, n_minmax, studyname, iter):
    # from https://optuna.readthedocs.io/en/stable/index.html
    # XDOT same meaning as from SINDy, with the first element being the training data, second validation

    def objective(trial):
        lambdas = trial.suggest_float('lambdas', l_minmax[0], l_minmax[1], log=True)
        nus = trial.suggest_float('nus', n_minmax[0], n_minmax[1], log=True)

        # todo maybe leave out this?
        lib_choice = trial.suggest_categorical('lib_choice',
                                               get_library_names())  # 'best' eats all the memory

        lib = get_custom_library_funcs(lib_choice)

        optimizer = ps.SR3(thresholder='l1', nu=nus,
                           threshold=lambdas)
        model = ps.SINDy(optimizer=optimizer,
                         feature_library=lib)
        model.fit(DATA["x_train"], u=DATA["u_train"], t=None, x_dot=XDOT[0])

        MSE = model.score(DATA["x_val"], u=DATA["u_val"], x_dot=XDOT[1],
                          t=None, metric=mean_squared_error)
        SPAR = np.count_nonzero(model.coefficients())

        return MSE, SPAR

    study_name = studyname + "-sr3-study"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(directions=['minimize', 'minimize'],
                                study_name=study_name,
                                storage=storage_name,
                                load_if_exists=True)

    study.optimize(objective, n_trials=iter, n_jobs=1)

    # DON't turn on when multiprocessing
    # optuna.visualization.plot_pareto_front(study, target_names= ["MSE","SPAR"]).show(renderer="browser")

    return


def plot_optuna_data(name, dirs = ""):
    stud = optuna.load_study(study_name=name, storage="sqlite:///" + "optuna_studies//"+dirs + name + ".db")
    optuna.visualization.plot_pareto_front(stud, target_names=["MSE", "SPAR"]).show(renderer="browser")
    print(f"Trial count: {len(stud.trials)}")
    return


def parameter_search(parameter_array, train_and_validation_data, method="lasso", name="", plot_now=True, library=None):
    # THIS IS UNUSED, TO BE REMOVED
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

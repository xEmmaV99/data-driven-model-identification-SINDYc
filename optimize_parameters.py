import joblib

from prepare_data import *
import optuna
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
import pysindy as ps
from libs import get_custom_library_funcs
from libs import get_library_names
import tqdm
import matplotlib.pyplot as plt
from optuna.study._multi_objective import _get_pareto_front_trials_by_trials
from source import set_plot_defaults


def optimize_parameters(
    path_to_data_files: str,
    mode: str = "torque",
    additional_name: str = "",
    n_jobs: int = 1,
    n_trials: int = 100,
    ecc_input=False,
    seed = None,
):
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

    #print("ecc_input =", ecc_input)

    DATA = prepare_data(
        path_to_data_files,
        number_of_trainfiles=30,
        usage_per_trainfile=0.50,
        ecc_input=ecc_input,
        seed = seed
    )

    # select the desired mode
    if mode == "currents":
        XDOT = [DATA["xdot_train"], DATA["xdot_val"]]
        namestr = "currents"
    elif mode == "torque":
        XDOT = [DATA["T_em_train"], DATA["T_em_val"]]
        namestr = "torque"
    elif mode == "ump":
        XDOT = [DATA["UMP_train"], DATA["UMP_val"]]
        namestr = "ump"
    elif mode == "wcoe":
        XDOT = [DATA["wcoe_train"], DATA["wcoe_val"]]
        namestr = "W"
    elif mode == "merged":
        # now, the xdot_train contains i and V and I,
        raise NotImplementedError("Merged mode is not fully implemented yet")
    else:
        raise ValueError("mode is either currents, torque or ump")

    # set the ranges of the hyperparameters, a for alpha (used in lasso and stlsq), l for lambda (used in sr3), n for nu (used in sr3)
    a_range = [1e-5, 1e2]
    l_range = [1e-10, 1e2]
    n_range = [1e-12, 1e2]

    _study(namestr + additional_name)  # initialize the study
    with joblib.parallel_config(n_jobs=n_jobs, backend="loky", inner_max_num_threads=1):
        joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(optuna_search)(
                DATA,
                XDOT,
                l_range,
                n_range,
                a_range,
                namestr + additional_name,
                n_trials,
            )
            for _ in range(n_jobs)
        )
    return


def optuna_search(
    DATA: dict,
    XDOT: np.array,
    lminmax: list,
    nminmax: list,
    aminmax: list,
    studyname: str,
    iter: int,
):
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
    # set parameters
    optimizer_list = ["lasso", "sr3", "stlsq"]
    library_list = get_library_names()

    def objective(trial):
        # this is the objective function that optuna will try to optimize
        lib_choice = trial.suggest_categorical("lib_choice", library_list)
        lib = get_custom_library_funcs(
            lib_choice, DATA["u"].shape[1] + DATA["x"].shape[1]
        )

        optimizer_name = trial.suggest_categorical("optimizer", optimizer_list)
        if optimizer_name == "lasso":
            alphas = trial.suggest_float("alphas", aminmax[0], aminmax[1], log=True)
            optimizer = ps.WrappedOptimizer(
                Lasso(alpha=alphas, fit_intercept=False, precompute=True)
            )

        elif optimizer_name == "sr3":
            lambdas = trial.suggest_float("lambdas", lminmax[0], lminmax[1], log=True)
            nus = trial.suggest_float("nus", nminmax[0], nminmax[1], log=True)
            #DEBUG optimizer = ps.SR3(thresholder="l1", nu=nus, threshold=lambdas)
            optimizer = ps.SR3(regularizer="l1", reg_weight_lam=lambdas, relax_coeff_nu=nus)

        elif optimizer_name == "stlsq":
            alphas = trial.suggest_float("alphas", aminmax[0], aminmax[1], log=True)
            # alpha is penalising the l2 norm of the coefficients (ridge regression)
            threshold = trial.suggest_float("threshold", 0.001, 1, log=True)
            optimizer = ps.STLSQ(alpha=alphas, threshold=threshold)

        else:
            raise ValueError("Optimizer not known.")

        try:  # OOM error handling
            model = ps.SINDy(optimizer=optimizer, feature_library=lib)
            model.fit(DATA["x_train"], u=DATA["u_train"], t=None, x_dot=XDOT[0])
        except Exception as e:
            print("Exception in model fitting:", e)
            raise optuna.TrialPruned()

        # Alternatively, use RMSE or MAE as metric
        '''
        MAE = model.score(
            DATA["x_val"],
            u=DATA["u_val"],
            x_dot=XDOT[1],
            t=None,
            metric=mean_absolute_error,
        )
        RMSE = model.score(
            DATA["x_val"],
            u=DATA["u_val"],
            x_dot=XDOT[1],
            t=None,
            metric=mean_squared_error,
        )
        '''
        MSE = model.score(
            DATA["x_val"],
            u=DATA["u_val"],
            x_dot=XDOT[1],
            t=None,
            metric=mean_squared_error,
        )
        SPAR = np.count_nonzero(model.coefficients())
        return MSE, SPAR

    study = _study(studyname)
    study.optimize(objective, n_trials=iter, n_jobs=1)
    return


def _study(studyname):
    """
    Initializes the optuna study
    :param studyname: str with the name of the study
    :return:
    """
    study_name = (
        "optuna_studies//" + studyname + "-optuna-study"
    )  # Unique identifier of the study.

    # check if the directory exists
    if not os.path.exists("optuna_studies"):
        os.makedirs("optuna_studies")

    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(
        directions=["minimize", "minimize"],  # minimize MSE and sparsity
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
    )
    return study


def plot_optuna_data(name):
    """
    Plots the pareto front of the optuna study
    :param name: name of the optuna study, usually ends with '-optuna-study', assuming it is in the optuna_studies directory
    :return:
    """
    print(
        optuna.study.get_all_study_names(
            storage="sqlite:///" + "optuna_studies/" + name + ".db"
        )
    )
    stud = optuna.load_study(
        study_name=None, storage="sqlite:///" + "optuna_studies/" + name + ".db"
    )
    optuna.visualization.plot_pareto_front(stud, target_names=["MSE", "SPAR"]).show(
        renderer="browser"
    )
    print(f"Trial count: {len(stud.trials)}")
    return


def plot_pareto(study, limits:list = None, target_names:list=None, logscale:bool=False,
                save_name:str='dummy', show:bool = False, mark_trials:list = None):  # uses matplotlib to plot the paretoplot
    '''
    This function plots the optuna study by using matplotlib
    :param study: a optuna study object
    :param limits: limits of the plot, [[xm, xM], [ym, yM]]
    :param target_names: Names for the x and y axis
    :param logscale: boolean, if True, logscale is used
    :param save_name: str, used for saving the pdf file
    :param show: if true, plt.show() is called
    :param mark_trials: list with trial indices to mark with a circle
    :return:
    '''
    if target_names is None:
        target_names = [r"Mean Squared Error", r"Nonzero elements"]

    print("Using new function for matplotlib plot.")
    ax = set_plot_defaults()
    cmap = plt.get_cmap("tab10")

    ax.set_xlabel(target_names[0], fontsize=7)
    ax.set_ylabel(target_names[1], fontsize=7)

    # label must be according to library choice
    markershapes = {'sr3': '.', 'stlsq': 'x', 'lasso': 'v'}
    libcolors = {"poly_2nd_order": cmap(0),
                 "linear-specific": cmap(1),
                 "torque": cmap(2),
                 "nonlinear_terms": cmap(3),
                 "interaction_only": cmap(4)}
    all_trials = study.trials

    # reduce all_trials to trials with values in limits
    if limits is not None:
        all_trials = [trial for trial in all_trials if trial.values is not None and all([limits[i][0] <= trial.values[i] <= limits[i][1] for i in range(len(limits))])]

    best_trials = _get_pareto_front_trials_by_trials(all_trials, study.directions) # use optuna to select the best trials

    # split per optimizer, because each has a different marker
    opt1_trials = [trial for trial in all_trials if trial.params['optimizer'] == 'sr3' if trial.values is not None]
    opt2_trials = [trial for trial in all_trials if trial.params['optimizer'] == 'stlsq' if trial.values is not None]
    opt3_trials = [trial for trial in all_trials if trial.params['optimizer'] == 'lasso' if trial.values is not None]

    if not opt1_trials == []: # if no trials, then skip
        ax.scatter(
            x=[trial.values[0] for trial in opt1_trials],
            y=[trial.values[1] for trial in opt1_trials],
            color=[libcolors[trial.params['lib_choice']] for trial in opt1_trials],
            label=[trial.params['lib_choice'] for trial in opt1_trials],
            marker=markershapes['sr3'], alpha=[1 if trial in best_trials else 0.3 for trial in opt1_trials],
            s = 13
        )
    if not opt2_trials == []:
        ax.scatter(
            x=[trial.values[0] for trial in opt2_trials],
            y=[trial.values[1] for trial in opt2_trials],
            color=[libcolors[trial.params['lib_choice']] for trial in opt2_trials],
            label=[trial.params['lib_choice'] for trial in opt2_trials],
            marker=markershapes['stlsq'], alpha=[1 if trial in best_trials else 0.3 for trial in opt2_trials],
            s=13
        )
    if not opt3_trials == []:
        ax.scatter(
            x=[trial.values[0] for trial in opt3_trials],
            y=[trial.values[1] for trial in opt3_trials],
            color=[libcolors[trial.params['lib_choice']] for trial in opt3_trials],
            label=[trial.params['lib_choice'] for trial in opt3_trials],
            marker=markershapes['lasso'], alpha=[1 if trial in best_trials else 0.3 for trial in opt3_trials],
            s=13
        )
    if all ([opt1_trials == [], opt2_trials == [], opt3_trials == []]):
        print("No trials found inside the limits")

    # extra mark the special trials
    if mark_trials is not None:
        ax.scatter(
            x = [study.trials[trial_id].values[0] for trial_id in mark_trials],
            y = [study.trials[trial_id].values[1] for trial_id in mark_trials],
            color='black', marker='o', alpha=1,
            s=70, facecolors='none'
        )

    if logscale:
        plt.xscale("log")

    plt.tight_layout()
    ax.set_axisbelow(True) # grid behind the points
    plt.grid(True, which="both")
    if limits is not None:
        plt.xlim(limits[0])
        plt.ylim(limits[1])

    # add legend
    lines = []
    labels = {"poly_2nd_order": r'A',
              "linear-specific": r'B',
              "torque": r'C',
              "nonlinear_terms": r'D',
              "interaction_only": r'E'}

    def in_scope(values):
        # inscope checks if a trial is inside the limits, to make sure the legend does not contain things that are not plotted
        if values is None: # failed trial iguess ?
            return False
        if limits is None:
            return True
        if values[0] < limits[0][0] or values[0] > limits[0][1]:
            return False
        if values[1] < limits[1][0] or values[1] > limits[1][1]:
            return False
        return True

    for key in libcolors:
        # check if trial is plotted inside the limits of the plots
        if key in [trial.params['lib_choice'] for trial in all_trials if in_scope(trial.values)]:
            lines.append(plt.Line2D([0], [0], marker='o', color='w', label=labels[key], markerfacecolor=libcolors[key],
                                    markersize=5))

    leg1 = plt.legend(handles=lines, loc='upper right', fontsize=7, title=r"Library", bbox_to_anchor=(1, 1))

    # create second legend with markershapes
    lines = []
    for key in markershapes:
        lines.append(plt.Line2D([0], [0], marker=markershapes[key], color='k', label=key, markersize=5, linestyle=""))
    leg2 = plt.legend(handles=lines, loc='upper right', fontsize=7, title=r"Optimizer", bbox_to_anchor=(0.8, 1))

    # add legends
    ax.add_artist(leg1), ax.add_artist(leg2)

    # # if delta lim is very small, only plot every 3rd other tick:
    if limits[0][0] > 1 and limits[0][1] - limits[0][0] < 0.01:
        for label in ax.xaxis.get_ticklabels()[::2]:
            label.set_visible(False)

    plt.savefig('pdfs//optuna//' + save_name + '.pdf', dpi=600.0) # save as pdf

    if show:
        plt.show()
    return ax




def parameter_search(
    parameter_array,
    train_and_validation_data,
    method="lasso",
    name="",
    library=None,
):
    """
    Old implementation of the parameter search, is not used since Optuna does a better job, including multi-parameter search
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
        if (
            model.coefficients().ndim == 1
        ):  # fix dimensions of this matrix, bug in pysindy, o this works
            model.optimizer.coef_ = model.coefficients().reshape(
                1, model.coefficients().shape[0]
            )
        variable["MSE"][i] = model.score(
            x_val, u=u_val, x_dot=xdot_val, metric=mean_squared_error
        )
        # the same as: mean_squared_error(model.predict(x_val, u_val), xdot_val)
        variable["SPAR"][i] = np.count_nonzero(
            model.coefficients()
        )  # number of non-zero elements
        variable["model"][i] = model

    # rel_sparsity = variable['SPAR'] / np.max(variable['SPAR'])

    # plot and save the results
    xlab = r"Sparsity weighting factor $\alpha$"
    ylab1 = "MSE"
    ylab2 = "Number of non-zero elements"
    title = "MSE and sparsity VS Sparsity weighting parameter, " + method + " method"
    xydata = [
        np.hstack(
            (
                variable["parameter"].reshape(len(parameter_array), 1),
                variable["MSE"].reshape(len(parameter_array), 1),
            )
        ),
        np.hstack(
            (
                variable["parameter"].reshape(len(parameter_array), 1),
                variable["SPAR"].reshape(len(parameter_array), 1),
            )
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

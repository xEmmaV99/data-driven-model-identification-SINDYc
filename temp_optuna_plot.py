import optuna
import matplotlib.pyplot as plt
from collections.abc import Sequence
from optuna.trial import FrozenTrial
from typing import Any
from typing import Callable
from optuna.study import Study
from optuna.visualization._pareto_front import _ParetoFrontInfo
from optuna.visualization.matplotlib._pareto_front import _get_pareto_front_info

def func(
        trials: Sequence[FrozenTrial],
        include_dominated_trials: bool,
        dominated_trials: bool = False,
        infeasible: bool = False,
) -> dict[str, Any]:  # overwrite function for html plot
    print("Using new function for html plot.")
    if dominated_trials and not include_dominated_trials:
        assert len(trials) == 0

    if infeasible:
        return {
            "color": "#34d42f",
        }
    elif dominated_trials:
        return {
            "line": {"width": 0.5, "color": "Grey"},
            "color": [t.number for t in trials],
            "colorscale": "Greens",
            "colorbar": {
                "title": "Trial",
            },
        }
    else:
        return {
            "line": {"width": 0.5, "color": "Grey"},
            "color": [t.number for t in trials],
            "colorscale": "Greens",
            "colorbar": {
                "title": "Best Trial",
                "x": 1.1 if include_dominated_trials else 1,
                "xpad": 40,
            },
        }
def plot_pareto_front(
    study: Study,
    *,
    target_names: list[str] | None = None,
    include_dominated_trials: bool = True,
    axis_order: list[int] | None = None,
    constraints_func: Callable[[FrozenTrial], Sequence[float]] | None = None,
    targets: Callable[[FrozenTrial], Sequence[float]] | None = None,
) -> "Axes":

    info = _get_pareto_front_info(
        study, target_names, include_dominated_trials, axis_order, constraints_func, targets
    )
    return _get_pareto_front_plot(info, study) #study is new here


def _get_pareto_front_plot(info: _ParetoFrontInfo, study) -> "Axes":
    if info.n_targets == 2:
        return _get_pareto_front_2d(info, study) #pass study here


def _get_pareto_front_2d(info: _ParetoFrontInfo, study) -> "Axes":  # overwrite function for matplotlib plot
    # Set up the graph style.
    print("Using new function for matplotlib plot.")
    plt.style.use("ggplot")  # Use ggplot style sheet for similar outputs to plotly.
    _, ax = plt.subplots()
    ax.set_title("Pareto-front Plot")
    cmap = plt.get_cmap("tab10")  # Use tab10 colormap for similar outputs to plotly.

    ax.set_xlabel(info.target_names[info.axis_order[0]])
    ax.set_ylabel(info.target_names[info.axis_order[1]])

    trial_label: str = "Trial"

    # plot all trials in info.non_best_trials_with_values and info.best_trials_with_values but change color and markersize according to library choice and optimiser choice
    # label must be according to library choice
    markershapes = {'sr3': 'o', 'stlsq': 'x', 'lasso': '.'}
    libcolors = { "poly_2nd_order": cmap(0),
        "linear-specific": cmap(1),
        "torque": cmap(2),
        "nonlinear_terms":cmap(3),
        "interaction_only":cmap(4)}
    all_trials = study.trials
    # for trial in all_trials:
    #     ax.scatter(
    #         x = trial.values[0], y = trial.values[1],
    #         color = libcolors[trial.params['lib_choice']],
    #         label = trial.params['lib_choice'],
    #         marker = markershapes[trial.params['optimizer']]
    #     )

    ax.scatter(
        x = [trial.values[0] for trial in all_trials],
        y = [trial.values[1] for trial in all_trials],
        color = [libcolors[trial.params['lib_choice']] for trial in all_trials],
        label = [trial.params['lib_choice'] for trial in all_trials],
        #marker = [markershapes[trial.params['optimizer']] for trial in all_trials] ## marker not possible
    )
    # set x axis to log
    ax.set_xscale('log')

    if info.non_best_trials_with_values is not None and ax.has_data():
        #ax.legend()
        pass
    return ax

# overwrite functions from optuna
optuna.visualization._pareto_front._make_marker = func

optuna.visualization.matplotlib._pareto_front._get_pareto_front_2d = _get_pareto_front_2d
optuna.visualization.matplotlib._pareto_front._get_pareto_front_plot = _get_pareto_front_plot
optuna.visualization.matplotlib.plot_pareto_front = plot_pareto_front

# plot pareto front
opt_study_name = 'torquelinear_premade-optuna-study'
stud = optuna.load_study(
    study_name=None, storage="sqlite:///" + "optuna_studies/" + opt_study_name + ".db"
)

### html figure
# optuna.visualization.plot_pareto_front(stud, target_names=["MSE", "SPAR"]).show(
#        renderer="browser"
#    )

### matplotlib figure
fig = optuna.visualization.matplotlib.plot_pareto_front(stud)

plt.show()

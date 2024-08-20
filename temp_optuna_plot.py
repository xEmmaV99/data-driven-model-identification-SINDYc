import optuna
import matplotlib.pyplot as plt
from collections.abc import Sequence
from optuna.trial import FrozenTrial
from typing import Any
from optuna import

def func(
        trials: Sequence[FrozenTrial],
        include_dominated_trials: bool,
        dominated_trials: bool = False,
        infeasible: bool = False,
) -> dict[str, Any]: #overwrite function for html plot
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

def _get_pareto_front_2d(info: _ParetoFrontInfo) -> "Axes": #overwrite function for matplotlib plot
    # Set up the graph style.
    plt.style.use("ggplot")  # Use ggplot style sheet for similar outputs to plotly.
    _, ax = plt.subplots()
    ax.set_title("Pareto-front Plot")
    cmap = plt.get_cmap("tab10")  # Use tab10 colormap for similar outputs to plotly.

    ax.set_xlabel(info.target_names[info.axis_order[0]])
    ax.set_ylabel(info.target_names[info.axis_order[1]])

    trial_label: str = "Trial"
    if len(info.infeasible_trials_with_values) > 0:
        ax.scatter(
            x=[values[info.axis_order[0]] for _, values in info.infeasible_trials_with_values],
            y=[values[info.axis_order[1]] for _, values in info.infeasible_trials_with_values],
            color="#cccccc",
            label="Infeasible Trial",
        )
        trial_label = "Feasible Trial"
    if len(info.non_best_trials_with_values) > 0:
        ax.scatter(
            x=[values[info.axis_order[0]] for _, values in info.non_best_trials_with_values],
            y=[values[info.axis_order[1]] for _, values in info.non_best_trials_with_values],
            color=cmap(0),
            label=trial_label,
        )
    if len(info.best_trials_with_values) > 0:
        ax.scatter(
            x=[values[info.axis_order[0]] for _, values in info.best_trials_with_values],
            y=[values[info.axis_order[1]] for _, values in info.best_trials_with_values],
            color=cmap(3),
            label="Best Trial",
        )

    if info.non_best_trials_with_values is not None and ax.has_data():
        ax.legend()

    return ax

optuna.visualization._pareto_front._make_marker = func

opt_study_name = 'torquelinear_premade-optuna-study'
stud = optuna.load_study(
    study_name=None, storage="sqlite:///" + "optuna_studies/" + opt_study_name + ".db"
)

#optuna.visualization.plot_pareto_front(stud, target_names=["MSE", "SPAR"]).show(
#        renderer="browser"
#    )

fig = optuna.visualization.matplotlib.plot_pareto_front(stud)

plt.show()

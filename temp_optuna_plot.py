import optuna
import matplotlib.pyplot as plt


def plot_pareto(study, limits, target_names=None, logscale=False, save_name = 'dummy') -> "Axes":  # overwrite function for matplotlib plot
    # Set up the graph style.
    if target_names is None:
        target_names = [r"Mean Squared Error", r"Nonzero elements"]

    print("Using new function for matplotlib plot.")
    _, ax = plt.subplots(figsize=(3.5, 3.5))
    plt.rcParams.update({'font.size': 7})
    plt.yticks(fontsize=7)
    plt.xticks(fontsize=7)
    plt.rcParams['text.usetex'] = True
    cmap = plt.get_cmap("tab10")  # Use tab10 colormap for similar outputs to plotly.

    ax.set_xlabel(target_names[0], fontsize=7)
    ax.set_ylabel(target_names[1], fontsize=7)

    # label must be according to library choice
    markershapes = {'sr3': '.', 'stlsq': 'x', 'lasso': 'v'}
    libcolors = { "poly_2nd_order": cmap(0),
        "linear-specific": cmap(1),
        "torque": cmap(2),
        "nonlinear_terms":cmap(3),
        "interaction_only":cmap(4)}
    all_trials = study.trials

    opt1_trials = [trial for trial in all_trials if trial.params['optimizer'] == 'sr3']
    opt2_trials = [trial for trial in all_trials if trial.params['optimizer'] == 'stlsq']
    opt3_trials = [trial for trial in all_trials if trial.params['optimizer'] == 'lasso']

    ax.scatter(
        x=[trial.values[0] for trial in opt1_trials],
        y=[trial.values[1] for trial in opt1_trials],
        color=[libcolors[trial.params['lib_choice']] for trial in opt1_trials],
        label=[trial.params['lib_choice'] for trial in opt1_trials],
        marker =  markershapes['sr3']
    )
    ax.scatter(
        x=[trial.values[0] for trial in opt2_trials],
        y=[trial.values[1] for trial in opt2_trials],
        color=[libcolors[trial.params['lib_choice']] for trial in opt2_trials],
        label=[trial.params['lib_choice'] for trial in opt2_trials],
        marker=markershapes['stlsq']
    )
    ax.scatter(
        x=[trial.values[0] for trial in opt3_trials],
        y=[trial.values[1] for trial in opt3_trials],
        color=[libcolors[trial.params['lib_choice']] for trial in opt3_trials],
        label=[trial.params['lib_choice'] for trial in opt3_trials],
        marker=markershapes['lasso']
    )
    if logscale:
        plt.xscale("log")
    plt.tight_layout()
    ax.set_axisbelow(True)
    plt.grid(True, which="both")
    plt.xlim(limits[0])
    plt.ylim(limits[1])

    #plot emtpy for legend
    lines = []
    labels =  {"poly_2nd_order": r'A',
        "linear-specific": r'B',
        "torque": r'C',
        "nonlinear_terms": r'D',
        "interaction_only":r'E'}
    def in_scope(values):
        if values[0] < limits[0][0] or values[0] > limits[0][1]:
            return False
        if values[1] < limits[1][0] or values[1] > limits[1][1]:
            return False
        return True
    for key in libcolors:
        # check if trial is plotted inside the limits of the plots
        if key in [trial.params['lib_choice'] for trial in all_trials if in_scope(trial.values)]:
            lines.append(plt.Line2D([0], [0], marker='o', color='w', label=labels[key], markerfacecolor=libcolors[key], markersize=5))

    leg1 = plt.legend(handles=lines, loc='upper right', fontsize=7, title= "Library", bbox_to_anchor = (1,1))

    # create second legend with markershapes
    lines = []
    for key in markershapes:
        lines.append(plt.Line2D([0], [0], marker=markershapes[key], color='k', label=key, markersize=5, linestyle=""))
    leg2 = plt.legend(handles=lines, loc='upper right', fontsize=7, title= "Optimizer", bbox_to_anchor = (0.8,1))

    # add legends
    ax.add_artist(leg1), ax.add_artist(leg2)

    plt.savefig('pdfs//'+save_name + '.pdf', dpi=600.0)
    return ax


# plot pareto front
opt_study_name = 'torquelinear_premade-optuna-study'
stud = optuna.load_study(
    study_name=None, storage="sqlite:///" + "optuna_studies/" + opt_study_name + ".db"
)

### matplotlib figure
plot_pareto(stud,
            limits = [[2e-6,1e-4],[-1,51]],
            logscale = True,
            save_name = 'test',
            target_names = [r'Mean Squared Error ($N^2 m^2$)', r'Nonzero elements'])

plt.show()

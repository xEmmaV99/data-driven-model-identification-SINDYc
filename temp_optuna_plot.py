import optuna
from optimize_parameters import plot_pareto, plot_optuna_data


# plot pareto front

opt_study_name = 'torquelinear_premade-optuna-study'
stud = optuna.load_study(
    study_name=None, storage="sqlite:///" + "optuna_studies/" + opt_study_name + ".db"
)


plot_optuna_data(opt_study_name)
### matplotlib figure
plot_pareto(stud,
            limits=[[2e-6, 5e-5], [-1, 51]],
            logscale=True,
            save_name='test',
            target_names=[r'Mean Squared Error ($N^2 m^2$)', r'Nonzero elements'],
            show = True, mark_trials = [396, 720])
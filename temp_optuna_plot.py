import optuna
from optimize_parameters import plot_pareto, plot_optuna_data
from source import *
import os
# todo write file to generate all pareto plots :)
# plot pareto front


# opt_study_name = 'torquelinear_premade-optuna-study'
# stud = optuna.load_study(
#     study_name=None, storage="sqlite:///" + "optuna_studies/" + opt_study_name + ".db"
# )
#
#
# ### matplotlib figure
# plot_pareto(stud,
#             limits=[[2e-6, 5e-5], [-1, 51]],
#             logscale=True,
#             save_name='test',
#             target_names=[r'Mean Squared Error ($N^2 m^2$)', r'Nonzero elements'],
#             show = True, mark_trials = [396, 720])


# Now, I want to plot a plot of the currents

path1 = os.path.join(os.getcwd(), 'plot_data','_w5','currents_nl','currents70.pkl')
path2 = os.path.join(os.getcwd(), 'plot_data','_w5','currents_50_nl','currents70.pkl')
path3 = os.path.join(os.getcwd(), 'plot_data','_w5','currents_d_nl','currents60.pkl')
path4 = os.path.join(os.getcwd(), 'plot_data','_w5','currents_nl','currents_simulation70.pkl')
path5 = os.path.join(os.getcwd(), 'plot_data','_w5','currents_50_nl','currents_simulation70.pkl')
path6 = os.path.join(os.getcwd(), 'plot_data','_w5','currents_d_nl','currents_simulation60.pkl')
datalist = [path1, path2, path3, path4, path5, path6]
#plot_tiled_curr(datalist)


path1 = os.path.join(os.getcwd(), 'plot_data','_w5','torque_nl','torque5.pkl')
path2 = os.path.join(os.getcwd(), 'plot_data','_w5','torque_50_nl','torque10.pkl')
path3 = os.path.join(os.getcwd(), 'plot_data','_w5','torque_d_nl','torque50.pkl')

path5 = os.path.join(os.getcwd(), 'plot_data','_w5','ump_50_nl','UMP100.pkl')
path6 = os.path.join(os.getcwd(), 'plot_data','_w5','ump_d_nl','UMP150.pkl')
datalist = [path1, path2, path3, None, path5, path6]


plot_tiled_TF(datalist)
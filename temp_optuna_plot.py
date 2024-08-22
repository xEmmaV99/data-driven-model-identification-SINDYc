import optuna
from optimize_parameters import plot_pareto, plot_optuna_data
from source import *
import os

# todo write file to generate all pareto plots :)
# plot pareto front
part1 = True
part2 = False

if part1:
    opt_study_names = ['currentsnonlinear',
                       'currentsnonlinear_50ecc',
                       'currentsnonlinear_dynamic_50ecc',
                       'torquenonlinear',
                       'torquenonlinear_50ecc',
                       'torquenonlinear_dynamic_50ecc',
                       'umpnonlinear',
                       'umpnonlinear_50ecc',
                       'umpnonlinear_dynamic_50ecc'
                       ]
    limit_list = [[[5e1, 3e3], [0, 400]],
                  [[5e1, 3e3], [0, 400]],
                  [[5e2, 3e3], [0, 400]],
                  [[1e-5, 1e-3], [0, 150]],
                  [[4e-5, 1e-3], [0, 150]],
                  [[5e-4, 2e-2], [0, 80]],
                  [[1.91, 1.9175], [0, 90]],
                  [[3, 1e4], [0, 500]],
                  [[5e3, 5e5], [0, 800]]
                  ]
    #mark_idx = [0, 0, 0, 0, 0, 0, 0, 0, 0] # todo

    for j, study in enumerate(opt_study_names):
        stud = optuna.load_study(study_name=None,
                                 storage="sqlite:///" + "optuna_studies/_w5/" + study + "-optuna-study.db")
        ### html interactive figure
        plot_optuna_data(study+'-optuna-study')

        ### matplotlib figure
        plot_pareto(stud,
                    limits=limit_list[j],
                    logscale=True,
                    save_name='test',
                    target_names=[r'Mean Squared Error ($N^2 m^2$)', r'Nonzero elements'],
                    show=True, mark_trials=None)

if part2:
    # Now, I want to plot a plot of the currents
    path1 = os.path.join(os.getcwd(), 'plot_data', '_w5', 'currents_nl', 'currents70.pkl')
    path2 = os.path.join(os.getcwd(), 'plot_data', '_w5', 'currents_50_nl', 'currents70.pkl')
    path3 = os.path.join(os.getcwd(), 'plot_data', '_w5', 'currents_d_nl', 'currents60.pkl')
    path4 = os.path.join(os.getcwd(), 'plot_data', '_w5', 'currents_nl', 'currents_simulation70.pkl')
    path5 = os.path.join(os.getcwd(), 'plot_data', '_w5', 'currents_50_nl', 'currents_simulation70.pkl')
    path6 = os.path.join(os.getcwd(), 'plot_data', '_w5', 'currents_d_nl', 'currents_simulation60.pkl')
    datalist = [path1, path2, path3, path4, path5, path6]
    plot_tiled_curr(datalist)

    path1 = os.path.join(os.getcwd(), 'plot_data', '_w5', 'torque_nl', 'torque5.pkl')
    path2 = os.path.join(os.getcwd(), 'plot_data', '_w5', 'torque_50_nl', 'torque10.pkl')
    path3 = os.path.join(os.getcwd(), 'plot_data', '_w5', 'torque_d_nl', 'torque50.pkl')
    path4 = os.path.join(os.getcwd(), 'plot_data', '_w5', 'ump_nl', 'ump0.pkl')
    path5 = os.path.join(os.getcwd(), 'plot_data', '_w5', 'ump_50_nl', 'UMP100.pkl')
    path6 = os.path.join(os.getcwd(), 'plot_data', '_w5', 'ump_d_nl', 'UMP150.pkl')
    datalist = [path1, path2, path3, path4, path5, path6]
    plot_tiled_TF(datalist)

## to do: the optuna plots, and mark the selected models...

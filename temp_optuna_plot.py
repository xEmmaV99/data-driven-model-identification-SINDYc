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
    limit_list = [[[5e1, 2e3], [0, 300]],
                  [[5e1, 2e3], [0, 300]], # maybe not good
                  [[5e2, 2e3], [0, 300]], # this one is shifted
                  [[1e-5, 5e-4], [0, 100]],
                  [[4e-5, 5e-4], [0, 100]],
                  [[5e-4, 15e-4], [0, 130]], #somethig is wrong here
                  [[1.91, 1.9175], [0, 90]], # xlab is weird
                  [[3, 600], [0, 500]],
                  [[5e3, 4e4], [0, 800]]
                  ]
    marks = [[528,320,491],
             [857,626,928],
             [463, 985],
             [1557,1045],
             [936,849],
             [703,109],
             [306],
             [634, 988],
             [1077,1171]]


    for j, study in enumerate(opt_study_names):
        stud = optuna.load_study(study_name=None,
                                 storage="sqlite:///" + "optuna_studies/_w5/" + study + "-optuna-study.db")
        ### html interactive figure
        #plot_optuna_data("_w5/"+study+'-optuna-study')

        ### matplotlib figure
        plot_pareto(stud,
                    limits=limit_list[j],
                    logscale=True,
                    target_names=[r'Mean Squared Error '+ ['($A^2$)','($N^2 m^2$)','($N^2$)'][j//3], r'Nonzero elements'],
                    show=True, mark_trials=marks[j],
                    save_name = study)
        # for torque MSE : ($N^2 m^2$) but for UMP it is in N^2 and for currents in A^2

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

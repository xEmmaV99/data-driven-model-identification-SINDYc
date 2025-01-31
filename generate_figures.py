import optuna
from optimize_parameters import plot_pareto, plot_optuna_data
from source import *
import numpy as np
import os
from tabulate import tabulate

# plot pareto front
part1 = False
part2 = False
part3 = True
part4 = False
#todo add coefficient plot

p = os.path.join(os.getcwd(), "plot_data", "_w5")
path1 = os.path.join(p, "currents_nl", "currents70.pkl")
path2 = os.path.join(p, "currents_50_nl", "currents70.pkl")
path3 = os.path.join(p, "currents_d_nl", "currents70.pkl")
path4 = os.path.join(p, "currents_nl", "currents_simulation70.pkl")
path5 = os.path.join(p, "currents_50_nl", "currents_simulation70.pkl")
path6 = os.path.join(p, "currents_d_nl", "currents_simulation70.pkl")

path7 = os.path.join(p, "torque_nl", "torque5.pkl")  # good enough? 1 term different from the reference
path8 = os.path.join(p, "torque_50_nl", "torque40.pkl")  # torque10 also possible but worse or 40
path9 = os.path.join(p, "torque_d_nl", "torque50.pkl") # 50 also possible or 100
path10 = os.path.join(p, "ump_nl", "ump0.pkl")
path11 = os.path.join(p, "ump_50_nl", "UMP100.pkl")
path12 = os.path.join(p, "ump_d_nl", "UMP150.pkl")
datalist = [path1, path2, path3, path4, path5, path6, path7, path8, path9, path10, path11, path12]

modellist = ['currents_nonlinear_70', 'currents_nonlinear_50ecc_70', 'currents_nonlinear_dynamic_70',
                'torque_nonlinear_5', 'torque_nonlinear_50ecc_40', 'torque_nonlinear_dynamic_50ecc_50',
                'ump_nonlinear_0', 'ump_nonlinear_50ecc_100', 'ump_nonlinear_dynamic_50ecc_150']

if part1:
    opt_study_names = [
        "currentsnonlinear",
        "currentsnonlinear_50ecc",
        "currentsnonlinear_dynamic_50ecc",
        "torquenonlinear",
        "torquenonlinear_50ecc",
        "torquenonlinear_dynamic_50ecc",
        "umpnonlinear",
        "umpnonlinear_50ecc",
        "umpnonlinear_dynamic_50ecc",
    ]
    limit_list = [
        [[5e1, 2e3], [0, 300]],
        [[5e1, 2e3], [0, 400]],
        [[5e2, 2e3], [0, 300]],
        [[1e-5, 5e-4], [0, 100]],
        [[4e-5, 5e-4], [0, 100]],
        [[5e-4, 15e-4], [0, 130]],
        [[1.911, 1.915], [0, 90]],
        [[3, 600], [0, 500]],
        [[5e3, 4e4], [0, 800]],
    ]

    # the marks correspond to the trials indices that will be marked
    # in de pareto plot. In order to find an index, open the paretoplot
    # via the buildin optuna plot function, in html format
    marks = [
        [528],#, 320, 491],
        [857],#, 626, 928],
        [946], # new pareto#[463, 985],
        [1557],#, 1045],
        [849],#[936],#, 849],
        [703],#, 109],
        [306],
        [988],#[634],#, 988],
        [1077],#, 1171],
    ]

    for j, study in enumerate(opt_study_names):
        stud = optuna.load_study(
            study_name=None,
            storage="sqlite:///" + "optuna_studies/_w5/" + study + "-optuna-study.db",
        )
        ### html interactive figure
        # plot_optuna_data("_w5/"+study+'-optuna-study')

        ### matplotlib figure
        plot_pareto(
            stud,
            limits=limit_list[j],
            logscale=True if study != "umpnonlinear" else False,
            target_names=[
                r"Mean Squared Error " + ["($\frac{A^2}{s^2}$)", "($N^2 m^2$)", "($N^2$)"][j // 3],
                r"Nonzero elements",
            ],
            show=False,
            mark_trials=marks[j],
            save_name=study,
        )
        # for torque MSE : ($N^2 m^2$) but for UMP it is in N^2 and for currents in A^2

if part2:
    plot_tiled_curr(datalist[:6], show=False, save_name='currents')
    plot_tiled_TF(datalist[-6:], show=True, save_name='torque_ump')

if part3:
    def get_sparsity(models):
        spar = np.zeros((6,3))
        for j,model in enumerate(models):
            model = load_model('_w5/'+model + '_model')
            if j < 3:
                spar[:3, j] = np.count_nonzero(model.coefficients(), axis = 1)
            elif j < 6:
                spar[3, j%3] = np.count_nonzero(model.coefficients())
            else:
                spar[4:, j%3] = np.count_nonzero(model.coefficients(), axis = 1)
        return spar
    # Calculate the mean ABSOLUTE error for all the models compared to their reference
    # save it in a big matrix with shape (10,3)
    MAE = np.zeros((10, 3))
    RMSE = np.zeros((10, 3))
    for i, path in enumerate(datalist[:6]):
        # datalist contains the plotdata -> So result and the reference of the models
        with open(path, "rb") as file:
            data = pkl.load(file)
            # data['plots'] has '0' and '1' corresponding to the currents and the reference
        m = np.mean(np.abs(data['plots']['0'][:, 1:] - data['plots']['1'][:, 1:]), axis=0)
        MAE[(0 + 3 * (i // 3)):3 + 3 * (i // 3), i % 3] = m
        rms = np.sqrt(np.mean((data['plots']['0'][:, 1:] - data['plots']['1'][:, 1:])**2, axis=0))
        RMSE[(0 + 3 * (i // 3)):3 + 3 * (i // 3), i % 3] = rms

    for i, path in enumerate(datalist[-6:-3]): # torque
        with open(path, "rb") as file:
            data = pkl.load(file) # '0' '1' and '2' where 1 is the reference
        m1 = np.mean(np.abs(data['plots']['0'][:, 1:] - data['plots']['1'][:, 1:]), axis=0) # MAE sindy
        m2 = np.mean(np.abs(data['plots']['2'][:, 1:] - data['plots']['1'][:, 1:]), axis=0) # MAE clarke model
        MAE[6:8, i % 3] = np.vstack((m1, m2)).T.flatten()
        rms1 = np.sqrt(np.mean((data['plots']['0'][:, 1:] - data['plots']['1'][:, 1:])**2, axis=0))
        rms2 = np.sqrt(np.mean((data['plots']['2'][:, 1:] - data['plots']['1'][:, 1:])**2, axis=0))
        RMSE[6:8, i % 3] = np.vstack((rms1, rms2)).T.flatten()

    for i, path in enumerate(datalist[-3:]): # ump
        with open(path, "rb") as file:
            data = pkl.load(file) # x, y, xy
        m1 = np.mean(np.abs(data['plots']['0'][:, 1] - data['plots']['2'][:, 1]), axis=0) # umpx
        m2 = np.mean(np.abs(data['plots']['1'][:, 1] - data['plots']['2'][:, 2]), axis=0) # umpy
        MAE[8:10, i % 3] = np.vstack((m1, m2)).T.flatten()
        rms1 = np.sqrt(np.mean((data['plots']['0'][:, 1] - data['plots']['2'][:, 1])**2, axis=0))
        rms2 = np.sqrt(np.mean((data['plots']['1'][:, 1] - data['plots']['2'][:, 2])**2, axis=0))
        RMSE[8:10, i % 3] = np.vstack((rms1, rms2)).T.flatten()

    # print a table with the MAE, column names are 'no ecc', '50 ecc', 'dynamic 50 ecc'
    # and rownames are did, diq, di0, id, iq, i0, T, Tc, umpx, umpy
    # the MAE contains the values in the correct order

    v = get_sparsity(modellist)
    V = np.zeros((10, 3))
    V[:3, :] = v[:3, :]
    V[3:6, :] = v[:3, :] # copy
    V[6] = v[3,:]
    V[8:] = v[-2:,:]
    combined = np.array([[f"{MAE[i, j]:.3e} ({int(V[i, j])})" if i not in [3,4,5,7] else f"{MAE[i,j]:.3e}" for j in range(MAE.shape[1])] for i in range(MAE.shape[0])])
    combined_rms = np.array([[f"{RMSE[i, j]:.3e} ({int(V[i, j])})" if i not in [3,4,5,7] else f"{RMSE[i,j]:.3e}" for j in range(RMSE.shape[1])] for i in range(RMSE.shape[0])])

    print(tabulate(combined, headers=["MAE", "no ecc", "50 ecc", "dynamic ecc"], showindex=[
        r"$\frac{\partial i_d}{\partial t} [A/s]$",
        r"$\frac{\partial i_q}{\partial t} [A/s]$",
        r"$\frac{\partial i_0}{\partial t} [A/s]$",
        r"$i_d [A]$",
        r"$i_q [A]$",
        r"$i_0 [A]$",
        r"$T [Nm]$",
        r"$T_c [Nm]$",
        r"$F_x [N]$",
        r"$F_y [N]$"],
                   tablefmt = 'latex_raw'))

if part4:
    #plot the coefficients of the models
    for i, model in enumerate(modellist):
        model = load_model('_w5/'+model + '_model')
        plot_coefs2(model, show=False, save_name="test"+str(i))

import optuna
from optimize_parameters import plot_pareto, plot_optuna_data
from source import *
import numpy as np
import os
from tabulate import tabulate

p = os.path.join(os.getcwd(), "plot_data", "_feb")
path1 = os.path.join(p, "currents70.pkl")
path2 = os.path.join(p, "currents_e_70.pkl")
path3 = os.path.join(p, "currents_ed_70.pkl")
path4 = os.path.join(p, "currents_simulation70.pkl")
path5 = os.path.join(p, "currents_e_simulation70.pkl")
path6 = os.path.join(p, "currents_ed_simulation70.pkl")

path7 = os.path.join(p,  "Torque5.pkl")
path8 = os.path.join(p,  "Torque_e_40.pkl")
path9 = os.path.join(p,  "Torque_ed_40.pkl")

path10 = os.path.join(p, "UMP0.pkl")
path11 = os.path.join(p, "UMP100.pkl")
path12 = os.path.join(p, "UMP_bad_160.pkl")
datalist = [path1, path2, path3, path4, path5, path6, path7, path8, path9, path10, path11, path12]

modellist = ['currents_70', 'currents_e_70', 'currents_ed_70',
                'torque_5', 'torque_e_40', 'torque_ed_50',
                'ump_0', 'ump_e_100', 'ump_ed_150']
##todo: change model names to correct model names!

def get_sparsity(models):
    spar = np.zeros((6,3))
    for j,model in enumerate(models):
        model = load_model('_feb/'+model + '_model')
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

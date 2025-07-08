"""
Main file to generate the 16 optuna studies
"""
import os
from optimize_parameters import optimize_parameters
from optimize_parameters import plot_optuna_data

if __name__ == "__main__":
    print("Note that ecc input is off. If results are worse, this might be the reason.")
    linear = 'test_run'
    n_cores = 1
    n_trials = int(500/n_cores)

    seed = 68

    if linear == 'test_run':
        data = os.path.join("train-data", "07-31-nonlin", "IMMEC_nonlinear-0ecc_5.0sec.npz")
        # 9) nonlinear UMP
        print("test run for nonlinear data, trivial UMP")
        optimize_parameters(data, mode='ump', additional_name="nonlinear", n_jobs=n_cores, n_trials=n_trials,
                            seed=seed)

    elif linear:
        data = os.path.join("train-data", "07-29-default", "IMMEC_0ecc_5.0sec.npz")
        # 1) linear currents
        print("1")
        optimize_parameters(data, mode='currents', additional_name="linear", n_jobs=n_cores, n_trials=n_trials, seed=seed)
        # 2) linear torque
        print("2")
        optimize_parameters(data, mode='torque', additional_name="linear", n_jobs=n_cores, n_trials=n_trials, seed=seed)

        data = os.path.join("train-data", "07-31-ecc-50", "IMMEC_50ecc_5.0sec.npz")
        # 3) linear currents 50% ecc
        print("3")
        optimize_parameters(data, mode='currents', additional_name="linear_50ecc", n_jobs=n_cores, n_trials=n_trials, seed=seed)
        # 4) linear torque 50% ecc
        print("4")
        optimize_parameters(data, mode='torque', additional_name="linear_50ecc", n_jobs=n_cores, n_trials=n_trials, seed=seed)
        # 5) linear ump 50% ecc
        print("5")
        optimize_parameters(data, mode='ump', additional_name="linear_50ecc", n_jobs=n_cores, n_trials=n_trials, seed=seed)

        data = os.path.join("train-data", "08-09", "IMMEC_dynamic_50ecc_5.0sec.npz")
        # 6) linear currents dynamic ecc
        print("6")
        optimize_parameters(data, mode='currents', additional_name="linear_dynamic_50ecc", n_jobs=n_cores, n_trials=n_trials, seed=seed)
        # 7) linear torque dynamic ecc
        print("7")
        optimize_parameters(data, mode='torque', additional_name="linear_dynamic_50ecc", n_jobs=n_cores, n_trials=n_trials, seed=seed)
        # 8) linear ump dynamic ecc
        print("8")
        optimize_parameters(data, mode='ump', additional_name="linear_dynamic_50ecc", n_jobs=n_cores, n_trials=n_trials, seed=seed)

    else:
        data = os.path.join("train-data", "07-31-nonlin", "IMMEC_nonlinear-0ecc_5.0sec.npz")
        # 9) nonlinear currents
        print("1")
        optimize_parameters(data, mode='currents', additional_name="nonlinear", n_jobs=n_cores, n_trials=n_trials, seed=seed)
        # 10) nonlinear torque
        print("2")
        optimize_parameters(data, mode='torque', additional_name="nonlinear", n_jobs=n_cores, n_trials=n_trials, seed=seed)

        data = os.path.join("train-data", "07-31-nonlin50", "IMMEC_nonlinear-50ecc_5.0sec.npz")
        # 11) nonlinear currents 50% ecc
        print("3")
        optimize_parameters(data, mode='currents', additional_name="nonlinear_50ecc", n_jobs=n_cores, n_trials=n_trials, seed=seed)
        # 12) nonlinear torque 50% ecc
        print("4")
        optimize_parameters(data, mode='torque', additional_name="nonlinear_50ecc", n_jobs=n_cores, n_trials=n_trials, seed=seed)
        # 13) nonlinear ump 50% ecc
        print("5")
        optimize_parameters(data, mode='ump', additional_name="nonlinear_50ecc", n_jobs=n_cores, n_trials=n_trials, seed=seed)

        data = os.path.join("train-data", "08-16", "IMMEC_dynamic_nonlinear_5.0sec.npz")
        # 14) nonlinear currents dynamic ecc
        print("6")
        optimize_parameters(data, mode='currents', additional_name="nonlinear_dynamic_50ecc", n_jobs=n_cores, n_trials=n_trials, seed=seed)
        # 15) nonlinear torque dynamic ecc
        print("7")
        optimize_parameters(data, mode='torque', additional_name="nonlinear_dynamic_50ecc", n_jobs=n_cores, n_trials=n_trials, seed=seed)
        # 16) nonlinear ump dynamic ecc
        print("8")
        optimize_parameters(data, mode='ump', additional_name="nonlinear_dynamic_50ecc", n_jobs=n_cores, n_trials=n_trials, seed=seed)


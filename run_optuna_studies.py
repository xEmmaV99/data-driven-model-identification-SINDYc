import os
from optimize_parameters import optimize_parameters
from optimize_parameters import plot_optuna_data
# optimize_parameters(path_to_data_files:str, mode:str='torque', additional_name:str="", n_jobs:int = 1, n_trials:int = 100)

if __name__ == "__main__":
    linear = True
    n_cores = 1
    n_trials = 500

    if linear:
        plot_optuna_data("currentslinear_dynamic_50ecc-optuna-study")

        data = os.path.join("train-data", "08-13", "IMMEC_50ecc_linear_5.0sec.npz")
        print("6") #433
        optimize_parameters(data, mode='currents', additional_name="linear_dynamic_50ecc", n_jobs=n_cores,
                            n_trials=n_trials, ecc_input=True)

        '''
        data = os.path.join("train-data", "07-29-default", "IMMEC_0ecc_5.0sec.npz")
        # 1) linear currents
        print("1")
        optimize_parameters(data, mode='currents', additional_name="linear", n_jobs=n_cores, n_trials=n_trials, ecc_input=False)
        # 2) linear torque
        print("2")
        optimize_parameters(data, mode='torque', additional_name="linear", n_jobs=n_cores, n_trials=n_trials, ecc_input=False)

        data = os.path.join("train-data", "07-31-ecc-50", "IMMEC_50ecc_5.0sec.npz")
        # 3) linear currents 50% ecc
        print("3")
        optimize_parameters(data, mode='currents', additional_name="linear_50ecc", n_jobs=n_cores, n_trials=n_trials, ecc_input=False)
        # 4) linear torque 50% ecc
        print("4")
        optimize_parameters(data, mode='torque', additional_name="linear_50ecc", n_jobs=n_cores, n_trials=n_trials, ecc_input=False)
        # 5) linear ump 50% ecc
        print("5")
        optimize_parameters(data, mode='ump', additional_name="linear_50ecc", n_jobs=n_cores, n_trials=n_trials, ecc_input=False)

        data = os.path.join("train-data", "08-09", "IMMEC_dynamic_50ecc_5.0sec.npz")
        # 6) linear currents dynamic ecc
        print("6")
        optimize_parameters(data, mode='currents', additional_name="linear_dynamic_50ecc", n_jobs=n_cores, n_trials=n_trials, ecc_input=True)
        # 7) linear torque dynamic ecc
        print("7")
        optimize_parameters(data, mode='torque', additional_name="linear_dynamic_50ecc", n_jobs=n_cores, n_trials=n_trials, ecc_input=True)
        # 8) linear ump dynamic ecc
        print("8")
        optimize_parameters(data, mode='ump', additional_name="linear_dynamic_50ecc", n_jobs=n_cores, n_trials=n_trials, ecc_input=True)
        '''
    else:
        data = os.path.join("train-data", "07-31-nonlin", "IMMEC_nonlinear-0ecc_5.0sec.npz")
        # 9) nonlinear currents
        print("1")
        optimize_parameters(data, mode='currents', additional_name="nonlinear", n_jobs=n_cores, n_trials=n_trials, ecc_input=False)
        # 10) nonlinear torque
        print("2")
        optimize_parameters(data, mode='torque', additional_name="nonlinear", n_jobs=n_cores, n_trials=n_trials, ecc_input=False)

        data = os.path.join("train-data", "07-31-nonlin50", "IMMEC_nonlinear-50ecc_5.0sec.npz")
        # 11) nonlinear currents 50% ecc
        print("3")
        optimize_parameters(data, mode='currents', additional_name="nonlinear_50ecc", n_jobs=n_cores, n_trials=n_trials, ecc_input=False)
        # 12) nonlinear torque 50% ecc
        print("4")
        optimize_parameters(data, mode='torque', additional_name="nonlinear_50ecc", n_jobs=n_cores, n_trials=n_trials, ecc_input=False)
        # 13) nonlinear ump 50% ecc
        print("5")
        optimize_parameters(data, mode='ump', additional_name="nonlinear_50ecc", n_jobs=n_cores, n_trials=n_trials, ecc_input=False)

        data = os.path.join("train-data", "08-16", "IMMEC_dynamic_nonlinear_5.0sec.npz")
        # 14) nonlinear currents dynamic ecc
        print("6")
        optimize_parameters(data, mode='currents', additional_name="nonlinear_dynamic_50ecc", n_jobs=n_cores, n_trials=n_trials, ecc_input=True)
        # 15) nonlinear torque dynamic ecc
        print("7")
        optimize_parameters(data, mode='torque', additional_name="nonlinear_dynamic_50ecc", n_jobs=n_cores, n_trials=n_trials, ecc_input=True)
        # 16) nonlinear ump dynamic ecc
        print("8")
        optimize_parameters(data, mode='ump', additional_name="nonlinear_dynamic_50ecc", n_jobs=n_cores, n_trials=n_trials, ecc_input=True)


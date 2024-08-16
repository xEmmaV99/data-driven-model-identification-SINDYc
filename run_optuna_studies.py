import os
from optimize_parameters import optimize_parameters
# optimize_parameters(path_to_data_files:str, mode:str='torque', additional_name:str="", n_jobs:int = 1, n_trials:int = 100)

linear = True

if linear:
    data = os.path.join("train-data", "07-29-default", "IMMEC_0ecc_5.0sec.npz")
    # 1) linear currents
    optimize_parameters(data, mode='currents', additional_name="linear", n_jobs=2, n_trials=1000, ecc_input=False)
    # 2) linear torque
    optimize_parameters(data, mode='torque', additional_name="linear", n_jobs=2, n_trials=1000, ecc_input=False)

    data = os.path.join("train-data", "07-31-ecc-50", "IMMEC_50ecc_5.0sec.npz")
    # 3) linear currents 50% ecc
    optimize_parameters(data, mode='currents', additional_name="linear_50ecc", n_jobs=2, n_trials=1000, ecc_input=False)
    # 4) linear torque 50% ecc
    optimize_parameters(data, mode='torque', additional_name="linear_50ecc", n_jobs=2, n_trials=1000, ecc_input=False)
    # 5) linear ump 50% ecc
    optimize_parameters(data, mode='ump', additional_name="linear_50ecc", n_jobs=2, n_trials=1000, ecc_input=False)

    data = os.path.join("train-data", "08-13", "IMMEC_50ecc_linear_5.0sec.npz")
    # 6) linear currents dynamic ecc
    optimize_parameters(data, mode='currents', additional_name="linear_dynamic_50ecc", n_jobs=2, n_trials=1000, ecc_input=True)
    # 7) linear torque dynamic ecc
    optimize_parameters(data, mode='torque', additional_name="linear_dynamic_50ecc", n_jobs=2, n_trials=1000, ecc_input=True)
    # 8) linear ump dynamic ecc
    optimize_parameters(data, mode='ump', additional_name="linear_dynamic_50ecc", n_jobs=2, n_trials=1000, ecc_input=True)

else:
    data = os.path.join("train-data", "07-31-nonlin", "IMMEC_nonlinear-0ecc_5.0sec.npz")
    # 9) nonlinear currents
    optimize_parameters(data, mode='currents', additional_name="nonlinear", n_jobs=2, n_trials=1000, ecc_input=False)
    # 10) nonlinear torque
    optimize_parameters(data, mode='torque', additional_name="nonlinear", n_jobs=2, n_trials=1000, ecc_input=False)

    data = os.path.join("train-data", "07-31-nonlin50", "IMMEC_nonlinear-50ecc_5.0sec.npz")
    # 11) nonlinear currents 50% ecc
    optimize_parameters(data, mode='currents', additional_name="nonlinear_50ecc", n_jobs=2, n_trials=1000, ecc_input=False)
    # 12) nonlinear torque 50% ecc
    optimize_parameters(data, mode='torque', additional_name="nonlinear_50ecc", n_jobs=2, n_trials=1000, ecc_input=False)
    # 13) nonlinear ump 50% ecc
    optimize_parameters(data, mode='ump', additional_name="nonlinear_50ecc", n_jobs=2, n_trials=1000, ecc_input=False)

    data = os.path.join("train-data", "08-16", "IMMEC_dynamic_nonlin_50ecc_5.0sec.npz")
    # 14) nonlinear currents dynamic ecc
    optimize_parameters(data, mode='currents', additional_name="nonlinear_dynamic_50ecc", n_jobs=2, n_trials=1000, ecc_input=True)
    # 15) nonlinear torque dynamic ecc
    optimize_parameters(data, mode='torque', additional_name="nonlinear_dynamic_50ecc", n_jobs=2, n_trials=1000, ecc_input=True)
    # 16) nonlinear ump dynamic ecc
    optimize_parameters(data, mode='ump', additional_name="nonlinear_dynamic_50ecc", n_jobs=2, n_trials=1000, ecc_input=True)


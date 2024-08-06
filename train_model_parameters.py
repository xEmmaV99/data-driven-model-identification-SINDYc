from optimize_parameters import parameter_search, optimize_parameters, plot_optuna_data
from source import *
from train_model_source import make_model, simulate_model

#path_to_data_files = os.path.join(os.getcwd(), 'train-data', '07-29-default', 'IMMEC_0ecc_5.0sec.npz')
#path_to_data_files = os.path.join(os.getcwd(), 'train-data', '07-31-ecc-50', 'IMMEC_50ecc_5.0sec.npz')
path_to_data_files = os.path.join(os.getcwd(), 'train-data', '07-31-nonlin', 'IMMEC_nonlinear-0ecc_5.0sec.npz')


# mode is either "currents", "torque" or "ump" (TO BE IMPLEMENTED W_mag)
# Creates an optuna study to optimize the parameters of the model

optimize_parameters(path_to_data_files, mode="currents", additional_name="Nonlinear-special")


# Plot optuna data
#plot_optuna_data('currentsNonlinear-special-optuna-study')


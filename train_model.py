from optimize_parameters import parameter_search, optimize_parameters, plot_optuna_data
from source import *
from train_model_source import make_model, simulate_model

#path_to_data_files = os.path.join(os.getcwd(), 'train-data', '07-29-default', 'IMMEC_0ecc_5.0sec.npz')
#path_to_data_files = os.path.join(os.getcwd(), 'train-data', '07-31-ecc-50', 'IMMEC_50ecc_5.0sec.npz')

path_to_data_files = os.path.join(os.getcwd(), 'train-data', '07-31-nonlin', 'IMMEC_nonlinear-0ecc_5.0sec.npz')


# mode is either "currents", "torque" or "ump" (TO BE IMPLEMENTED W_mag)
# Creates an optuna study to optimize the parameters of the model
### PART 1: OPTIMIZE PARAMETERS

#optimize_parameters(path_to_data_files, mode="currents", additional_name="Nonlinear-special")
#plot_optuna_data('currentsNonlinear-special-optuna-study')

### PART 2: TRAIN MODEL

'''make_model(path_to_data_files, modeltype = "currents", optimizer = "sr3",
           nmbr_of_train=-1, lib="nonlinear_terms",
               alpha=None, nu=9.4, lamb=7.16e-6, modelname="example_A_currents")

make_model(path_to_data_files, modeltype = "currents", optimizer = "sr3",
           nmbr_of_train=-1, lib="poly_2nd_order",
               alpha=None, nu=2.8e-12, lamb=6.8e-5, modelname="example_B_currents")
'''

### PART 3: SIMULATE MODEL
path_to_test_file = os.path.join(os.getcwd(), 'test-data', '08-06', 'IMMEC_nonlin_0ecc_5.0sec.npz')

pr, test = simulate_model("example_A_currents_model", path_to_test_file, modeltype="currents", do_time_simulation=False)
prB, testB = simulate_model("example_B_currents_model", path_to_test_file, modeltype="currents", do_time_simulation=False)
plot_fourier(test, pr, dt = 1e-4, tmax = 5.0)
plot_fourier(testB, prB, dt = 1e-4, tmax = 5.0)

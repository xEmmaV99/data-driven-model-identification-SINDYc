from optimize_parameters import parameter_search, optimize_parameters, plot_optuna_data
from source import *
from train_model_source import make_model, simulate_model

do_part1 = False
do_part15 = True
do_part2 = False
do_part3 = True

### DATA TRAINING FILES
path_to_data_files = os.path.join(os.getcwd(), 'train-data', '07-29-default', 'IMMEC_0ecc_5.0sec.npz')
#path_to_data_files = os.path.join(os.getcwd(), 'train-data', '07-31-ecc-50', 'IMMEC_50ecc_5.0sec.npz')
#path_to_data_files = os.path.join(os.getcwd(), 'train-data', '07-31-nonlin', 'IMMEC_nonlinear-0ecc_5.0sec.npz')


### TEST FILES
#path_to_test_file = os.path.join(os.getcwd(), 'test-data', '08-07', 'IMMEC_nonlin_0ecc_5.0sec.npz')
path_to_test_file = os.path.join(os.getcwd(), 'test-data', '07-29-default', 'IMMEC_0ecc_5.0sec.npz')


### PART 1: OPTIMIZE PARAMETERS
if do_part1:
    # mode is either "currents", "torque" or "ump" (TO BE IMPLEMENTED W_mag)
    # Creates an optuna study to optimize the parameters of the
    optimize_parameters(path_to_data_files, mode="currents", additional_name="Nonlinear-special")

if do_part15:
    plot_optuna_data('currents-optuna-study', dirs = 'w3-presentation-0208//')

### PART 2: TRAIN MODEL
if do_part2:
    # Create a model with the optimized parameters
    make_model(path_to_data_files, modeltype = "currents", optimizer = "lasso",
               nmbr_of_train=-1, lib="custom",
                   alpha=10.77, nu=None, lamb=None, modelname="linear_example_1_currents")

    make_model(path_to_data_files, modeltype = "currents", optimizer = "sr3",
               nmbr_of_train=-1, lib="poly_2nd_order",
                   alpha=None, nu=4.88e-9, lamb=1.0193e-5, modelname="linear_example_2_currents")

    make_model(path_to_data_files, modeltype = "currents", optimizer = "sr3",
               nmbr_of_train=-1, lib="custom",
                   alpha=None, nu=6.84e-10, lamb=0.00269, modelname="linear_example_3_currents") #this one is really bad

    make_model(path_to_data_files, modeltype = "currents", optimizer = "sr3",
               nmbr_of_train=-1, lib="custom",
                   alpha=None, nu=1.145e-11, lamb=1.017e-5, modelname="linear_example_4_currents")
    '''
    make_model(path_to_data_files, modeltype = "currents", optimizer = "sr3",
               nmbr_of_train=-1, lib="nonlinear_terms",
                   alpha=None, nu=9.4, lamb=7.16e-6, modelname="example_A_currents")

    make_model(path_to_data_files, modeltype = "currents", optimizer = "sr3",
               nmbr_of_train=-1, lib="poly_2nd_order",
                   alpha=None, nu=2.8e-12, lamb=6.8e-5, modelname="example_B_currents")
    '''

### PART 3: SIMULATE MODEL
if do_part3:
    models = ["example_A_currents", "example_B_currents"]
    models = ["linear_example_2_currents", "linear_example_3_currents"]
    for m in models:
        # simulate the model and plot the results
        pr, test = simulate_model(m+'_model', path_to_test_file, modeltype="currents", do_time_simulation=True, show=False)

        plot_fourier(test, pr, dt = 5e-5, tmax = 5.0, show=False)

    plt.show()


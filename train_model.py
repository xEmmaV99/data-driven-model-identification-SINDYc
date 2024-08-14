import os

from optimize_parameters import parameter_search, optimize_parameters, plot_optuna_data
from source import *
from train_model_source import make_model, simulate_model

do_part1 = False
do_part2 = False
do_part3 = True
do_part4 = False

### DATA TRAINING FILES
#path_to_data_files = os.path.join(os.getcwd(), 'train-data', '07-29-default', 'IMMEC_0ecc_5.0sec.npz')
#path_to_data_files = os.path.join(os.getcwd(), 'train-data', '07-31-ecc-50', 'IMMEC_50ecc_5.0sec.npz')
#path_to_data_files = os.path.join(os.getcwd(), 'train-data', '07-31-nonlin', 'IMMEC_nonlinear-0ecc_5.0sec.npz')
#path_to_data_files = os.path.join(os.getcwd(), 'train-data', '07-31-nonlin50', 'IMMEC_nonlinear-50ecc_5.0sec.npz')
#path_to_data_files = os.path.join(os.getcwd(), 'train-data', 'ecc_random_direction', 'IMMEC_lin_ecc_randomecc_5.0sec.npz')
path_to_data_files = os.path.join(os.getcwd(), 'train-data', '08-09', 'IMMEC_dynamic_50ecc_5.0sec.npz')
#path_to_data_files = os.path.join(os.getcwd(), 'train-data', '08-13', 'IMMEC_default_linear_5.0sec.npz')

### TEST FILES
#path_to_test_file = os.path.join(os.getcwd(), 'test-data', '08-13', 'IMMEC_default_linear_5.0sec.npz')
path_to_test_file = os.path.join(os.getcwd(), 'test-data', '08-09', 'IMMEC_dynamic_50ecc_5.0sec.npz')
#path_to_test_file = os.path.join(os.getcwd(), 'test-data', '08-07', 'IMMEC_nonlin_0ecc_5.0sec.npz')
#path_to_test_file = os.path.join(os.getcwd(), 'test-data', '07-29-default', 'IMMEC_0ecc_5.0sec.npz')
#path_to_test_file = os.path.join(os.getcwd(), 'test-data', '08-07', 'IMMEC_lin_ecc_randomecc_5.0sec.npz')
#path_to_test_file = os.path.join(os.getcwd(), 'test-data', '08-05', 'IMMEC_50eccecc_5.0sec.npz')
#path_to_test_file = os.path.join(os.getcwd(), 'test-data', '08-02', 'IMMEC_y50ecc_5.0sec.npz')

plot_immec_data(path_to_test_file)

### PART 1: OPTIMIZE PARAMETERS
if do_part1:
    # mode is either "currents", "torque" or "ump" (TO BE IMPLEMENTED W_mag)
    # Creates an optuna study to optimize the parameters of the
    optimize_parameters(path_to_data_files, mode="wcoe", additional_name="W_linear")

### PART 2: plot the optuna study to choose the hyperparameters
if do_part2:
    #plot_optuna_data('currents-optuna-study', dirs = 'w3-presentation-0208//')
    #plot_optuna_data('currentsLinear-specific-optuna-study')
    #plot_optuna_data('ump_dynamic-optuna-study')
    plot_optuna_data('W_linear-optuna-study')

### PART 3: TRAIN MODEL
if do_part3:
    name = make_model(path_to_data_files, modeltype='torque', optimizer='STLSQ',
               nmbr_of_train=-1, lib='poly_2nd_order', alpha = 0.001, threshold = 0.1,
               modelname='T_ecc')
    simulate_model(name, path_to_test_file, modeltype='torque')

    '''
    make_model(path_to_data_files, modeltype='torque', optimizer='lasso',
               nmbr_of_train=-1, lib='linear-specific', alpha=1.23, modelname='torque_linear_4')
    '''
    '''
    make_model(path_to_data_files, modeltype='torque', optimizer='sr3',
               nmbr_of_train=-1, lib='torque', nu=0.00029, lamb=0.00578,
               modelname='torque_linear')
    make_model(path_to_data_files, modeltype='torque', optimizer='sr3',
               nmbr_of_train=-1, lib='torque', nu=4e-6, lamb=0.0020,
               modelname='torque_linear_2')
    make_model(path_to_data_files, modeltype='torque', optimizer='sr3',
               nmbr_of_train=-1, lib='torque', nu=4.9e-7, lamb=9.8e-5,
               modelname='torque_linear_3')
    '''
    '''
    make_model(path_to_data_files, modeltype='currents', optimizer='lasso',
               nmbr_of_train=-1, lib='nonlinear_terms', alpha=86.47,
               modelname= 'currents_nonlinear_50ecc_60')
    make_model(path_to_data_files, modeltype='currents', optimizer='sr3',
               nmbr_of_train=-1, lib='linear-specific', nu=1.075e-11,lamb= 0.00226,
               modelname= 'currents_nonlinear_50ecc_90')
    make_model(path_to_data_files, modeltype='currents', optimizer='sr3',
               nmbr_of_train=-1, lib='poly_2nd_order', nu=1.424e-8, lamb=4.5e-5,
               modelname='currents_nonlinear_50ecc_200')
    make_model(path_to_data_files, modeltype='currents', optimizer='sr3',
               nmbr_of_train=-1, lib='nonlinear_terms', nu=2e-9, lamb=3.4e-5,
               modelname='currents_nonlinear_50ecc_550')
    '''
    '''
    make_model(path_to_data_files, modeltype="torque", optimizer= "sr3",
               nmbr_of_train=-1, lib="poly_2nd_order",
               alpha=None, nu=5.043e-7, lamb=3.8157e-10, modelname="torque_nonlinear_100coef")
    make_model(path_to_data_files, modeltype="torque", optimizer= "sr3",
               nmbr_of_train=-1, lib="torque",
               alpha=None, nu=3.416e-6, lamb=1.74e-6, modelname="torque_nonlinear_20coef")
    make_model(path_to_data_files, modeltype="torque", optimizer= "sr3",
               nmbr_of_train=-1, lib="nonlinear_terms",
               alpha=None, nu=0.002, lamb=1.36e-9, modelname="torque_nonlinear_280coef")
    ''''''
    make_model(path_to_data_files, modeltype = "currents", optimizer = "sr3",
               nmbr_of_train=-1, lib="linear-specific",
                   alpha=None, nu=4.0115e-7, lamb=1.1038e-9, modelname="linear_example_new_1_currents")

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
    
    make_model(path_to_data_files, modeltype = "currents", optimizer = "sr3",
               nmbr_of_train=-1, lib="nonlinear_terms",
                   alpha=None, nu=9.4, lamb=7.16e-6, modelname="example_A_currents")
    
    make_model(path_to_data_files, modeltype = "currents", optimizer = "sr3",
               nmbr_of_train=-1, lib="poly_2nd_order",
                   alpha=None, nu=2.8e-12, lamb=6.8e-5, modelname="example_B_currents")
    '''
### PART 4: SIMULATE MODEL
if do_part4:
    models = [ "example_A_currents"] #"example_A_currents",
    models = ["linear_example_new_1_currents", "linear_example_2_currents", "linear_example_3_currents"]
    models = ["currents_nonlinear"]
    models = ["torque_linear", "torque_linear_2"]
    models = ["W_lin", "W_lin_sparser"]
    models = ["W_lin_2"]
    pref = "0908//"
    pref = "w_linear//"

    for m in models:
        # simulate the model and plot the results
        pr, test = simulate_model(pref+m+'_model', path_to_test_file, modeltype="wcoe", do_time_simulation=False, show=True)
        plot_fourier(test, pr, dt = 1e-4, tmax = 5.0, show=False)

    plt.show()


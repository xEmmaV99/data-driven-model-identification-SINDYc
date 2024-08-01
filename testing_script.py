from source import *

from optimize_parameters import plot_optuna_data
#path = os.path.join(os.getcwd(), 'train-data', "07-30-ecc-BAD-TOBEDELETED","IMMEC_50ecc_5.0sec.npz")
path = os.path.join(os.getcwd(), 'test-data', '07-31-50ecc-load', 'IMMEC_50ecc_5.0sec.npz')
path = os.path.join(os.getcwd(), 'train-data', '07-31-ecc-90', 'IMMEC_90ecc_5.0sec.npz')
path = os.path.join(os.getcwd(), 'train-data', '07-31-ecc-50', 'IMMEC_50ecc_5.0sec.npz')
path = os.path.join(os.getcwd(), 'train-data', '07-31-nonlin', 'IMMEC_nonlinear-0ecc_5.0sec.npz')
path = os.path.join(os.getcwd(), 'train-data', '07-31-nonlin50', 'IMMEC_nonlinear-50ecc_5.0sec.npz')
path = os.path.join(os.getcwd(), 'train-data', '07-31-nonlin90', 'IMMEC_nonlinear-90ecc_5.0sec.npz')


plot_immec_data(path, simulation_number=10)

#plot_optuna_data('example-study-poly-vs-best')
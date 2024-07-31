from source import *
from param_optimizer import plot_optuna_data
path = os.path.join(os.getcwd(), 'train-data', "07-30-ecc-BAD-TOBEDELETED","IMMEC_50ecc_5.0sec.npz")
plot_immec_data(path, simulation_number=30)

#plot_optuna_data('example-study-poly-vs-best')
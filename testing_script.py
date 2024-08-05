from source import *
from optimize_parameters import plot_optuna_data
from train_model_currents import *

path_to_test_file = os.path.join(os.getcwd(), 'test-data', '07-29-default', 'IMMEC_0ecc_5.0sec.npz')

x_test_pr, x_test = simulate_currents('currents_model', path_to_test_file, do_time_simulation=False)

plot_fourier(x_test, x_test_pr, dt = 4e-5, tmax = 5.0)
test_plot_fourier()

import matplotlib.pyplot as plt

from source import *
from optimize_parameters import plot_optuna_data
from train_model_currents import *

path_to_test_file = os.path.join(os.getcwd(), 'test-data', '07-29-default', 'IMMEC_0ecc_5.0sec.npz')

x_pred, x_test = simulate_currents('currents_model_accurate', path_to_test_file, do_time_simulation=True)

plt.figure()
leg = ['dot i_d ref', 'dot i_d', 'dot i_q ref', 'dot i_q', 'dot i_0 ref', 'dot i_0']
leg = ['i_d ref', 'i_d', 'i_q ref', 'i_q', 'i_0 ref', 'i_0']
plot_fourier(x_test, x_pred, dt = 5e-5, tmax = 1.0, leg=leg)


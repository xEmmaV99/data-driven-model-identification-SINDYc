import matplotlib.pyplot as plt

from source import *
from optimize_parameters import plot_optuna_data


#model = load_model('linear_example_2_currents_model')
#plot_coefs2(model, show=True, log=True)
#plot_immec_data(os.path.join(os.getcwd(), 'test-data', '08-07', 'IMMEC_nonlin_ecc_randomecc_5.0sec.npz'))

plot_immec_data(os.path.join(os.getcwd(), 'test-data', '08-02', 'IMMEC_y50ecc_5.0sec.npz'))
plot_immec_data(os.path.join(os.getcwd(), 'test-data', '08-02', 'IMMEC_xy50ecc_5.0sec.npz'))
plot_immec_data(os.path.join(os.getcwd(), 'test-data', '08-08', 'IMMEC_50ecc_ecc_5.0sec.npz'))

#plot_optuna_data('currentsLinear-specific-optuna-study')

#p = os.path.join(os.getcwd(), 'plot_data','currents08-07_16-28-24.pkl')
#plot_data(p, show=True)

"""
path_to_test_file = os.path.join(os.getcwd(), 'test-data', '07-29-default', 'IMMEC_0ecc_5.0sec.npz')

x_pred, x_test = simulate('currents_model_accurate', path_to_test_file, do_time_simulation=True)

plt.figure()
leg = ['dot i_d ref', 'dot i_d', 'dot i_q ref', 'dot i_q', 'dot i_0 ref', 'dot i_0']
leg = ['i_d ref', 'i_d', 'i_q ref', 'i_q', 'i_0 ref', 'i_0']
plot_fourier(x_test, x_pred, dt = 5e-5, tmax = 1.0, leg=leg)
"""
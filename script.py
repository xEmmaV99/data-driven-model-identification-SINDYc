import matplotlib.pyplot as plt

from source import *
from optimize_parameters import plot_optuna_data
plot_immec_data(os.path.join(os.getcwd(), 'train-data', '08-09', 'IMMEC_dynamic_50ecc_5.0sec.npz'), simulation_number=0)

"""
path_to_data_files = os.path.join(os.getcwd(), 'train-data', '07-31-ecc-50', 'IMMEC_50ecc_5.0sec.npz')
DATA = prepare_data(path_to_data_files, number_of_trainfiles=-1)
library = get_custom_library_funcs("poly_2nd_order")
train = DATA['T_em_train']
train = DATA['xdot_train']
name = "Torque"
opt = ps.SR3(thresholder="l1", threshold=0.0001, nu=1)
model = ps.SINDy(optimizer=opt, feature_library=library,
                 feature_names=DATA['feature_names'])

print("Fitting model")
model.fit(DATA['x_train'], u=DATA['u_train'], t=None, x_dot=train)
model.print(precision=10)
model = ps.SINDy(optimizer=opt, feature_library=library,
                 feature_names=DATA['feature_names'])
model.fit(np.hstack((DATA['x_train'],DATA['u_train'])), t=None, x_dot=train)
model.print(precision=10)
"""

"""
#model = load_model('linear_example_2_currents_model')
#plot_coefs2(model, show=True, log=True)
#plot_immec_data(os.path.join(os.getcwd(), 'test-data', '08-07', 'IMMEC_nonlin_ecc_randomecc_5.0sec.npz'))

plot_immec_data(os.path.join(os.getcwd(), 'test-data', '08-02', 'IMMEC_y50ecc_5.0sec.npz'))
plot_immec_data(os.path.join(os.getcwd(), 'test-data', '08-02', 'IMMEC_xy50ecc_5.0sec.npz'))
plot_immec_data(os.path.join(os.getcwd(), 'test-data', '08-08', 'IMMEC_50ecc_ecc_5.0sec.npz'))

#plot_optuna_data('currentsLinear-specific-optuna-study')

#p = os.path.join(os.getcwd(), 'plot_data','currents08-07_16-28-24.pkl')
#plot_data(p, show=True)


path_to_test_file = os.path.join(os.getcwd(), 'test-data', '07-29-default', 'IMMEC_0ecc_5.0sec.npz')

x_pred, x_test = simulate('currents_model_accurate', path_to_test_file, do_time_simulation=True)

plt.figure()
leg = ['dot i_d ref', 'dot i_d', 'dot i_q ref', 'dot i_q', 'dot i_0 ref', 'dot i_0']
leg = ['i_d ref', 'i_d', 'i_q ref', 'i_q', 'i_0 ref', 'i_0']
plot_fourier(x_test, x_pred, dt = 5e-5, tmax = 1.0, leg=leg)
"""
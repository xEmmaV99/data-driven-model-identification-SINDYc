import matplotlib.pyplot as plt
from pysindy.utils import equations
from source import *
from optimize_parameters import plot_optuna_data
#plot_immec_data(os.path.join(os.getcwd(), 'train-data', '08-09', 'IMMEC_dynamic_50ecc_5.0sec.npz'), simulation_number=0)

# testing the difference method by using virtual work
def numerical_diff(TEST, W_model):
    dth = 0.001*2*np.pi
    pr1 = W_model.predict(TEST['x'], u = TEST['u'])
    pr1 = np.array(pr1)
    u = np.copy(TEST['u'])
    u[:,9] += dth
    u[:,9] = u[:,9] % (2*np.pi)
    pr2 = W_model.predict(TEST['x'], u = u)

    torque = (pr2 - pr1) / dth
    #torque = np.diff(pr1[1:,0], pr1[:-1,0]) / np.diff(TEST['u'][:,9])
    #plt.plot(u[:,9])
    #plt.plot(TEST['u'][:,9])
    #plt.show()
    return torque
    
plot_immec_data(os.path.join(os.getcwd(), 'test-data', '08-13', 'IMMEC_default_linear_5.0sec.npz'))

W_model = load_model('w_linear//w_model')
#print(W_model.equations(precision=10))

# model.equations is something else then the equations function form utils

TEST = prepare_data(os.path.join(os.getcwd(), 'test-data', '08-13', 'IMMEC_default_linear_5.0sec.npz'), test_data=True)
pr1 = W_model.predict(TEST['x'], u=TEST['u'])
pr1 = np.array(pr1)
tvec = TEST['t'][:,0,0]

# plt.figure()
# plt.plot(tvec, pr1)
# plt.plot(tvec, TEST['wcoe'], 'k:')
# plt.xlabel('Time (s)')
# plt.legend(['Predicted', 'True'])
# plt.title('Magnetic Coenergy')

fig, ax = plt.subplots()
l1 = ax.plot(tvec, pr1)
l2 = ax.plot(tvec, TEST['wcoe'], 'k:')
ax.set_xlabel('Time (s)')
plt.title('Magnetic Coenergy')
left, bottom, width, height = [0.4, 0.3, 0.4, 0.4]
ax_inset = fig.add_axes([left, bottom, width, height])
ax_inset.plot(tvec, pr1)
ax_inset.plot(tvec, TEST['wcoe'], 'k:')
ax_inset.set_xlim([1.450, 1.465])
ax_inset.set_ylim([1.2876, 1.2888])
ax.legend(['Predicted', 'True'])


plt.figure()
plt.plot(tvec, TEST['T_em'])
plt.title("T_em")
plt.xlabel('Time (s)')
plt.ylabel('Torque (Nm)')

# plt.figure()
# plt.plot(TEST['u'][:,9])
# plt.title("gamma")

plt.figure()
# if np.diff (u) is 0, replace by NaN
dtheta = np.diff(TEST['u'][:,9])
dtheta[np.isclose(dtheta, 0)] = np.nan
plt.plot(tvec[:-1], np.diff(TEST['wcoe']) / dtheta)
plt.xlabel('Time (s)')
plt.title(r"$\partial W/ \partial \gamma$")
plt.ylim([-0.25,0.25])
plt.ylabel('Torque ? (Nm)')
#torque = np.diff(pr1[:,0]) / dtheta
#plt.plot(torque)
plt.show()




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
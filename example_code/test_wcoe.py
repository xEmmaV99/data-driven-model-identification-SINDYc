from source import *

# testing the difference method by using virtual work
def numerical_diff(TEST, W_model):
    dth = 0.001 * 2 * np.pi
    pr1 = W_model.predict(TEST['x'], u=TEST['u'])
    pr1 = np.array(pr1)
    u = np.copy(TEST['u'])
    u[:, 9] += dth
    u[:, 9] = u[:, 9] % (2 * np.pi)
    pr2 = W_model.predict(TEST['x'], u=u)

    torque = (pr2 - pr1) / dth
    # torque = np.diff(pr1[1:,0], pr1[:-1,0]) / np.diff(TEST['u'][:,9])
    # plt.plot(u[:,9])
    # plt.plot(TEST['u'][:,9])
    # plt.show()
    return torque


plot_immec_data(os.path.join(os.getcwd(), 'test-data', '08-13', 'IMMEC_default_linear_5.0sec.npz'))

W_model = load_model('w_linear//w_model')
# print(W_model.equations(precision=10))
# model.equations is something else then the equations function form utils

TEST = prepare_data(os.path.join(os.getcwd(), 'test-data', '08-13', 'IMMEC_default_linear_5.0sec.npz'), test_data=True)
pr1 = W_model.predict(TEST['x'], u=TEST['u'])
pr1 = np.array(pr1)
tvec = TEST['t'][:, 0, 0]

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
dtheta = np.diff(TEST['u'][:, 9])
dtheta[np.isclose(dtheta, 0)] = np.nan
plt.plot(tvec[:-1], np.diff(TEST['wcoe']) / dtheta)
plt.xlabel('Time (s)')
plt.title(r"$\partial W/ \partial \gamma$")
plt.ylim([-0.25, 0.25])
plt.ylabel('Torque ? (Nm)')
# torque = np.diff(pr1[:,0]) / dtheta
# plt.plot(torque)
plt.show()
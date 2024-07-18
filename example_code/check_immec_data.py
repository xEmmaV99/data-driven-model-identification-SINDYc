import matplotlib.pyplot as plt

from source import *

path = '/data/data_files/IMMEC_history_80.0V_1.0sec.pkl'
with open(path, 'rb') as f:
    dataset = pkl.load(f)

# chekc the currents and voltages
plt.plot(dataset['time'], reference_abc_to_dq0(dataset['i_st']))
plt.figure()
#calculate x_dot
x_dot = np.diff(reference_abc_to_dq0(dataset['i_st']), axis=0) / dataset['time'][1]

plt.plot(dataset['time'][1:], x_dot)
plt.title('Time derivative of the currents with tolerance 1e-4 (default)')
plt.xlabel('time (s)')
plt.ylabel('$\dot{x}$')

# the derivs are indeed not nice.

# check with other data, smaller timestep 1e-5
path = '/data/IMMEC_history_soltol_80.0V_1.0sec.pkl'
with open(path, 'rb') as f:
    dataset = pkl.load(f)

# chekc the currents and voltages
plt.figure()
plt.plot(dataset['time'], reference_abc_to_dq0(dataset['i_st']))
plt.figure()
#calculate x_dot
x_dot = np.diff(reference_abc_to_dq0(dataset['i_st']), axis=0) / dataset['time'][1]

plt.plot(dataset['time'][1:], x_dot)
plt.title('Time derivative of the currents with tolerance 1e-5')
plt.xlabel('time (s)')
plt.ylabel('$\dot{x}$')
plt.show()

#######################
path = '/data/data_files/IMMEC_history_400.0V_1.0sec.pkl'
with open(path, 'rb') as f:
    dataset = pkl.load(f)

# chekc the currents and voltages
plt.plot(dataset['time'], reference_abc_to_dq0(dataset['i_st']))
plt.figure()
#calculate x_dot
x_dot = np.diff(reference_abc_to_dq0(dataset['i_st']), axis=0) / dataset['time'][1]

plt.plot(dataset['time'][1:], x_dot)
plt.title('Time derivative of the currents with tolerance 1e-4 (default)')
plt.xlabel('time (s)')
plt.ylabel('$\dot{x}$')

# the derivs are indeed not nice.

# check with other data, smaller timestep 1e-5
path = '/data/IMMEC_history_soltol_400.0V_1.0sec.pkl'
with open(path, 'rb') as f:
    dataset = pkl.load(f)

# chekc the currents and voltages
plt.figure()
plt.plot(dataset['time'], reference_abc_to_dq0(dataset['i_st']))
plt.figure()
#calculate x_dot
x_dot = np.diff(reference_abc_to_dq0(dataset['i_st']), axis=0) / dataset['time'][1]

plt.plot(dataset['time'][1:], x_dot)
plt.title('Time derivative of the currents with tolerance 1e-5')
plt.xlabel('time (s)')
plt.ylabel('$\dot{x}$')
plt.show()
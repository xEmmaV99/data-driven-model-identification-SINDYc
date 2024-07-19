import numpy as np

from source import *
from prepare_data import *
import os

##ALTERNATIVE TORQUE PREDICTION

cwd = os.getcwd()
path_to_data = os.path.join(cwd, 'data/07-19/')

T_train, x_train, u_train, T_val, x_val, u_val, TESTDATA = prepare_data(path_to_data, Torque=True)
# u_val is already in dq0 reference frame

u_test = TESTDATA['u']
V = u_test[:, 6:9]
I = u_test[:, 3:6]
x = TESTDATA['x']
with open(path_to_data + 'SIMULATION_DATA.pkl', 'rb') as f:
    data = pkl.load(f)
R = data['R_st']
lambda_dq0 = (V.T - np.dot(R, I.T)).T
Torque = lambda_dq0[:, 0]*x[:, 1] - lambda_dq0[:, 1]*x[:, 0]  # lambda_d * i_q - lambda_q * i_d


threshold = 1e-4
optimizer = ps.SR3(thresholder="l1", threshold=threshold)
library = ps.PolynomialLibrary(degree=2, include_interaction=True)

model = ps.SINDy(optimizer=optimizer, feature_library=library)
model.fit(x_train, u=u_train, t=None, x_dot=T_train)

model.print()

T_test = TESTDATA['T_em']
x_test = TESTDATA['x']
u_test = TESTDATA['u']
t = TESTDATA['t']

T_test_predicted = model.predict(x_test, u_test)

plt.xlabel(r"$t$")
plt.ylabel(r"$T_{em}$ (Newton-meters)")
plt.plot(t, T_test_predicted)
plt.plot(t, T_test, 'k--')
plt.title('Predicted vs test Torque on test set V = ' + str(TESTDATA['V']))

plt.plot(t, Torque, 'r--')
plt.legend([r"Predicted by SINDYc", r"Reference",r"Simplified Torque model"])

plt.figure() #dont plot first point
plt.semilogy(t[1:], np.abs(T_test_predicted-T_test)[1:])
plt.semilogy(t[1:], np.abs(Torque.reshape(T_test.shape) - T_test)[1:], 'r--')
plt.legend([r"Predicted by SINDYc", r"Simplified Torque model"])
plt.title(r'Absolute error compared to Reference')
plt.xlabel(r"$t$")
plt.ylabel(r"$\log_{10}|T_{em} - T_{em,ref}|$")
plt.show()

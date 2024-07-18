import matplotlib.pyplot as plt

from source import *
from prepare_data import *
path_to_data_files = 'C:/Users/emmav/PycharmProjects/SINDY_project/data/data_files'

# get the data
xdot_train, x_train, u_train, xdot_val, x_val, u_val, TESTDATA = prepare_data()

# Fit the model
threshold = 0.0001 #todo CONSIDER THIS TO BE OPTIMISED
optimizer = ps.SR3(thresholder="l0", threshold=threshold)

library = ps.PolynomialLibrary(degree=2, include_interaction=True)

model = ps.SINDy(optimizer=optimizer, feature_library=library)

model.fit(x_train, u=u_train, t=None, x_dot=xdot_train)

model.print()
# so: x = i, u0_2 = v, u3_5 = I, u6_8 = V, u_9 = theta, u_10 = omega

#todo: use validation data for optimising the threshold


# generate the model on the testdata

xdot_test = TESTDATA['xdot']
x_test = TESTDATA['x']
u_test = TESTDATA['u']
t = TESTDATA['t']

# Predict derivatives using the learned model
x_dot_test_predicted = model.predict(x_test, u_test)

plt.xlabel("$t$")
plt.ylabel("$\dot{x}$")
plt.plot(t, x_dot_test_predicted)
plt.plot(t, xdot_test, 'k--')
plt.legend(["$\partial_t{i_d}$", "$\partial_t{i_q}$", "$\partial_t{i_0}$", "computed"])
plt.title('Predicted vs computed derivatives on test set V = '+str(TESTDATA['V']))
# plt.ylim([x_dot_test_computed.min(), x_dot_test_computed.max()])
plt.show()



# todo torque
# todo add non linear to immec model (and try to solve that with sindy)
# todo add static ecc
# todo add dynamic ecc

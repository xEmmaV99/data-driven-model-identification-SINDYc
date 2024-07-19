from source import *
from prepare_data_copy import *

path_to_data = 'C:/Users/emmav/PycharmProjects/SINDY_project/data/data_files'



#temporary, to test with negative torque
use_test_negative_torque = True
if use_test_negative_torque:
    path_to_testdata  = 'C:/Users/emmav/PycharmProjects/SINDY_project/data/IMMEC_history_soltol_-400.0V_1.0sec.pkl'
    T_train, x_train, u_train, T_val, x_val, u_val, TESTDATA = prepare_data(Torque=True, path_to_test_file=path_to_testdata)
else:
    T_train, x_train, u_train, T_val, x_val, u_val, TESTDATA = prepare_data(Torque=True)

threshold = 0.0001
optimizer = ps.SR3(thresholder="l1", threshold=threshold)
library = ps.PolynomialLibrary(degree=2, include_interaction=True)

model = ps.SINDy(optimizer=optimizer, feature_library=library)
model.fit(x_train, u=u_train, t=None, x_dot = T_train)

model.print()

T_test = TESTDATA['T_em']
x_test = TESTDATA['x']
u_test = TESTDATA['u']
t = TESTDATA['t']

T_test_predicted = model.predict(x_test,u_test)

plt.xlabel("$t$")
plt.ylabel("$T_{em}$ (Newton-meters)")
plt.plot(t,T_test_predicted)
plt.plot(t, T_test, 'k--')
plt.legend(["Predicted","Reference"])
plt.title('Predicted vs test Torque on test set V = '+str(TESTDATA['V']))
plt.show()
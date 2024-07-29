import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from libs import *
from source import *
from prepare_data_old_data import *

path_to_data_files = 'C:/Users/emmav/PycharmProjects/SINDY_project/data/07-24-default-5e-5'

# get the data
xdot_train, x_train, u_train, xdot_val, x_val, u_val, TESTDATA = prepare_data(path_to_data_files, t_end = 1.0, number_of_trainfiles=40)

# Fit the model
library = ps.PolynomialLibrary(degree=2, include_interaction=True)
#library = get_custom_library_funcs("exp")
threshold = 0.0

opt = ps.SR3(thresholder="l1", threshold=threshold, nu =1000)
#opt = Lasso(alpha = threshold)

print("starting model")
model2 = ps.SINDy(optimizer=opt, feature_names= TESTDATA['feature_names'], feature_library=library)
model2.fit(x_train, u=u_train, t=None, x_dot=xdot_train)
print(mean_squared_error(xdot_train, model2.predict(x_train, u=u_train)))
'''model2.print()
plot_coefs2(model2, log=True)


# Predict
x_test_pred = model2.predict(TESTDATA['x'], u=TESTDATA['u'])
plt.figure()
plt.plot(TESTDATA['t'], x_test_pred)
plt.plot(TESTDATA['t'], TESTDATA['xdot'], 'k:')
plt.show()
'''

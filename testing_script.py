import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import mean_squared_error

from source import *
from prepare_data import *


def error_value(model, x_val, u_val, xdot_val):  # The theshold = 0 is a minumum, less sparcity
    sparsity = np.count_nonzero(model.coefficients()) / (model.coefficients().shape[0]*model.coefficients().shape[1]) #percentage of non-zero elements

    mse = model.score(x_val, u = u_val, x_dot = xdot_val, metric = mean_squared_error)
    return mse, sparsity  # add penalty for sparsity

path_to_data_files = 'C:/Users/emmav/PycharmProjects/SINDY_project/data/Combined/07-20-load'

# get the data
xdot_train, x_train, u_train, xdot_val, x_val, u_val, TESTDATA = prepare_data(path_to_data_files, t_end = 2.5)

# Fit the model
library = ps.PolynomialLibrary(degree=2, include_interaction=True)

threshold = 0.0
threshold = 1

opt = ps.SR3(thresholder="l1", threshold=threshold)
#opt = Lasso(alpha=threshold, fit_intercept=False)
print("starting model")
model = ps.SINDy(optimizer=opt, feature_names=['i_d', 'i_q', 'i_0'] + TESTDATA['u_names'], feature_library=library)
model.fit(x_train, u=u_train, t=None, x_dot=xdot_train)
model.print()
print(model.coefficients()) #model object cannot be pickled
#plot_coefs2(model)

path = 'C:/Users/emmav/PycharmProjects/SINDY_project/models/test.pkl'
save_model_coef(model, 'test')
temp_coefs = model.coefficients()

new_model = ps.SINDy(optimizer= opt, feature_names=['i_d', 'i_q', 'i_0'] + TESTDATA['u_names'], feature_library=library)
new_model.optimizer.coef_ = model.coefficients()

new_model.predict(x_val, u_val)
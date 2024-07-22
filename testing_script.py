import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import mean_squared_error

from source import *
from prepare_data import *


def error_value(model, x_val, u_val, xdot_val):  # The theshold = 0 is a minumum, less sparcity
    sparsity = np.count_nonzero(model.coefficients()) / (model.coefficients().shape[0]*model.coefficients().shape[1]) #percentage of non-zero elements

    mse = mean_squared_error(model.predict(x_val, u_val), xdot_val) # this is shuffeled...
    return mse, sparsity  # add penalty for sparsity


path_to_data_files = 'C:/Users/emmav/PycharmProjects/SINDY_project/data/data_files'

# get the data
xdot_train, x_train, u_train, xdot_val, x_val, u_val, TESTDATA = prepare_data(path_to_data_files)

# Fit the model
library = ps.PolynomialLibrary(degree=2, include_interaction=True)

threshold_list = np.linspace(1e-5,0.1, 5)

coefs = []
errorlist = []
spar = []
# rmse = mean_squared_error(x_train, np.zeros(x_train.shape), squared=False)
for i, threshold in enumerate(threshold_list):
    print(i)
    opt = ps.SR3(thresholder="l1", threshold=threshold)
    model = ps.SINDy(optimizer=opt)
    model.fit(x_train, u=u_train, t=None, x_dot=xdot_train)
    coefs.append(model.coefficients())
    err, sp = error_value(model, x_val, u_val, xdot_val)
    errorlist.append(err)
    spar.append(sp)

#model.score()
#plt.scatter(threshold_list,errorlist)
#plt.show()



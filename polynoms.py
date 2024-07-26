from prepare_data import *

path_to_data_files = 'C:/Users/emmav/PycharmProjects/SINDY_project/data/07-24-default-5e-5'

# get the data
xdot_train, x_train, u_train, xdot_val, x_val, u_val, TESTDATA = prepare_data(path_to_data_files, t_end = 1.0, number_of_trainfiles=40)

# Fit the model
library = get_custom_library_funcs('exp')
threshold = 0.0

opt = ps.SR3(thresholder="l1", threshold=threshold)


print("starting model")
model = ps.SINDy(optimizer=opt, feature_names=['i_d', 'i_q', 'i_0'] + TESTDATA['u_names'], feature_library=library)
model.fit(x_train, u=u_train, t=None, x_dot=xdot_train)
model.print()
plot_coefs2(model, log=False)


# Predict
x_test_pred = model.predict(TESTDATA['x'], u=TESTDATA['u'])
plt.figure()
plt.plot(TESTDATA['t'], x_test_pred)
plt.plot(TESTDATA['t'], TESTDATA['xdot'], 'k:')
plt.show()


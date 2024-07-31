from prepare_data import *
path_to_data_files = 'C:/Users/emmav/PycharmProjects/SINDY_project/data/Combined/07-20-load'

# make a model
xdot_train, x_train, u_train, xdot_val, x_val, u_val, TESTDATA = prepare_data(path_to_data_files, t_end = 2.5)

# Fit the model
library = ps.PolynomialLibrary(degree=2, include_interaction=True)
threshold = 0.0
opt = ps.SR3(thresholder="l1", threshold=threshold)
print("Starting model")
model2 = ps.SINDy(optimizer=opt, feature_names=['i_d', 'i_q', 'i_0'] + TESTDATA['u_names'], feature_library=library)
model2.fit(x_train, u=u_train, t=None, x_dot=xdot_train)
model2.print()
print("Model done")

u_test = TESTDATA['u']
x_test = TESTDATA['x']
xdot_test = TESTDATA['xdot']

path = 'C:/Users/emmav/PycharmProjects/SINDY_project/models/test.pkl'
save_model(model2, 'model2')

new_model = load_model('model2')

oldmod = model2.predict(x_test, u=u_test)
newmod = new_model.predict(x_test, u=u_test)


plt.plot(xdot_test, 'k')
plt.plot(newmod, 'r')
plt.plot(oldmod, 'b--')
plt.show()

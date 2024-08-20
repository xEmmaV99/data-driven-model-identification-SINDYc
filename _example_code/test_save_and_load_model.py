import os.path
from source import *
from prepare_data import *
path_to_data_files = os.path.join(os.path.dirname(os.getcwd()), 'train-data', '08-16', 'IMMEC_dynamic_nonlinear_5.0sec.npz')

# make a model
DATA = prepare_data(path_to_data_files)

# Fit the model
library = get_custom_library_funcs('poly_2nd_order')
threshold = 0.0
opt = ps.SR3(thresholder="l1", threshold=threshold)
print("Starting model")
model2 = ps.SINDy(optimizer=opt, feature_names=DATA['feature_names'], feature_library=library)
model2.fit(DATA['x_train'], u=DATA['u_train'], t=None, x_dot=DATA['xdot_train'])
model2.print()
print("Model done")

# this is not good -> need other file
testpath = os.path.join(os.path.dirname(os.getcwd()), 'test-data', '08-18', 'IMMEC_dynamic_nonlinear_5.0sec.npz')
TEST = prepare_data(testpath, test_data=True)
u_test = TEST['u']
x_test = TEST['x']
xdot_test = TEST['xdot']


save_name = save_model(model2, 'model2', libstr='poly_2nd_order')

new_model = load_model(save_name)
oldmod = model2.predict(x_test, u=u_test)
newmod = new_model.predict(x_test, u=u_test)


plt.plot(xdot_test, 'k')
plt.plot(newmod, 'r')
plt.plot(oldmod, 'b--')
plt.show()

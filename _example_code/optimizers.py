from source import *
from prepare_data import *
# testing different optimisers

cwd = os.getcwd()
path_to_data = os.path.join(cwd, 'data', '07-20-small-dt')

# check for MSE, sparsity and computation time 
# methods, SR3 (l0,l1 and l2), SQTL, SRR and LASSO
V_test = 219

number_of_models = 40
thesholds = np.logspace(-2,2,number_of_models)

xdot_train, x_train, u_train, xdot_val, x_val, u_val, TESTDATA = prepare_data(path_to_data, 
                                                                              V_test_data=V_test, 
                                                                              t_end=1.0, 
                                                                              number_of_trainfiles=10)
meth_list = ["SR3_l1", "LASSO"] #"SR3_l0", "SR3_l2", "SRR", "SQLRNTS"

models = []
for method in meth_list:
    models.append(theshold_search(threshold_array=thesholds, 
                    train_and_validation_data=[xdot_train, x_train, u_train, xdot_val, x_val, u_val],
                    method = method, name = method+"V_test_230", plot_now = False))


names = 'MSE_sparsity_vs_threshold_'
for method in meth_list:
    p = os.path.join(cwd, 'plot_data', names + method + 'V_test_'+str(V_test)+'.pkl')
    plot_data(p, show = False, figure = False)
plt.show()


# CHECK THE BEST MODEL ON THE TESTDATA
xdot_test = TESTDATA['xdot']
x_test = TESTDATA['x']
u_test = TESTDATA['u']
t = TESTDATA['t']
for i, model in enumerate(models):
    # Predict derivatives using the learned model
    x_dot_test_predicted = model.predict(x_test, u_test)
    plt.plot(t, x_dot_test_predicted)
    plt.plot(t, xdot_test, 'k--')
    plt.title(meth_list[i])
plt.show()
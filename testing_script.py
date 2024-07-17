from source import *
dt = 1e-4
t_end = 1.0

save_path = 'C:/Users/emmav/PycharmProjects/SINDY_project/data/IMMEC_history_40V_'+str(t_end)+'sec' #for conventional naming
load_path = save_path + '.pkl'

x_train, u_train = get_immec_training_data(path_to_data_logger=load_path, use_estimate_for_v=True)

# testing one model
# Fit the model
threshold = 1e-4
optimizer = ps.SR3(thresholder="l0", threshold=threshold)

# library = ps.FourierLibrary(n_frequencies=1) + ps.PolynomialLibrary(degree = 1)
library = ps.PolynomialLibrary(degree=
                               2, include_interaction=True) #+ ps.FourierLibrary()

t_train = np.linspace(0, t_end, x_train.shape[0])  # THIS MIGHT BE OFF BY ONE
model = ps.SINDy(optimizer=optimizer, feature_library=library)
model.fit(x_train, u=u_train, t=t_train)
model.print()
# so: x = i, u0_2 = v, u3_5 = I, u6_8 = V, u_9 = theta, u_10 = omega


# can we compare the derivatives directly?
# model is now trained,CONSIDER NEW DATA, i'll test with training data
t_end_test = t_end

t_test = np.arange(0, t_end_test, dt)
x0_test = x_train[0, :]  # THIS CAN BE CHOSEN, IT IS NOW [0,0,0]
t_test_span = (t_test[0], t_test[-1])

# generate new model to test it on (to do)
x_test = x_train
u_test = u_train # new model!

# Predict derivatives using the learned model
x_dot_test_predicted = model.predict(x_test, u_test)
# Compute derivatives with a finite difference method, for comparison
x_dot_test_computed = model.differentiate(x_test, t=dt) # differentiates the x_test values, by the differentiation method of the model

tvec = np.arange(0,dt*x_test.shape[0],dt) # corresponding time vector
plt.xlabel("$t$")
plt.ylabel("$\dot{x}$")
plt.plot(tvec, x_dot_test_predicted)
plt.plot(tvec, x_dot_test_computed, 'k--')
plt.legend(["$\partial_t{i_d}$", "$\partial_t{i_q}$", "$\partial_t{i_0}$", "computed"])
plt.ylim([-5000,5000])
plt.show()
import matplotlib.pyplot as plt
import numpy as np

from source import *

# Generate training data
# From MCC model: find dx/dt

dt = 1e-4
t_end = 0.5
motor_path = 'C:/Users/emmav/PycharmProjects/SINDY_project/Cantoni.pkl'
save_path = 'C:/Users/emmav/PycharmProjects/SINDY_project/data/IMMEC_history_train_0.5sec.pkl'

#create_and_save_immec_data(mode='linear', timestep=dt, t_end=t_end, path_to_motor=motor_path, save_path= save_path)

x_train, u_train = get_immec_training_data(timestep=dt, check_training_data=False, path_to_data_logger=save_path)

# Fit the model
threshold = 1e-4
optimizer = ps.SR3(thresholder="l1", threshold=threshold)

#library = ps.FourierLibrary(n_frequencies=1) + ps.PolynomialLibrary(degree = 1)
library =  ps.PolynomialLibrary(degree =2, include_interaction = True)


t_train = np.linspace(0,t_end, x_train.shape[0]) # THIS MIGHT BE OFF BY ONE
model = ps.SINDy(optimizer=optimizer, feature_library=library)
model.fit(x_train, u=u_train, t=t_train)
model.print()
# so: x = i, u0_2 = v, u3_5 = I, u6_8 = V, u_9 = theta, u_10 = omega




# can we compare the derivatives directly?
# model is now trained,CONSIDER NEW DATA, i'll test with training data
t_end_test = t_end

t_test = np.arange(0, t_end_test, dt)
x0_test = x_train[0,:] # THIS CAN BE CHOSEN, IT IS NOW [0,0,0]
t_test_span = (t_test[0], t_test[-1])
x_test = x_train
u_test = u_train # new model!

# Predict derivatives using the learned model
x_dot_test_predicted = model.predict(x_test, u_test)

# Compute derivatives with a finite difference method, for comparison
x_dot_test_computed = model.differentiate(x_test, t=dt)

plt.plot(x_dot_test_predicted)
plt.plot(x_dot_test_computed, 'k--')
plt.show()
plt.legend(["di_d","di_q","di_0","ref"])


simulate_in_time = True # this takes really long to simulate!
if simulate_in_time:
    plt.figure()
    #how to check this? Simulate in time
    print("starting simulation")
    # simulate the testing data, sould be equal
    x_sim = model.simulate(x_train[0,:], u=u_train, t=t_train, integrator="odeint") # why is this one shorter?
    print("ended simulation")
    
    plt.plot(t_train[:-1], x_sim)
    plt.plot(t_train, x_train, 'k--')
    plt.legend(["i_d","i_q","i_0","ref"])
    plt.ylim([-20,20])
    plt.show()





# todo validation
# todo torque
# todo add non linear to immec model (and try to solve that with sindy
# todo add static ecc
# todo add dynamic ecc

import matplotlib.pyplot as plt
import numpy as np

from source import *

# Generate training data
# From MCC model: find dx/dt

dt = 1e-4
t_end = 1e-3
motor_path = 'C:/Users/emmav/PycharmProjects/SINDY_project/Cantoni.pkl'
save_path = 'C:/Users/emmav/PycharmProjects/SINDY_project/data/IMMEC_history_unnamed.pkl'


create_and_save_immec_data(mode='linear', timestep=dt, t_end=t_end, path_to_motor=motor_path, save_path= save_path)

x_train, u_train = get_immec_training_data(timestep=dt, check_training_data=False, path_to_data_logger=save_path)

# Fit the model
threshold = 1e-5
optimizer = ps.SR3(thresholder="l1", threshold=threshold)

library = ps.FourierLibrary() + ps.PolynomialLibrary(degree = 3)

t_train = np.arange(0,t_end-dt,dt) # THIS MIGHT BE OFF BY ONE
model = ps.SINDy(optimizer=optimizer, feature_library=library)
model.fit(x_train, u=u_train, t=t_train)
model.print()
# so: x = i, u0_2 = v, u3_5 = I, u6_8 = V, u_9 = theta, u_10 = omega

#how to check this?

do_simulation = False
if do_simulation:
    print("starting simulation")
    x_sim = model.simulate(x_train[0,:], u=u_train, t=t_train, integrator="odeint") # why is this one shorter?
    print("ended simulation")

    plt.plot(t_train[:-1], x_sim)
    plt.plot(x_train, 'k--')
    plt.legend(["i_a","i_b","i_c","ref"])
    plt.show()







# todo validation
# todo torque
# todo add static ecc
# todo add dynamic ecc

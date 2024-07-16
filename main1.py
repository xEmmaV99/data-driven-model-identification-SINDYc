import numpy as np

from source import *

# Generate training data
# From MCC model: find dx/dt

dt = 1e-4
t_end = 1.0
path = 'C:/Users/emmav/PycharmProjects/SINDY_project/Cantoni.pkl'
logger_path = 'C:/Users/emmav/PycharmProjects/SINDY_project/data/IMMEC_history_1sec.pkl'
x_train, u_train = get_immec_training_data(mode='linear', timestep=dt, t_end=t_end, path_to_motor=path,
                                           check_training_data=False, load_from_file=True, path_to_data_logger=logger_path)


# Fit the model
threshold = 0.05
optimizer = ps.SR3(thresholder="l1", threshold=threshold)

model = ps.SINDy(optimizer=optimizer, feature_library=ps.PolynomialLibrary(degree = 2))
model.fit(x_train, u=u_train, t=dt)
model.print()
# so: x = i, u0_2 = v, u3_5 = I, u6_8 = V, u_9 = theta, u_10 = omega

#how to check this?
t_train = np.arange(0,t_end-dt,dt) # THIS MIGHT BE OFF BY ONE
x_sim = model.simulate(x_train[1,:], u=u_train, t=t_train)
plot_kws = dict(linewidth=2)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].plot(t_train, x_train[:, 0], "r", label="$x_0$", **plot_kws)
axs[0].plot(t_train, x_train[:, 1], "b", label="$x_1$", alpha=0.4, **plot_kws)
axs[0].plot(t_train, x_sim[:, 0], "k--", label="model", **plot_kws)
axs[0].plot(t_train, x_sim[:, 1], "k--")
axs[0].legend()
axs[0].set(xlabel="t", ylabel="$x_k$")

axs[1].plot(x_train[:, 0], x_train[:, 1], "r", label="$x_k$", **plot_kws)
axs[1].plot(x_sim[:, 0], x_sim[:, 1], "k--", label="model", **plot_kws)
axs[1].legend()
axs[1].set(xlabel="$x_1$", ylabel="$x_2$")
plt.show()

# Simulate and plot the results
'''
x_sim = model.simulate(x0_train, t_train)
plot_kws = dict(linewidth=2)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].plot(t_train, x_train[:, 0], "r", label="$x_0$", **plot_kws)
axs[0].plot(t_train, x_train[:, 1], "b", label="$x_1$", alpha=0.4, **plot_kws)
axs[0].plot(t_train, x_sim[:, 0], "k--", label="model", **plot_kws)
axs[0].plot(t_train, x_sim[:, 1], "k--")
axs[0].legend()
axs[0].set(xlabel="t", ylabel="$x_k$")

axs[1].plot(x_train[:, 0], x_train[:, 1], "r", label="$x_k$", **plot_kws)
axs[1].plot(x_sim[:, 0], x_sim[:, 1], "k--", label="model", **plot_kws)
axs[1].legend()
axs[1].set(xlabel="$x_1$", ylabel="$x_2$")
plt.show()
'''

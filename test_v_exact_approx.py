from prepare_data import prepare_data
import os
import matplotlib.pyplot as plt
import numpy as np

# why is estimated better ? huh?

path = os.path.join(os.getcwd(), "test-data", "07-29-default", "IMMEC_0ecc_5.0sec.npz")
# path = os.path.join(os.getcwd(), "test-data","07-29-nonlin-nonzero-initload","IMMEC_nonlin_0ecc_5.0sec.npz")
path = os.path.join(os.getcwd(), "test-data", "07-31-50ecc-load", "IMMEC_50ecc_5.0sec.npz")

data_real = prepare_data(path, test_data=True, use_estimate_for_v=False)
data_esti = prepare_data(path, test_data=True, use_estimate_for_v=True)

v_real = data_real["u"][:, :3]
v_esti = data_esti["u"][:, :3]

fig, ax = plt.subplots()
l1, _,_ = ax.plot(data_real['t'][:,:,0],v_real, 'k--')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Voltage [V]')
l2, _,_ = ax.plot(data_esti['t'][:,:,0],v_esti, ':')
ax.legend([l1,l2],["Exact calculation", "Approx from line voltages"])
ax.set_xlim(3.15, 3.40)

left, bottom, width, height = [0.2, 0.6, 0.25, 0.25]
ax_inset = fig.add_axes([left, bottom, width, height])
# Plot the zoomed-in data
ax_inset.plot(data_real['t'][:,:,0],v_real, 'k--')
ax_inset.plot(data_esti['t'][:,:,0],v_esti, ':')
ax_inset.set_xlim(3.25, 3.30)  # adjust limits of the inset
ax_inset.set_ylim(-1.2, 1.2)



v_diff = np.abs(v_real - v_esti)
plt.figure()
plt.semilogy(v_diff)
plt.show()

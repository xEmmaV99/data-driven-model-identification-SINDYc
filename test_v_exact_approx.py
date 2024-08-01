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

plt.figure()
plt.plot(v_real, 'k--')
plt.plot(v_esti, ':')
plt.legend(["exact", "", "", "from line"])
print(np.max(v_real[:,-1]))

v_diff = np.abs(v_real - v_esti)
plt.figure()
plt.semilogy(v_diff)
plt.show()

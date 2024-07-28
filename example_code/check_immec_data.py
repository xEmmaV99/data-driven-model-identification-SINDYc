import matplotlib.pyplot as plt
import numpy as np

from source import *

"""
path = 'C:/Users/emmav/PycharmProjects/SINDY_project/test-data/07-24/IMMEC_history_40V_5.0sec.pkl'

with open(path, 'rb') as f:
    dataset = pkl.load(f)

plt.plot(dataset['time'],dataset['omega_rot'])
plt.title("omega_rot")
plt.figure()
plt.plot(dataset['time'],dataset['i_st'])
plt.title("i_st")
plt.figure()
plt.plot(dataset['time'],dataset['T_em'])
plt.title("T_em")
plt.show()
"""
path = "C:/Users/emmav/PycharmProjects/SINDY_project/test-data/07-27/IMMEC_0ecc_5.0sec.npz"
dataset = np.load(path)

plt.subplot(2, 3, 1)
plt.title("omega_rot"), plt.xlabel("time (s)")
plt.plot(dataset["time"], dataset["omega_rot"])

plt.subplot(2, 3, 2)
plt.title("i_st"), plt.xlabel("time (s)")
plt.plot(dataset["time"], dataset["i_st"])

plt.subplot(2, 3, 3)
plt.title("T_em"), plt.xlabel("time (s)")
plt.plot(dataset["time"], dataset["T_em"])

plt.subplot(2, 3, 4)
plt.title("T_l"), plt.xlabel("time (s)")
plt.plot(dataset["time"], dataset["T_l"])

plt.subplot(2, 3, 5)
plt.title("V"), plt.xlabel("time (s)")
plt.plot(dataset["time"], dataset["v_applied"])

# Add padding so title and labels dont overlap
plt.tight_layout()

plt.show()

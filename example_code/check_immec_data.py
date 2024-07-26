import matplotlib.pyplot as plt
import numpy as np

from source import *
'''
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
'''
path = 'C:/Users/emmav/PycharmProjects/SINDY_project/test-data/07-26/IMMEC_0ecc_1.0sec.npz'
dataset = np.load(path)
plt.plot(dataset['time'],dataset['omega_rot'])
plt.title("omega_rot")
plt.figure()
plt.plot(dataset['time'],dataset['i_st'])
plt.title("i_st")
plt.figure()
plt.plot(dataset['time'],dataset['T_em'])
plt.title("T_em")
plt.show()
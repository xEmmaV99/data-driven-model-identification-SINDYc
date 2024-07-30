import os.path

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
#path = "C:/Users/emmav/PycharmProjects/SINDY_project/test-data/07-27/IMMEC_0ecc_5.0sec-nosweep.npz"
#path = "C:/Users/emmav/PycharmProjects/SINDY_project/test-data/07-27/IMMEC_0ecc_5.0sec-nosweep.npz"
path = "C:/Users/emmav/PycharmProjects/SINDY_project/test-data/07-29/IMMEC_0ecc_3.0sec.npz"
path = "C:/Users/emmav/PycharmProjects/SINDY_project/test-data/07-29/IMMEC_0ecc_1.6sec.npz"
#path = os.path.join(os.path.dirname(os.getcwd()), 'train-data', '07-25', 'IMMEC_0ecc_1.0sec.npz')
path = 'C:\\Users\\emmav\\PycharmProjects\\SINDY_project\\train-data\\07-29\\IMMEC_0ecc_5.0sec.npz'
dataset = dict(np.load(path))
d_air = 0.000477 # for the Cantoni motor

testdata = False
traindata = True
if testdata:
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

if traindata:
    simulation_number = 30
    plt.subplot(2, 3, 1)
    plt.title("omega_rot"), plt.xlabel("time (s)"), plt.ylabel("rad/s")
    plt.plot(dataset["time"][:,0,simulation_number], dataset["omega_rot"][:,0,simulation_number])

    plt.subplot(2, 3, 2)
    plt.title("i_st"), plt.xlabel("time (s)"), plt.ylabel("A/m") #debug
    plt.plot(dataset["time"][:,0,simulation_number], dataset["i_st"][:,:,simulation_number])

    plt.subplot(2, 3, 3)
    plt.title("T_l and T_em"), plt.xlabel("time (s)"), plt.ylabel("Nm")
    #plt.title("T_em"), plt.xlabel("time (s)")
    plt.plot(dataset["time"][:,0,simulation_number], dataset["T_em"][:,0,simulation_number])

    #plt.subplot(2, 3, 3)
    #plt.title("T_l"), plt.xlabel("time (s)")
    plt.plot(dataset["time"][:,0,simulation_number], dataset["T_l"][:,0,simulation_number], 'k--')
    plt.legend(["T_em", "T_l"])

    plt.subplot(2, 3, 5)
    plt.title("V"), plt.xlabel("time (s)"), plt.ylabel("V")
    plt.plot(dataset["time"][:,0,simulation_number], dataset["v_applied"][:,:,simulation_number])

    plt.subplot(2,3,4)
    plt.title("UMP"), plt.xlabel("time (s)"), plt.ylabel("N")
    plt.plot(dataset["time"][:,0,simulation_number], dataset["F_em"][:,:,simulation_number])

    plt.subplot(2,3,6)
    plt.title("Eccentricity"), plt.xlabel("time (s)"), plt.ylabel("% airgap")
    plt.plot(dataset["time"][:, 0, simulation_number], dataset["ecc"][:, 0, simulation_number] / d_air)

    # Add padding so title and labels dont overlap
    plt.tight_layout()
    plt.show()


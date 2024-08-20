import os.path

import matplotlib.pyplot as plt
import numpy as np

from source import *

path = os.path.join(os.path.dirname(os.getcwd()), 'train-data', "07-31-nonlin","IMMEC_nonlinear-0ecc_5.0sec.npz")
#path = os.path.join(os.path.dirname(os.getcwd()), 'train-data', "07-29-nonlin","IMMEC_nonlin_0ecc_5.0sec.npz")
plot_immec_data(path, 10)

path = os.path.join(os.path.dirname(os.getcwd()), 'test-data', "08-19","IMMEC_dynamic_linear_5.0sec.npz")
#path = os.path.join(os.path.dirname(os.getcwd()), 'test-data','08-06', 'IMMEC_nonlin_0ecc_5.0sec.npz')
plot_immec_data(path)

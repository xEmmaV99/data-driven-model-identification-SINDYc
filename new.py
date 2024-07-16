import matplotlib.pyplot as plt
import numpy as np

from source import *

# Generate training data
# From MCC model: find dx/dt

dt = 1e-4
t_end = 3
motor_path = 'C:/Users/emmav/PycharmProjects/SINDY_project/Cantoni.pkl'
save_path = 'C:/Users/emmav/PycharmProjects/SINDY_project/data/IMMEC_history_3sec.pkl'

create_and_save_immec_data(mode='linear', timestep=dt, t_end=t_end, path_to_motor=motor_path, save_path= save_path)
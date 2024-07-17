from source import *

path_to_data_files = 'C:\Users\emmav\PycharmProjects\SINDY_project\data\data_files'

# specific data values
V_range = np.linspace(40, 400, 10)
t_end = 1.0

number_of_samples = V_range.shape[0]  # last index is nmbr_samples -1

#np.random.shuffle() only shuffles along the rows

np.random.shuffle()


# todo devide datasets into train, validation and testdata and shuffle

# todo torque
# todo add non linear to immec model (and try to solve that with sindy)
# todo add static ecc
# todo add dynamic ecc
